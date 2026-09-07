"""Unconditional joint-Flow inference under the HybridOptimization problem."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from pymatgen.core import Structure

from ._io import atomic_json, sha256_file
from .constraints import evaluate_feasibility
from .differentiable_uma_hull import DifferentiableUMAHull
from .generate import load_model, sample_atom_counts, variable_random_source
from .hybrid_adam import HybridAdamConfig, HybridProjectedAdamSolver
from .hybrid_lbfgs import (
    HybridLBFGSConfig,
    UnconditionalHybridLBFGSSolver,
)
from .hybrid_ipopt import HybridIPOPTConfig, HybridIPOPTSolver
from .novelty import NoveltyIndex, crystal_descriptor
from .sample import decode_terminal, integrate_flow
from .spec import DEFAULT_SPEC
from .uma_guidance import (
    load_uma_calculator_vjp_factory,
    load_uma_native_second_order_evaluator,
)

ATOMIC_NUMBERS = torch.tensor((11, 26, 15, 8), dtype=torch.long)


def _state(state) -> dict:
    return {
        "atom": state.atom.detach().cpu().tolist(),
        "frac": state.frac.detach().cpu().tolist(),
        "lattice": state.lattice.detach().cpu().tolist(),
    }


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, checkpoint = load_model(args.joint_checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    histogram = checkpoint["n_histogram"].clone().to(device)
    histogram[: DEFAULT_SPEC.n_primitive_min] = 0
    if args.max_atoms + 1 < histogram.numel():
        histogram[args.max_atoms + 1 :] = 0
    n_atoms = sample_atom_counts(histogram, 1, generator)
    initial_source, mask = variable_random_source(
        n_atoms, model.config.vocab_size, device, generator
    )
    with torch.no_grad():
        unconditional_terminal = integrate_flow(
            model, initial_source, mask, steps=args.ode_steps,
            method="midpoint", freeze_atom=False,
        )
        base_types, base_frac, base_lattice = decode_terminal(
            model, unconditional_terminal, mask
        )
    index_to_element = {
        int(index): element
        for element, index in checkpoint["element_to_index"].items()
    }

    def decode(types: torch.Tensor) -> list[str]:
        return [index_to_element[int(value)] for value in types[0, mask[0]].cpu()]

    baseline_elements = decode(base_types)
    baseline_exact = evaluate_feasibility(
        baseline_elements, base_frac[0].cpu().numpy(),
        base_lattice[0].cpu().numpy(), None,
    )
    hull = DifferentiableUMAHull.from_cache(args.uma_hull_cache).to(device)
    novelty = NoveltyIndex.load(args.novelty_index, device)

    def exact_callback(types, frac, lattice, e_hull: float) -> bool:
        report = evaluate_feasibility(
            decode(types), frac[0].cpu().numpy(), lattice[0].cpu().numpy(), e_hull
        )
        return bool(report["feasible"])

    if args.solver == "adam":
        evaluator, uma_metadata = load_uma_calculator_vjp_factory(
            args.uma_checkpoint, device=args.device, task_name=args.task_name
        )
        config = HybridAdamConfig(
            max_iterations=args.max_iterations,
            ode_steps=args.ode_steps,
            learning_rate_atom=args.adam_lr_atom,
            learning_rate_frac=args.adam_lr_frac,
            learning_rate_lattice=args.adam_lr_lattice,
            gradient_clip_norm=args.gradient_clip_norm,
            augmented_lagrangian_rho=args.al_rho,
            dual_learning_rate=args.dual_learning_rate,
            consecutive_feasible_required=args.consecutive_feasible,
            hull_feasibility_only=args.hull_feasibility_only,
        )
        solver = HybridProjectedAdamSolver(
            model, mask, evaluator, hull, ATOMIC_NUMBERS.to(device),
            exact_callback,
        )
    elif args.solver == "ipopt":
        evaluator, uma_metadata = load_uma_native_second_order_evaluator(
            args.uma_checkpoint, device=args.device, task_name=args.task_name
        )
        config = HybridIPOPTConfig(
            max_iterations=args.max_iterations, ode_steps=args.ode_steps,
            constraint_mode=args.constraint_mode,
            hull_objective_weight=args.hull_objective_weight,
            hull_violation_weight=args.hull_violation_weight,
            novelty_weight=args.novelty_weight,
            composition_weight=args.composition_weight,
            print_level=args.ipopt_print_level,
        )
        solver = HybridIPOPTSolver(
            model, mask, evaluator, hull, ATOMIC_NUMBERS.to(device), novelty,
            exact_callback,
        )
    else:
        factory, uma_metadata = load_uma_calculator_vjp_factory(
            args.uma_checkpoint, device=args.device, task_name=args.task_name
        )
        config = HybridLBFGSConfig(
            max_iterations=args.max_iterations,
            history_size=args.history_size,
            learning_rate=args.learning_rate,
            ode_steps=args.ode_steps,
            hull_weight=args.hull_objective_weight,
            hull_violation_weight=args.hull_violation_weight,
            novelty_weight=args.novelty_weight,
            composition_weight=args.composition_weight,
            hull_feasibility_only=args.hull_feasibility_only,
        )
        solver = UnconditionalHybridLBFGSSolver(
            model, mask, factory, hull, ATOMIC_NUMBERS.to(device), novelty,
            exact_callback,
        )
    result = solver.solve(initial_source, config)
    with torch.no_grad():
        decoded_types, frac, lattice = decode_terminal(model, result.terminal, mask)
        # In the hull-only feasibility experiment the discrete composition is
        # the unconditional endpoint assignment and is held fixed.  All three
        # source tensors remain continuous optimization variables, but z_A can
        # only influence the coupled X/L trajectory; it does not redefine A.
        types = base_types if args.hull_feasibility_only else decoded_types
        final_elements = baseline_elements if args.hull_feasibility_only else decode(types)
        descriptor = crystal_descriptor(types, frac, lattice, mask)
        novelty_value, nearest = novelty.score(descriptor)
    selected = result.selected_values
    final_exact = evaluate_feasibility(
        final_elements, frac[0].cpu().numpy(), lattice[0].cpu().numpy(),
        selected["E_hull_eV_atom"],
    )
    relaxation_ok = bool(
        selected["Fmax_eV_A"] <= 0.01
        and selected["stress_fro_eV_A3"] <= 0.10 / 160.21766208
    )
    complete = bool(final_exact["feasible"] and relaxation_ok)
    structure = Structure(
        lattice[0].cpu().numpy(), final_elements, frac[0].cpu().numpy(),
        coords_are_cartesian=False,
    )
    out = Path(args.out_directory)
    out.mkdir(parents=True, exist_ok=True)
    if complete:
        structure.to(filename=out / "candidate.cif")
    if args.solver == "ipopt":
        gradient_audit = {
            "flow_mapping": "endpoint = Phi_theta(N, z_A, z_X, z_L); theta frozen",
            "objective_gradients_finite": all(
                torch.isfinite(torch.tensor(row["objective_gradient_norm"]))
                for row in result.history
            ),
            "constraint_jacobians_finite": all(
                torch.isfinite(torch.tensor(row["constraint_jacobian_row_norms"])).all()
                for row in result.history
            ),
            "force_stress_second_derivatives": True,
            "z_A_relaxation": (
                "continuous source logits inside the fixed unconditional endpoint "
                "type region; exact hard UMA species in both value and gradient"
            ),
        }
    elif args.solver == "adam":
        gradient_audit = {
            "flow_mapping": "endpoint = Phi_theta(N, z_A, z_X, z_L); theta frozen",
            "objective_gradients_finite": all(
                torch.isfinite(torch.tensor(row["gradient_norm_pre_clip"]))
                for row in result.history
            ),
            "force_stress_second_derivatives": False,
            "z_A_relaxation": (
                "continuous source logits inside the fixed unconditional endpoint "
                "type region; exact hard UMA species in both value and gradient"
            ),
            "constraint_method": (
                "simultaneous projected primal-dual Adam with augmented Lagrangian"
            ),
        }
    else:
        gradient_audit = {
            "flow_mapping": "endpoint = Phi_theta(N, z_A, z_X, z_L); theta frozen",
            "z_A": {
                "finite": all(torch.isfinite(torch.tensor(row["gradient_norm_z_A"])) for row in result.history),
                "nonzero_observed": any(row["gradient_norm_z_A"] > 0 for row in result.history),
                "limitation": "UMA candidate energy uses hard integer species; z_A receives gradients from other continuous terms",
            },
            "z_X": {"finite_nonzero": any(row["gradient_norm_z_X"] > 0 for row in result.history)},
            "z_L": {"finite_nonzero": any(row["gradient_norm_z_L"] > 0 for row in result.history)},
            "force_stress_second_derivatives": False,
        }
    atomic_json(out / "gradient_audit.json", gradient_audit)
    record = {
        "version": f"unconditional-hybrid-all-uma-{args.solver}-v1",
        "status": "complete" if complete else "constraints_not_satisfied",
        "single_unified_optimization": True,
        "postprocessing_filter": False,
        "endpoint_relaxation": False,
        "seed": args.seed,
        "N_sampled_from_pN": int(n_atoms[0]),
        "N_sampling_support": [DEFAULT_SPEC.n_primitive_min, args.max_atoms],
        "config": asdict(config),
        "solver": args.solver,
        "active_solver_status": result.status,
        "solver_stats": getattr(result, "solver_stats", None),
        "inputs": {
            "joint_checkpoint": str(Path(args.joint_checkpoint).resolve()),
            "joint_checkpoint_sha256": sha256_file(args.joint_checkpoint),
            "uma_checkpoint": str(Path(args.uma_checkpoint).resolve()),
            "uma_checkpoint_sha256": sha256_file(args.uma_checkpoint),
            "uma_hull_cache": str(Path(args.uma_hull_cache).resolve()),
            "uma_hull_cache_sha256": sha256_file(args.uma_hull_cache),
        },
        "uma": uma_metadata,
        "unconditional_generation": {
            "initial_noise": _state(initial_source),
            "endpoint_elements": baseline_elements,
            "endpoint_structure": Structure(
                base_lattice[0].cpu().numpy(), baseline_elements,
                base_frac[0].cpu().numpy(), coords_are_cartesian=False,
            ).as_dict(),
            "exact_without_hull": baseline_exact,
        },
        "optimized_initial_noise": _state(result.source),
        "selected_values": selected,
        "exact_feasibility": final_exact,
        "relaxation_constraints": {
            "passed": relaxation_ok,
            "Fmax_eV_A": selected["Fmax_eV_A"],
            "stress_fro_GPa": selected["stress_fro_eV_A3"] * 160.21766208,
        },
        "novelty": {
            "nearest_descriptor_distance": float(novelty_value[0]),
            "nearest_reference_id": novelty.ids[int(nearest[0])],
        },
        "final_composition": dict(Counter(final_elements)),
        "final_structure": structure.as_dict(),
        "history": result.history,
    }
    atomic_json(out / "result.json", record)
    print(json.dumps({
        "status": record["status"], "out": str(out),
        "N": int(n_atoms[0]), "initial_composition": dict(Counter(baseline_elements)),
        "final_composition": record["final_composition"],
        "E_hull_eV_atom": selected["E_hull_eV_atom"],
        "Fmax_eV_A": selected["Fmax_eV_A"],
        "stress_fro_GPa": record["relaxation_constraints"]["stress_fro_GPa"],
        "exact_feasible": final_exact["feasible"],
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument(
        "--solver", choices=("lbfgs", "ipopt", "adam"), default="ipopt"
    )
    parser.add_argument(
        "--constraint-mode",
        choices=("hull", "hull_force", "hull_force_stress", "full"),
        default="full",
    )
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--uma-hull-cache", required=True)
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--hull-violation-weight", type=float, default=0.0)
    parser.add_argument("--hull-objective-weight", type=float, default=1.0)
    parser.add_argument("--novelty-weight", type=float, default=0.01)
    parser.add_argument("--composition-weight", type=float, default=40.0)
    parser.add_argument("--ipopt-print-level", type=int, default=0)
    parser.add_argument("--adam-lr-atom", type=float, default=1.0e-2)
    parser.add_argument("--adam-lr-frac", type=float, default=5.0e-3)
    parser.add_argument("--adam-lr-lattice", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--al-rho", type=float, default=10.0)
    parser.add_argument("--dual-learning-rate", type=float, default=0.1)
    parser.add_argument("--consecutive-feasible", type=int, default=10)
    parser.add_argument("--hull-feasibility-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

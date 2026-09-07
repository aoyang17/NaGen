"""Unconditional generation followed by one joint nonlinear D-Flow solve."""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ._io import atomic_json, sha256_file, sha256_json
from .calibrated_mp_hull import (
    CalibratedMPHullCache,
    build_calibrated_cache,
)
from .constraints import evaluate_feasibility
from .generate import load_model, sample_unconditional_compositions
from .differentiable_mp_hull import DifferentiableMP2020Hull
from .joint_dflow import (
    JointCrystalProblem,
    JointDFlowConfig,
    JointDFlowSolver,
    model_sha256,
)
from .model import CrystalState
from .novelty import NoveltyIndex
from .sample import decode_terminal, integrate_flow
from .uma_guidance import load_uma_calculator_vjp_factory


ATOMIC_NUMBERS = {"Na": 11, "Fe": 26, "P": 15, "O": 8}


def _state(state: CrystalState) -> dict[str, Any]:
    return {
        "atom": state.atom.detach().cpu().tolist(),
        "frac": state.frac.detach().cpu().tolist(),
        "lattice": state.lattice.detach().cpu().tolist(),
    }


def _exact(
    model, checkpoint: dict[str, Any], terminal: CrystalState,
    mask: torch.Tensor, expected_elements: list[str], e_hull: float,
) -> dict[str, Any]:
    with torch.no_grad():
        types, frac, lattice = decode_terminal(model, terminal, mask)
    inverse = {
        int(index): element
        for element, index in checkpoint["element_to_index"].items()
    }
    decoded = [inverse[int(index)] for index in types[0].cpu().tolist()]
    feasibility = evaluate_feasibility(
        decoded,
        frac[0].cpu().numpy(),
        lattice[0].cpu().numpy(),
        e_hull,
    )
    return {
        "fixed_generated_composition_preserved": decoded == expected_elements,
        "feasibility": feasibility,
        "structure": {
            "elements": decoded,
            "frac_coords": frac[0].cpu().tolist(),
            "lattice": lattice[0].cpu().tolist(),
        },
    }


def _counts(elements: list[str]) -> dict[str, int]:
    counts = Counter(elements)
    return {element: int(counts[element]) for element in ("Na", "Fe", "P", "O")}


def run(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("campaign count must be positive")
    device = torch.device(args.device)
    joint_model, joint_checkpoint = load_model(args.joint_checkpoint, device)
    geometry_model, geometry_checkpoint = load_model(
        args.geometry_checkpoint, device
    )
    if joint_checkpoint["element_to_index"] != geometry_checkpoint["element_to_index"]:
        raise ValueError("joint and geometry checkpoints use different vocabularies")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    # Draw directly from learned p(N,A). Composition constraints are not used
    # here; z_A is optimized jointly with z_X and z_L below.
    generated_types, masks = sample_unconditional_compositions(
        joint_model,
        joint_checkpoint,
        args.count,
        generator,
        args.ode_steps,
        "midpoint",
        max_atoms=args.max_atoms,
    )
    inverse = {
        int(index): element
        for element, index in geometry_checkpoint["element_to_index"].items()
    }
    prepared: list[dict[str, Any]] = []
    for row in range(args.count):
        n_atoms = int(masks[row].sum())
        mask = torch.ones(1, n_atoms, dtype=torch.bool, device=device)
        types = generated_types[row : row + 1, :n_atoms]
        elements = [
            inverse[int(index)]
            for index in types[0, mask[0]].detach().cpu().tolist()
        ]
        n_atoms = len(elements)
        atom = F.one_hot(
            (types - 1).clamp_min(0),
            num_classes=geometry_model.config.vocab_size,
        ).to(dtype=next(geometry_model.parameters()).dtype)
        atom = atom * 2.0 * mask.unsqueeze(-1)
        z0 = CrystalState(
            atom,
            torch.rand(
                1, mask.shape[1], 3, device=device, generator=generator
            ),
            torch.randn(1, 6, device=device, generator=generator),
        )
        prepared.append({
            "sample_id": row,
            "N": n_atoms,
            "elements": elements,
            "counts": _counts(elements),
            "mask": mask,
            "z0": z0,
        })
    root = Path(args.out_directory)
    root.mkdir(parents=True, exist_ok=True)
    calibrated_path = root / "calibrated_mp2020_hull_cache.json"
    calibrated_payload = build_calibrated_cache(
        args.phase_cache,
        [item["counts"] for item in prepared],
    )
    atomic_json(calibrated_path, calibrated_payload, sort_keys=True)
    calibrated = CalibratedMPHullCache(calibrated_path)
    novelty = NoveltyIndex.load(args.novelty_index, device)
    factory, uma_metadata = load_uma_calculator_vjp_factory(
        args.uma_checkpoint, device=args.device, task_name=args.task_name
    )
    differentiable_hull = DifferentiableMP2020Hull.from_phase_cache(
        args.phase_cache
    ).to(device)
    atomic_number_lookup = torch.tensor(
        [ATOMIC_NUMBERS[element] for element in ("Na", "Fe", "P", "O")],
        device=device,
    )
    config = JointDFlowConfig(
        steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        ode_steps=args.ode_steps,
        gradient_clip=args.gradient_clip,
        e_hull_scale_eV_atom=args.hull_scale,
        novelty_scale=args.novelty_scale,
        novelty_weight=args.novelty_weight,
        hull_threshold_eV_atom=args.hull_threshold,
        hull_threshold_weight=args.hull_threshold_weight,
        hull_constraint_tolerance_eV_atom=args.hull_constraint_tolerance,
        distance_weight=args.distance_weight,
        p_coordination_weight=args.p_coordination_weight,
        fe_coordination_weight=args.fe_coordination_weight,
        volume_weight=args.volume_weight,
        source_prior_weight=args.source_prior_weight,
        constraint_margin_A=args.distance_margin_A,
        ipopt_tolerance=args.ipopt_tolerance,
        ipopt_acceptable_tolerance=args.ipopt_acceptable_tolerance,
        ipopt_print_level=args.ipopt_print_level,
    )
    identity = {
        "version": "unconditional-joint-dflow-campaign-v4",
        "count": args.count,
        "seed": args.seed,
        "generation": {
            "distribution": "learned_unconditional_p(N,A)_then_joint_z_A_z_X_z_L_IPOPT",
            "composition_pool": None,
            "joint_checkpoint": str(Path(args.joint_checkpoint).resolve()),
            "joint_checkpoint_sha256": sha256_file(args.joint_checkpoint),
            "joint_model_sha256": model_sha256(joint_model),
            "geometry_checkpoint": str(Path(args.geometry_checkpoint).resolve()),
            "geometry_checkpoint_sha256": sha256_file(args.geometry_checkpoint),
            "geometry_model_sha256": model_sha256(geometry_model),
        },
        "optimization": {
            "solver": "CasADi/IPOPT",
            "derivatives": "exact_PyTorch_autograd_first_derivatives",
            "hessian": "IPOPT_limited_memory",
            "decision_variables": ["z_A", "z_X", "z_L"],
            "fixed_within_solve": ["N"],
            "single_joint_nonlinear_problem": True,
            "restoration_phase": False,
            "endpoint_post_optimization": False,
            "config": asdict(config),
        },
        "e_hull": {
            "definition": calibrated_payload["definition"],
            "ipopt_constraint": "raw_E_hull <= hull_threshold_eV_atom",
            "constraint_equivalence": (
                "raw_E_hull <= positive threshold is equivalent to "
                "max(0,raw_E_hull) <= threshold"
            ),
            "gradient_source": (
                "UMA force/stress VJP plus analytic active-facet MP2020 "
                "composition gradient through straight-through z_A; frozen Flow"
            ),
            "phase_cache": str(Path(args.phase_cache).resolve()),
            "phase_cache_sha256": sha256_file(args.phase_cache),
            "calibrated_cache_sha256": sha256_file(calibrated_path),
            "reference_implementation": (
                "/mnt/data2/LiuSW/shared_space/NaCathode/CodingSpace/"
                "relaxation_n_ehull.py"
            ),
        },
        "novelty_index": str(Path(args.novelty_index).resolve()),
        "novelty_index_sha256": sha256_file(args.novelty_index),
        "uma": uma_metadata,
    }
    identity_hash = sha256_json(identity)
    manifest_path = root / "manifest.json"
    manifest = identity | {
        "identity_sha256": identity_hash,
        "status": "running",
        "completed_records": 0,
    }
    atomic_json(manifest_path, manifest)
    records = root / "records"
    records.mkdir(exist_ok=True)
    for item in prepared:
        target = records / f"sample_{item['sample_id']:06d}.json"
        if target.exists() and args.resume:
            previous = json.loads(target.read_text())
            if previous.get("identity_sha256") != identity_hash:
                raise RuntimeError("resume identity mismatch")
            if previous.get("status") == "complete":
                continue
        elif target.exists():
            raise FileExistsError(target)
        record: dict[str, Any] = {
            "version": "unconditional-joint-dflow-sample-v4",
            "identity_sha256": identity_hash,
            "sample_id": item["sample_id"],
            "N": item["N"],
            "elements": item["elements"],
            "counts": item["counts"],
            "unconditional_z0": _state(item["z0"]),
            "status": "running",
        }
        atomic_json(target, record)
        try:
            numbers = torch.tensor(
                [ATOMIC_NUMBERS[element] for element in item["elements"]],
                device=device,
            )
            mp_reference, correction = calibrated.constants(item["counts"])
            problem = JointCrystalProblem(
                geometry_model,
                item["mask"],
                factory(numbers),
                mp_reference,
                correction,
                novelty,
                fe_coordination_prior=tuple(args.fe_coordination_prior),
                energy_factory=factory,
                differentiable_hull=differentiable_hull,
                atomic_number_lookup=atomic_number_lookup,
            )
            with torch.no_grad():
                baseline_terminal = integrate_flow(
                    geometry_model,
                    item["z0"],
                    item["mask"],
                    steps=args.ode_steps,
                    method="midpoint",
                    freeze_atom=False,
                )
            baseline_values = {
                name: float(value)
                for name, value in problem.values(baseline_terminal).items()
            }
            record["unconditional"] = {
                "values": baseline_values,
                "terminal": _state(baseline_terminal),
                "exact": _exact(
                    geometry_model, geometry_checkpoint, baseline_terminal,
                    item["mask"], item["elements"], baseline_values["E_hull"],
                ),
            }
            result = JointDFlowSolver(geometry_model, problem).solve(
                item["z0"], config
            )
            optimized_values = {
                name: float(value)
                for name, value in problem.values(result.terminal).items()
            }
            optimized_exact = _exact(
                geometry_model, geometry_checkpoint, result.terminal,
                item["mask"], item["elements"], optimized_values["E_hull"],
            )
            optimized_elements = optimized_exact["structure"]["elements"]
            optimized_counts = _counts(optimized_elements)
            # ``_exact`` receives the old expected list only to audit whether
            # composition changed; the decoded structure is authoritative.
            record["elements"] = optimized_elements
            record["counts"] = optimized_counts
            record["optimized"] = {
                "solver_status": result.status,
                "solver_stats": result.solver_stats,
                "values": optimized_values,
                "z0_star": _state(result.z0_star),
                "terminal": _state(result.terminal),
                "history": result.history,
                "exact": optimized_exact,
            }
            record["status"] = result.status
            if result.status == "complete" and not optimized_exact[
                "feasibility"
            ]["feasible"]:
                record["status"] = "constraints_violated"
        except Exception as error:
            record.update({
                "status": "error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            })
        atomic_json(target, record)
        print(json.dumps({
            "sample_id": item["sample_id"],
            "status": record["status"],
            "E_hull": record.get("optimized", {}).get("values", {}).get("E_hull"),
            "feasible": record.get("optimized", {}).get("exact", {}).get(
                "feasibility", {}
            ).get("feasible", False),
        }), flush=True)
    complete = 0
    failures = 0
    for path in records.glob("sample_*.json"):
        status = json.loads(path.read_text()).get("status")
        complete += int(status == "complete")
        failures += int(status != "complete")
    manifest.update({
        "completed_records": complete,
        "failed_records": failures,
        "status": "complete" if complete == args.count else "incomplete",
    })
    atomic_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--phase-cache", required=True)
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--optimization-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--ipopt-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--ipopt-acceptable-tolerance", type=float, default=1.0e-4
    )
    parser.add_argument("--ipopt-print-level", type=int, default=0)
    parser.add_argument("--hull-scale", type=float, default=10.0)
    parser.add_argument("--novelty-scale", type=float, default=100.0)
    parser.add_argument("--novelty-weight", type=float, default=0.10)
    parser.add_argument("--hull-threshold", type=float, default=0.150)
    parser.add_argument("--hull-threshold-weight", type=float, default=1.0)
    parser.add_argument(
        "--hull-constraint-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument("--distance-margin-A", type=float, default=0.10)
    parser.add_argument("--distance-weight", type=float, default=40.0)
    parser.add_argument("--p-coordination-weight", type=float, default=12.0)
    parser.add_argument("--fe-coordination-weight", type=float, default=2.0)
    parser.add_argument("--volume-weight", type=float, default=2.0)
    parser.add_argument("--source-prior-weight", type=float, default=0.02)
    parser.add_argument(
        "--fe-coordination-prior", type=float, nargs=3,
        default=(0.084, 0.1025, 0.8135),
    )
    parser.add_argument("--composition-pool-factor", type=int, default=8)
    parser.add_argument("--composition-max-rounds", type=int, default=40)
    parser.add_argument("--max-atoms", type=int, default=120)
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

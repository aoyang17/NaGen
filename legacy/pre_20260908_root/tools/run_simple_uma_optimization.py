"""SimpleOptimization with UMA replacing the learned E_hull surrogate.

For a fixed composition this minimizes the UMA energy per atom through the
complete frozen Flow map.  UMA forces and stress are constrained at every
optimization step; exact PBC minimum distances are deliberately only a hard
acceptance filter and never contribute a gradient.

An absolute E_hull is not reported: it requires a competing-phase hull whose
entries were all evaluated with the same UMA checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from pymatgen.core import Lattice, Structure

from nagen.inverse.constraints import evaluate_feasibility
from nagen.inverse.generate import load_model, sample_feasible_compositions
from nagen.inverse.lattice import decode_lattice, unstandardize_lattice
from nagen.inverse.model import CrystalState
from nagen.inverse.novelty import crystal_descriptor
from nagen.inverse.sample import integrate_flow
from nagen.inverse.uma_guidance import load_uma_native_second_order_evaluator
from run_surrogate_shootingflow import reference_descriptors


ATOMIC_NUMBER = {"Na": 11, "Fe": 26, "P": 15, "O": 8}
NOVELTY_TYPE = {"Na": 1, "Fe": 2, "P": 3, "O": 4}


def endpoint(flow, atom, mask, z_frac, z_lattice, ode_steps):
    terminal = integrate_flow(
        flow, CrystalState(atom, z_frac, z_lattice), mask,
        steps=ode_steps, method="midpoint", freeze_atom=True,
    )
    lattice = decode_lattice(
        unstandardize_lattice(
            terminal.lattice, flow.lattice_mean, flow.lattice_std,
        ),
        mask.sum(dim=1),
    )
    return terminal, lattice


def distance_check(elements, frac, lattice):
    result = evaluate_feasibility(
        elements, frac.detach().cpu().numpy(), lattice.detach().cpu().numpy(), None,
    )
    return (
        bool(result["checks"]["minimum_pbc_distances"]),
        result["values"]["minimum_distance_A"],
        result["details"]["distance_violations"],
    )


def select_valid_source(flow, atom, mask, elements, generator, args):
    """Screen Flow sources using only the exact, non-differentiable distance gate."""
    n_atoms = mask.shape[1]
    best = None
    with torch.no_grad():
        for _ in range(args.source_screen_rounds):
            count = args.source_screen_batch
            z_frac = torch.rand(
                count, n_atoms, 3, device=mask.device, generator=generator,
            )
            z_lattice = torch.randn(
                count, 6, device=mask.device, generator=generator,
            )
            batch_atom = atom.expand(count, -1, -1)
            batch_mask = mask.expand(count, -1)
            terminal, lattice = endpoint(
                flow, batch_atom, batch_mask, z_frac, z_lattice, args.ode_steps,
            )
            for index in range(count):
                ok, minimum, _ = distance_check(
                    elements, terminal.frac[index], lattice[index],
                )
                if ok and (best is None or minimum > best[0]):
                    best = (
                        float(minimum), z_frac[index:index + 1].clone(),
                        z_lattice[index:index + 1].clone(),
                    )
            if best is not None:
                break
    if best is None:
        return None
    return best[1], best[2], best[0]


def evaluate(
    *, flow, uma, atomic_numbers, novelty_types, atom, mask,
    z_frac, z_lattice, z0_frac, z0_lattice,
    ref_desc, ref_mean, ref_std, args,
):
    terminal, lattice = endpoint(
        flow, atom, mask, z_frac, z_lattice, args.ode_steps,
    )
    energy, forces, stress = uma(atomic_numbers, terminal.frac, lattice)
    energy = energy.reshape(() )
    fmax = torch.linalg.vector_norm(forces, dim=-1).amax()
    stress_fro = torch.linalg.vector_norm(stress.reshape(-1), ord=2)
    descriptor = crystal_descriptor(
        novelty_types, terminal.frac, lattice, mask,
    )
    novelty = torch.cdist(
        (descriptor - ref_mean) / ref_std,
        (ref_desc - ref_mean) / ref_std,
    ).amin() / math.sqrt(descriptor.shape[-1])
    drift = (
        (z_frac - z0_frac).square().mean()
        + (z_lattice - z0_lattice).square().mean()
    )
    stress_tolerance = args.stress_tolerance_gpa / 160.21766208
    force_violation = torch.relu(fmax - args.force_tolerance) / args.force_scale
    stress_violation = torch.relu(stress_fro - stress_tolerance) / args.stress_scale
    loss = (
        energy
        - args.novelty_weight * novelty
        + args.source_prior_weight * drift
        + args.constraint_weight
        * (force_violation.square() + stress_violation.square())
    )
    metrics = {
        "E_UMA_eV_atom": energy,
        "Fmax_eV_A": fmax,
        "stress_fro_eV_A3": stress_fro,
        "stress_fro_GPa": stress_fro * 160.21766208,
        "novelty": novelty,
        "source_drift": drift,
        "force_violation": force_violation,
        "stress_violation": stress_violation,
        "loss": loss,
    }
    return loss, metrics, terminal, lattice


def detached(metrics):
    return {key: float(value.detach()) for key, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--packed-data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--force-tolerance", type=float, default=0.01)
    parser.add_argument("--stress-tolerance-gpa", type=float, default=0.10)
    parser.add_argument("--force-scale", type=float, default=0.20)
    parser.add_argument("--stress-scale", type=float, default=0.005)
    parser.add_argument("--constraint-weight", type=float, default=5.0)
    parser.add_argument("--novelty-weight", type=float, default=0.02)
    parser.add_argument("--source-prior-weight", type=float, default=0.01)
    parser.add_argument("--frac-trust-radius", type=float, default=0.15)
    parser.add_argument("--lattice-trust-radius", type=float, default=0.50)
    parser.add_argument("--source-screen-batch", type=int, default=16)
    parser.add_argument("--source-screen-rounds", type=int, default=4)
    args = parser.parse_args()

    started = time.monotonic()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    flow, flow_ckpt = load_model(args.flow, device, use_ema=True)
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    uma, uma_metadata = load_uma_native_second_order_evaluator(
        args.uma_checkpoint, device=str(device), task_name="omat",
    )
    # count=0 means every supported train/validation/test row is used.
    ref_desc, ref_mean, ref_std = reference_descriptors(
        args.packed_data, 0, device,
    )
    types, masks = sample_feasible_compositions(
        flow, flow_ckpt, args.count, generator, args.ode_steps, "midpoint",
        pool_factor=512, max_rounds=60, max_atoms=184,
        proposal_batch_limit=512,
    )
    types, masks = types.to(device), masks.to(device)
    inverse = {
        int(value): key for key, value in flow_ckpt["element_to_index"].items()
    }
    output_path = Path(args.out)
    structure_dir = output_path.parent / f"{output_path.stem}_structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    records = []
    rejected_source = 0

    for sample_id in range(args.count):
        row_types = types[sample_id, masks[sample_id]]
        elements = [inverse[int(value)] for value in row_types.tolist()]
        n_atoms = len(elements)
        mask = torch.ones(1, n_atoms, dtype=torch.bool, device=device)
        atom = torch.nn.functional.one_hot(
            row_types - 1, num_classes=flow.config.vocab_size,
        ).float().unsqueeze(0) * 2.0
        selected = select_valid_source(
            flow, atom, mask, elements, generator, args,
        )
        if selected is None:
            rejected_source += 1
            print(json.dumps({
                "sample_id": sample_id,
                "rejected": "no_exact_distance_valid_flow_source",
            }), flush=True)
            continue
        z0_frac, z0_lattice, initial_minimum_distance = selected
        z_frac = z0_frac.clone().requires_grad_(True)
        z_lattice = z0_lattice.clone().requires_grad_(True)
        atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[element] for element in elements],
            dtype=torch.long, device=device,
        )
        novelty_types = torch.tensor(
            [[NOVELTY_TYPE[element] for element in elements]],
            dtype=torch.long, device=device,
        )
        optimizer = torch.optim.Adam(
            [z_frac, z_lattice], lr=args.learning_rate,
        )
        best = None
        initial_metrics = None
        history = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, terminal, lattice = evaluate(
                flow=flow, uma=uma, atomic_numbers=atomic_numbers,
                novelty_types=novelty_types, atom=atom, mask=mask,
                z_frac=z_frac, z_lattice=z_lattice,
                z0_frac=z0_frac, z0_lattice=z0_lattice,
                ref_desc=ref_desc, ref_mean=ref_mean, ref_std=ref_std,
                args=args,
            )
            values = detached(metrics)
            if initial_metrics is None:
                initial_metrics = values
            exact_ok, minimum_distance, violations = distance_check(
                elements, terminal.frac[0], lattice[0],
            )
            stationary_violation = (
                max(0.0, values["Fmax_eV_A"] - args.force_tolerance)
                / args.force_tolerance
                + max(0.0, values["stress_fro_GPa"] - args.stress_tolerance_gpa)
                / args.stress_tolerance_gpa
            )
            key = (
                0 if exact_ok else 1,
                stationary_violation,
                values["E_UMA_eV_atom"] - args.novelty_weight * values["novelty"],
            )
            if exact_ok and (best is None or key < best[0]):
                best = (
                    key, z_frac.detach().clone(), z_lattice.detach().clone(),
                    float(minimum_distance), violations,
                )
            history.append({"step": step, "distance_ok": exact_ok, **values})
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z_frac, z_lattice], 10.0)
            optimizer.step()
            with torch.no_grad():
                delta_frac = (z_frac - z0_frac).clamp(
                    -args.frac_trust_radius, args.frac_trust_radius,
                )
                z_frac.copy_((z0_frac + delta_frac).remainder(1.0))
                delta_lattice = (z_lattice - z0_lattice).clamp(
                    -args.lattice_trust_radius, args.lattice_trust_radius,
                )
                z_lattice.copy_((z0_lattice + delta_lattice).clamp(-5.0, 5.0))
            print(json.dumps({
                "sample_id": sample_id, "step": step,
                "distance_ok": exact_ok,
                "E_UMA_eV_atom": values["E_UMA_eV_atom"],
                "Fmax_eV_A": values["Fmax_eV_A"],
                "stress_fro_GPa": values["stress_fro_GPa"],
            }), flush=True)

        if best is None:
            print(json.dumps({
                "sample_id": sample_id,
                "rejected": "all_optimization_iterates_failed_distance_filter",
            }), flush=True)
            continue
        best_frac = best[1].requires_grad_(True)
        best_lattice = best[2].requires_grad_(True)
        _, final_metrics, terminal, lattice = evaluate(
            flow=flow, uma=uma, atomic_numbers=atomic_numbers,
            novelty_types=novelty_types, atom=atom, mask=mask,
            z_frac=best_frac, z_lattice=best_lattice,
            z0_frac=z0_frac, z0_lattice=z0_lattice,
            ref_desc=ref_desc, ref_mean=ref_mean, ref_std=ref_std, args=args,
        )
        final_values = detached(final_metrics)
        stationary = bool(
            final_values["Fmax_eV_A"] <= args.force_tolerance
            and final_values["stress_fro_GPa"] <= args.stress_tolerance_gpa
        )
        structure = Structure(
            Lattice(lattice[0].detach().cpu().numpy()), elements,
            terminal.frac[0].detach().cpu().numpy(),
        )
        cif_path = structure_dir / f"sample_{sample_id:03d}.cif"
        structure.to(filename=str(cif_path))
        record = {
            "sample_id": sample_id, "seed": args.seed, "N": n_atoms,
            "formula": structure.formula,
            "initial": initial_metrics,
            "initial_minimum_distance_A": initial_minimum_distance,
            "final": final_values,
            "energy_change_eV_atom": (
                final_values["E_UMA_eV_atom"]
                - initial_metrics["E_UMA_eV_atom"]
            ),
            "stationary_constraints_feasible": stationary,
            "minimum_distance_filter_passed": True,
            "minimum_distance_A": best[3],
            "cif": str(cif_path.resolve()),
            "history": history,
        }
        records.append(record)
        print(json.dumps({
            "sample_id": sample_id, "formula": structure.formula,
            "stationary": stationary, "final": final_values,
        }), flush=True)

    records.sort(key=lambda row: (
        not row["stationary_constraints_feasible"],
        max(0.0, row["final"]["Fmax_eV_A"] - args.force_tolerance)
        / args.force_tolerance
        + max(0.0, row["final"]["stress_fro_GPa"] - args.stress_tolerance_gpa)
        / args.stress_tolerance_gpa,
        row["final"]["E_UMA_eV_atom"],
    ))
    summary = {
        "version": "nagen-simple-optimization-uma-v1",
        "interpretation": (
            "Fixed-composition UMA energy minimization; absolute E_hull omitted "
            "because no all-UMA competing-phase hull is available."
        ),
        "flow_checkpoint": str(Path(args.flow).resolve()),
        "flow_training_data": "all data represented by supplied full checkpoint",
        "uma": uma_metadata,
        "novelty_reference_count": int(ref_desc.shape[0]),
        "all_differentiable_constraints_active_every_step": True,
        "force_stress_use_second_order_derivatives": True,
        "minimum_distance_role": "exact non-gradient source/final filter",
        "attempted_count": args.count,
        "rejected_source": rejected_source,
        "count": len(records),
        "stationary_feasible": sum(
            row["stationary_constraints_feasible"] for row in records
        ),
        "wall_time_seconds": time.monotonic() - started,
        "best": records[0] if records else None,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "count": summary["count"],
        "stationary_feasible": summary["stationary_feasible"],
        "best_formula": summary["best"]["formula"] if summary["best"] else None,
        "wall_time_seconds": summary["wall_time_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()

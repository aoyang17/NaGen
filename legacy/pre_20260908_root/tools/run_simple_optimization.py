"""Single-surrogate DFlow source optimization from SimpleOptimization.md."""
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
from nagen.surrogate.ehull_mace import load_surrogate
from run_surrogate_shootingflow import build_surrogate_graph_batch, reference_descriptors


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


def metrics_and_loss(
    *, flow, surrogate, surrogate_ckpt, elements, novelty_types,
    atom, mask, z_frac, z_lattice, z0_frac, z0_lattice,
    ref_desc, ref_mean, ref_std, args,
):
    terminal, lattice = endpoint(
        flow, atom, mask, z_frac, z_lattice, args.ode_steps,
    )
    graph = build_surrogate_graph_batch(
        [elements], terminal.frac, lattice,
        surrogate_ckpt["element_to_index"], surrogate.config.cutoff_A,
        surrogate.config.max_neighbors,
    )
    graph = {key: value.to(z_frac.device) for key, value in graph.items()}
    graph["frac"] = terminal.frac[mask]
    graph["lattice"] = lattice
    prediction = surrogate(graph).reshape(())
    grad_x, grad_l = torch.autograd.grad(
        prediction, (terminal.frac, lattice), create_graph=True,
    )
    grad_x_norm = torch.linalg.vector_norm(grad_x[mask], ord=2)
    grad_l_norm = torch.linalg.vector_norm(grad_l.reshape(-1), ord=2)
    descriptor = crystal_descriptor(novelty_types, terminal.frac, lattice, mask)
    novelty = (
        torch.cdist(
            (descriptor - ref_mean) / ref_std,
            (ref_desc - ref_mean) / ref_std,
        ).amin() / math.sqrt(descriptor.shape[-1])
    )
    # Smooth probability of the physically valid stability interval.  Unlike
    # a one-sided sigmoid, this does not reward negative hull predictions.
    p_stab = (
        torch.sigmoid((prediction - args.min_e_hull) / args.tau_e)
        * torch.sigmoid((args.max_e_hull - prediction) / args.tau_e)
    )
    drift = (
        (z_frac - z0_frac).square().mean()
        + (z_lattice - z0_lattice).square().mean()
    )
    objective = (
        args.energy_weight * torch.log(p_stab.clamp_min(1.0e-8))
        + args.novelty_weight * novelty
        - args.source_prior_weight * drift
    )
    violations = torch.stack((
        torch.relu(args.min_e_hull - prediction) / args.e_hull_constraint_scale,
        torch.relu(prediction - args.max_e_hull) / args.e_hull_constraint_scale,
        torch.relu(grad_x_norm - args.epsilon_x) / args.epsilon_x,
        torch.relu(grad_l_norm - args.epsilon_l) / args.epsilon_l,
    ))
    loss = -objective + args.constraint_weight * violations.square().sum()
    metrics = {
        "surrogate_e_hull": prediction,
        "p_stab": p_stab,
        "novelty": novelty,
        "grad_x_norm": grad_x_norm,
        "grad_l_norm": grad_l_norm,
        "source_drift": drift,
        "objective_score": objective,
        "constraint_violation": violations.sum(),
    }
    return loss, metrics, terminal, lattice


def detached(metrics):
    return {key: float(value.detach()) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", required=True)
    parser.add_argument("--surrogate", required=True)
    parser.add_argument("--packed-data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--min-e-hull", type=float, default=0.0)
    parser.add_argument("--max-e-hull", type=float, default=0.001)
    parser.add_argument("--e-hull-constraint-scale", type=float, default=0.005)
    parser.add_argument("--epsilon-e", type=float, default=0.001)
    parser.add_argument("--tau-e", type=float, default=0.005)
    parser.add_argument("--p-min", type=float, default=0.50)
    parser.add_argument("--epsilon-x", type=float, default=0.10)
    parser.add_argument("--epsilon-l", type=float, default=0.05)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--novelty-weight", type=float, default=0.05)
    parser.add_argument("--source-prior-weight", type=float, default=0.01)
    parser.add_argument("--constraint-weight", type=float, default=100.0)
    args = parser.parse_args()

    started = time.monotonic()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    flow, flow_ckpt = load_model(args.flow, device, use_ema=True)
    surrogate, surrogate_ckpt = load_surrogate(args.surrogate, device)
    for model in (flow, surrogate):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    # count=0 deliberately loads every supported row across train/val/test as
    # the novelty reference set; it does not use their labels for optimization.
    ref_desc, ref_mean, ref_std = reference_descriptors(args.packed_data, 0, device)
    types, masks = sample_feasible_compositions(
        flow, flow_ckpt, args.count, generator, args.ode_steps, "midpoint",
        pool_factor=512, max_rounds=60, max_atoms=184,
        proposal_batch_limit=512,
    )
    types, masks = types.to(device), masks.to(device)
    inverse = {int(value): key for key, value in flow_ckpt["element_to_index"].items()}
    novelty_map = {"Na": 1, "Fe": 2, "P": 3, "O": 4}
    records = []
    rejected_negative = 0
    rejected_minimum_distance = 0
    output_path = Path(args.out)
    structure_dir = output_path.parent / f"{output_path.stem}_structures"
    structure_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in range(args.count):
        row_types = types[sample_id, masks[sample_id]]
        elements = [inverse[int(value)] for value in row_types.tolist()]
        n_atoms = len(elements)
        mask = torch.ones(1, n_atoms, dtype=torch.bool, device=device)
        atom = torch.nn.functional.one_hot(
            row_types - 1, num_classes=flow.config.vocab_size,
        ).float().unsqueeze(0) * 2.0
        novelty_types = torch.tensor(
            [[novelty_map[element] for element in elements]],
            dtype=torch.long, device=device,
        )
        z0_frac = torch.rand(
            1, n_atoms, 3, device=device, generator=generator,
        )
        z0_lattice = torch.randn(1, 6, device=device, generator=generator)
        z_frac = z0_frac.clone().requires_grad_(True)
        z_lattice = z0_lattice.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z_frac, z_lattice], lr=args.learning_rate)
        best_key = float("inf")
        best_frac, best_lattice = z0_frac.clone(), z0_lattice.clone()
        initial_metrics = None
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, _, _ = metrics_and_loss(
                flow=flow, surrogate=surrogate, surrogate_ckpt=surrogate_ckpt,
                elements=elements, novelty_types=novelty_types, atom=atom,
                mask=mask, z_frac=z_frac, z_lattice=z_lattice,
                z0_frac=z0_frac, z0_lattice=z0_lattice,
                ref_desc=ref_desc, ref_mean=ref_mean, ref_std=ref_std, args=args,
            )
            if initial_metrics is None:
                initial_metrics = detached(metrics)
            key = float((metrics["constraint_violation"] - 0.01 * metrics["objective_score"]).detach())
            if key < best_key:
                best_key = key
                best_frac.copy_(z_frac.detach())
                best_lattice.copy_(z_lattice.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z_frac, z_lattice], 10.0)
            optimizer.step()
            with torch.no_grad():
                z_frac.remainder_(1.0)
                z_lattice.clamp_(-5.0, 5.0)

        best_frac.requires_grad_(True)
        best_lattice.requires_grad_(True)
        _, final_metrics, terminal, lattice = metrics_and_loss(
            flow=flow, surrogate=surrogate, surrogate_ckpt=surrogate_ckpt,
            elements=elements, novelty_types=novelty_types, atom=atom,
            mask=mask, z_frac=best_frac, z_lattice=best_lattice,
            z0_frac=z0_frac, z0_lattice=z0_lattice,
            ref_desc=ref_desc, ref_mean=ref_mean, ref_std=ref_std, args=args,
        )
        final_values = detached(final_metrics)
        if final_values["surrogate_e_hull"] < args.min_e_hull:
            rejected_negative += 1
            print(json.dumps({
                "sample_id": sample_id,
                "rejected": "negative_surrogate_e_hull",
            }), flush=True)
            continue
        exact_checks = evaluate_feasibility(
            elements,
            terminal.frac[0].detach().cpu().numpy(),
            lattice[0].detach().cpu().numpy(),
            final_values["surrogate_e_hull"],
        )
        if not exact_checks["checks"]["minimum_pbc_distances"]:
            rejected_minimum_distance += 1
            print(json.dumps({
                "sample_id": sample_id,
                "rejected": "minimum_pbc_distances",
                "minimum_distance_A": exact_checks["values"]["minimum_distance_A"],
            }), flush=True)
            continue
        valid_e_hull_interval = bool(
            final_values["surrogate_e_hull"] < args.max_e_hull
        )
        feasible = bool(
            valid_e_hull_interval
            and final_values["grad_x_norm"] <= args.epsilon_x
            and final_values["grad_l_norm"] <= args.epsilon_l
        )
        structure = Structure(
            Lattice(lattice[0].detach().cpu().numpy()), elements,
            terminal.frac[0].detach().cpu().numpy(),
        )
        cif_path = structure_dir / f"sample_{sample_id:03d}.cif"
        structure.to(filename=str(cif_path))
        record = {
            "sample_id": sample_id, "seed": args.seed, "N": n_atoms,
            "formula": structure.formula, "initial": initial_metrics,
            "e_hull_interval_eV_atom": [args.min_e_hull, args.max_e_hull],
            "final": final_values, "stationary_constraints_feasible": feasible,
            "minimum_distance_A": exact_checks["values"]["minimum_distance_A"],
            "ranking_score": final_values["objective_score"]
            - args.constraint_weight * final_values["constraint_violation"] ** 2,
            "cif": str(cif_path.resolve()), "elements": elements,
            "frac_coords": terminal.frac[0].detach().cpu().tolist(),
            "lattice": lattice[0].detach().cpu().tolist(),
        }
        records.append(record)
        print(json.dumps({
            key: record[key] for key in (
                "sample_id", "formula", "stationary_constraints_feasible", "ranking_score"
            )
        }), flush=True)

    records.sort(key=lambda row: row["ranking_score"], reverse=True)
    summary = {
        "version": "nagen-simple-optimization-single-surrogate-v1",
        "flow_checkpoint": str(Path(args.flow).resolve()),
        "surrogate_checkpoint": str(Path(args.surrogate).resolve()),
        "surrogate_dataset": surrogate_ckpt["dataset"],
        "surrogate_best_validation_MAE_eV_atom": surrogate_ckpt["best_validation_MAE_eV_atom"],
        "novelty_reference_count": int(ref_desc.shape[0]),
        "all_constraints_active_every_step": True,
        "second_order_surrogate_derivatives": True,
        "attempted_count": args.count,
        "rejected_negative_e_hull": rejected_negative,
        "rejected_minimum_distance": rejected_minimum_distance,
        "count": len(records), "feasible": sum(
            row["stationary_constraints_feasible"] for row in records
        ),
        "wall_time_seconds": time.monotonic() - started,
        "best": records[0] if records else None, "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "count": summary["count"], "feasible": summary["feasible"],
        "best_formula": summary["best"]["formula"] if summary["best"] else None,
        "best_score": summary["best"]["ranking_score"] if summary["best"] else None,
    }), flush=True)


if __name__ == "__main__":
    main()

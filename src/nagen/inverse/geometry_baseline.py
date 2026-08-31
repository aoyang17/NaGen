"""Paired fixed-composition geometry baseline for checkpoint selection."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ._campaign import fixed_composition_source as _source
from .constraints import evaluate_feasibility
from .generate import load_model
from .guidance import TerminalWeights, optimize_terminal
from .multiobjective_shooting import (
    CrystalDesignProblem, ShootingConfig, ShootingFlowRunner,
)
from .sample import decode_terminal, integrate_flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--composition-pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--restoration-steps", type=int, default=0)
    parser.add_argument("--guidance-steps", type=int, default=0)
    parser.add_argument("--uma-checkpoint")
    parser.add_argument("--endpoint-steps", type=int, default=0)
    parser.add_argument("--fixed-coordination-assignment", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--distance-weight", type=float, default=20.0)
    parser.add_argument(
        "--distance-margin-A", type=float, default=0.035,
        help="Safety margin added to exact pair minima during source restoration",
    )
    parser.add_argument("--p-coordination-weight", type=float, default=8.0)
    parser.add_argument("--fe-coordination-weight", type=float, default=5.0)
    parser.add_argument(
        "--fe-coordination-prior",
        help="Comma-separated prior probabilities for Fe CN=4,5,6",
    )
    args = parser.parse_args()
    fe_prior = None
    if args.fe_coordination_prior:
        fe_prior = tuple(
            float(value) for value in args.fe_coordination_prior.split(",")
        )
        if len(fe_prior) != 3:
            raise ValueError("--fe-coordination-prior requires three values")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device, use_ema=True)
    inverse = {
        int(index): element
        for element, index in checkpoint["element_to_index"].items()
    }
    energy_factory = None
    energy_metadata = None
    if args.guidance_steps > 0:
        if not args.uma_checkpoint:
            raise ValueError("--uma-checkpoint is required when --guidance-steps > 0")
        from .uma_guidance import load_uma_calculator_vjp_factory
        energy_factory, energy_metadata = load_uma_calculator_vjp_factory(
            args.uma_checkpoint, device=args.device, task_name="omat"
        )
    pool = json.loads(Path(args.composition_pool).read_text())
    stop = args.start_index + args.count
    selected = pool["unique_compositions"][args.start_index:stop]
    if len(selected) != args.count:
        raise ValueError(
            f"requested composition slice [{args.start_index}:{stop}] has "
            f"only {len(selected)} records"
        )
    records = []
    for local_id, composition in enumerate(selected):
        sample_id = args.start_index + local_id
        source, mask = _source(model, composition, args.seed + sample_id, device)
        history = []
        if args.restoration_steps > 0 or args.guidance_steps > 0:
            if energy_factory is None:
                energy_function = lambda frac, lattice: frac.sum((1, 2)) * 0.0
            else:
                from ase.data import atomic_numbers
                numbers = torch.tensor(
                    [atomic_numbers[inverse[int(index)]] for index in composition["type_indices"]],
                    dtype=torch.long,
                    device=device,
                )
                energy_function = energy_factory(numbers)
            problem = CrystalDesignProblem(
                model,
                mask,
                energy_function,
                0.0,
                fe_coordination_prior=fe_prior,
                constraint_margin_A=args.distance_margin_A,
            )
            result = ShootingFlowRunner(model, problem).run(
                int(mask.sum()),
                source.atom,
                source,
                ShootingConfig(
                    restoration_steps=args.restoration_steps,
                    guidance_steps=args.guidance_steps,
                    learning_rate=args.learning_rate,
                    ode_steps=args.ode_steps,
                    use_hull=args.guidance_steps > 0,
                    report_hull=False if args.guidance_steps > 0 else True,
                    use_novelty=False,
                    distance_weight=args.distance_weight,
                    p_coordination_weight=args.p_coordination_weight,
                    fe_coordination_weight=args.fe_coordination_weight,
                ),
            )
            terminal = result.terminal
            history = result.history
        else:
            with torch.no_grad():
                terminal = integrate_flow(
                    model, source, mask, steps=args.ode_steps,
                    method="midpoint", freeze_atom=True,
                )
        endpoint_history = []
        if args.endpoint_steps > 0:
            terminal, endpoint_history = optimize_terminal(
                model,
                terminal,
                mask,
                optimization_steps=args.endpoint_steps,
                learning_rate=args.learning_rate,
                weights=TerminalWeights(
                    distance=args.distance_weight,
                    p_coordination=args.p_coordination_weight,
                    fe_coordination=args.fe_coordination_weight,
                ),
                fixed_coordination_assignment=args.fixed_coordination_assignment,
            )
        with torch.no_grad():
            types, frac, lattice = decode_terminal(model, terminal, mask)
        elements = [inverse[int(value)] for value in types[0].cpu().tolist()]
        feasibility = evaluate_feasibility(
            elements,
            frac[0].cpu().numpy(),
            lattice[0].cpu().numpy(),
            None,
        )
        records.append({
            "sample_id": sample_id,
            "seed": args.seed + sample_id,
            "counts": composition["counts"],
            "structure": {
                "elements": elements,
                "frac_coords": frac[0].cpu().tolist(),
                "lattice": lattice[0].cpu().tolist(),
            },
            "feasibility": feasibility,
            "restoration_history": history,
            "endpoint_history": endpoint_history,
        })
    geometry_checks = (
        "positive_lattice_determinant", "volume_per_atom_range",
        "minimum_pbc_distances", "p_coordination_eq_4",
        "fe_coordination_in_4_5_6",
    )
    summary = {
        "n": len(records),
        "geometry_pass": sum(
            all(record["feasibility"]["checks"][key] for key in geometry_checks)
            for record in records
        ),
        "check_pass_counts": {
            key: sum(record["feasibility"]["checks"][key] for record in records)
            for key in geometry_checks
        },
        "median_minimum_distance_A": statistics.median(
            record["feasibility"]["values"]["minimum_distance_A"]
            for record in records
        ),
    }
    payload = {
        "version": "paired-geometry-baseline-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": args.seed,
        "ode_steps": args.ode_steps,
        "restoration": {
            "steps": args.restoration_steps,
            "learning_rate": args.learning_rate,
            "distance_weight": args.distance_weight,
            "distance_margin_A": args.distance_margin_A,
            "p_coordination_weight": args.p_coordination_weight,
            "fe_coordination_weight": args.fe_coordination_weight,
            "fe_coordination_prior": fe_prior,
        },
        "energy_guidance": {
            "steps": args.guidance_steps,
            "objective": "raw_UMA_energy_per_atom" if args.guidance_steps else None,
            "absolute_E_hull_gate_applied": False,
            "metadata": energy_metadata,
        },
        "endpoint_projection": {
            "steps": args.endpoint_steps,
            "fixed_coordination_assignment": args.fixed_coordination_assignment,
        },
        "summary": summary,
        "records": records,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

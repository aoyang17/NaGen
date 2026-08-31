"""Run the post-generation UMA relaxation gate on a small pilot set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ._uma import load_calculator as _load_calculator
from .constraints import evaluate_feasibility
from .relax_uma import RelaxationConfig, relax_structure


GEOMETRY_CHECKS = (
    "positive_lattice_determinant",
    "volume_per_atom_range",
    "minimum_pbc_distances",
    "p_coordination_eq_4",
    "fe_coordination_in_4_5_6",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-directory", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--start-selected", type=int, default=0)
    parser.add_argument("--include-infeasible", action="store_true")
    parser.add_argument(
        "--sample-ids",
        help="Comma-separated sample IDs; explicit selection overrides geometry prefilter",
    )
    parser.add_argument(
        "--required-checks",
        help="Comma-separated pre-relaxation checks; defaults to all geometry checks",
    )
    parser.add_argument("--fire-fmax", type=float, default=0.05)
    parser.add_argument("--fire-steps", type=int, default=1500)
    parser.add_argument("--bfgs-fmax", type=float, default=0.01)
    parser.add_argument("--bfgs-steps", type=int, default=1500)
    args = parser.parse_args()

    from ase import Atoms

    payload = json.loads(Path(args.input).read_text())
    requested_ids = (
        {int(value) for value in args.sample_ids.split(",")}
        if args.sample_ids else None
    )
    required_checks = (
        tuple(value for value in args.required_checks.split(",") if value)
        if args.required_checks else GEOMETRY_CHECKS
    )
    eligible = []
    for record in payload["records"]:
        checks = record["feasibility"]["checks"]
        if requested_ids is not None:
            if record["sample_id"] in requested_ids:
                eligible.append(record)
        elif args.include_infeasible or all(checks[key] for key in required_checks):
            eligible.append(record)
    if requested_ids is not None and {r["sample_id"] for r in eligible} != requested_ids:
        raise RuntimeError("one or more requested sample IDs are absent from the input")
    selected = eligible[
        args.start_selected:args.start_selected + args.limit
    ]
    if len(selected) != args.limit:
        raise RuntimeError(
            f"requested {args.limit} pilot structures from selected offset "
            f"{args.start_selected}, but only {len(selected)} are available"
        )

    calculator = _load_calculator(args.checkpoint, args.device, args.task_name)
    config = RelaxationConfig(
        fire_fmax_eV_A=args.fire_fmax,
        fire_steps=args.fire_steps,
        bfgs_fmax_eV_A=args.bfgs_fmax,
        bfgs_steps=args.bfgs_steps,
    )
    work = Path(args.work_directory)
    records = []
    for record in selected:
        structure = record["structure"]
        atoms = Atoms(
            symbols=structure["elements"],
            scaled_positions=np.asarray(structure["frac_coords"], dtype=float),
            cell=np.asarray(structure["lattice"], dtype=float),
            pbc=True,
        )
        result = relax_structure(
            atoms,
            calculator,
            str(work / f"sample_{record['sample_id']:04d}"),
            config,
        )
        relaxed_frac = result.atoms.get_scaled_positions(wrap=True)
        relaxed_lattice = np.asarray(result.atoms.cell.array, dtype=float)
        final_energy = float(result.atoms.get_potential_energy()) / len(result.atoms)
        final_stress = np.asarray(
            result.atoms.get_stress(voigt=False), dtype=float
        )
        feasibility = evaluate_feasibility(
            structure["elements"], relaxed_frac, relaxed_lattice, None
        )
        records.append({
            "sample_id": record["sample_id"],
            "converged": result.converged,
            "status": result.status,
            "final_fmax_eV_A": result.final_fmax_eV_A,
            "final_energy_eV_atom": final_energy,
            "final_stress_eV_A3": final_stress.tolist(),
            "final_max_abs_stress_eV_A3": float(np.abs(final_stress).max()),
            "fire_converged": result.fire_converged,
            "bfgs_converged": result.bfgs_converged,
            "geometry_pass": all(
                feasibility["checks"][key] for key in GEOMETRY_CHECKS
            ),
            "feasibility": feasibility,
            "structure": {
                "elements": structure["elements"],
                "frac_coords": relaxed_frac.tolist(),
                "lattice": relaxed_lattice.tolist(),
            },
        })
    summary = {
        "selected": len(selected),
        "relaxation_converged": sum(record["converged"] for record in records),
        "post_relaxation_geometry_pass": sum(
            record["geometry_pass"] for record in records
        ),
    }
    output = {
        "version": "uma-relaxation-pilot-v1",
        "source": str(Path(args.input).resolve()),
        "uma_checkpoint": str(Path(args.checkpoint).resolve()),
        "device": args.device,
        "task_name": args.task_name,
        "config": config.__dict__,
        "summary": summary,
        "records": records,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

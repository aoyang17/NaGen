"""Merge and independently re-audit energy-guided candidate shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from pymatgen.core import Structure

from .campaign import analytical_feasible
from .constraints import evaluate_feasibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected", type=int, default=200)
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob("constrained_shard_*.jsonl"))
    records = []
    for path in paths:
        with path.open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(
        key=lambda record: record.get("energy_guidance_provenance", {}).get(
            "source_index", 10**9
        )
    )
    ids = [record["candidate_id"] for record in records]
    duplicate_ids = len(ids) - len(set(ids))
    reaudit_failures = 0
    errors = 0
    for record in records:
        if "energy_guidance_error" in record:
            errors += 1
            continue
        structure = Structure.from_dict(record["structure"])
        feasibility = evaluate_feasibility(
            [str(specie) for specie in structure.species],
            structure.frac_coords,
            structure.lattice.matrix,
            None,
        )
        record["feasibility_after_energy_guidance"] = feasibility
        record["analytical_feasible_after_energy_guidance"] = analytical_feasible(
            feasibility
        )
        reaudit_failures += int(not record["analytical_feasible_after_energy_guidance"])

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = output_dir / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "energy_guided.jsonl"
    with output_path.open("w") as handle:
        for index, record in enumerate(records):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if "energy_guidance_error" not in record:
                Structure.from_dict(record["structure"]).to(
                    filename=str(cif_dir / f"candidate_{index + 1:04d}.cif")
                )

    valid = [
        record
        for record in records
        if "thermodynamic_stability_score" in record
    ]
    converged = sum(
        record["thermodynamic_stability_score"][
            "constrained_optimizer_converged"
        ]
        for record in valid
    )
    analytic = sum(
        record.get("analytical_feasible_after_energy_guidance", False)
        for record in valid
    )
    energies = [
        record["thermodynamic_stability_score"]["final_physical_energy_eV_atom"]
        for record in valid
    ]
    changes = [
        record["thermodynamic_stability_score"]["physical_energy_change_eV_atom"]
        for record in valid
    ]
    biased_forces = [
        record["thermodynamic_stability_score"]["final_biased_max_force_eV_A"]
        for record in valid
    ]
    summary = {
        "input_shards": [str(path) for path in paths],
        "records": len(records),
        "expected": args.expected,
        "expected_count_met": len(records) == args.expected,
        "duplicate_candidate_ids": duplicate_ids,
        "errors": errors,
        "reaudit_failures": reaudit_failures,
        "analytically_feasible": analytic,
        "constrained_optimizer_converged": converged,
        "both_analytic_and_converged": sum(
            record.get("analytical_feasible_after_energy_guidance", False)
            and record["thermodynamic_stability_score"][
                "constrained_optimizer_converged"
            ]
            for record in valid
        ),
        "final_physical_energy_eV_atom": {
            "minimum": min(energies) if energies else None,
            "median": median(energies) if energies else None,
            "mean": mean(energies) if energies else None,
            "maximum": max(energies) if energies else None,
        },
        "physical_energy_change_eV_atom": {
            "minimum": min(changes) if changes else None,
            "median": median(changes) if changes else None,
            "mean": mean(changes) if changes else None,
            "maximum": max(changes) if changes else None,
        },
        "final_biased_max_force_eV_A": {
            "minimum": min(biased_forces) if biased_forces else None,
            "median": median(biased_forces) if biased_forces else None,
            "mean": mean(biased_forces) if biased_forces else None,
            "maximum": max(biased_forces) if biased_forces else None,
        },
        "delta_e_hull_evaluated": False,
        "output": str(output_path),
        "cif_directory": str(cif_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if len(records) != args.expected or duplicate_ids or errors or reaudit_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

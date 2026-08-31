"""Merge independent MLFF relaxations and retain audited unique survivors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .campaign import analytical_feasible
from .constraints import evaluate_feasibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob("relaxed_shard_*.jsonl"))
    records: list[dict] = []
    for path in paths:
        with path.open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(
        key=lambda record: record.get("relaxation_provenance", {}).get(
            "source_index", 10**9
        )
    )

    errors = 0
    converged_count = 0
    analytic_count = 0
    both_count = 0
    for record in records:
        if "relaxation_error" in record:
            errors += 1
            continue
        structure = Structure.from_dict(record["structure"])
        feasibility = evaluate_feasibility(
            [str(specie) for specie in structure.species],
            structure.frac_coords,
            structure.lattice.matrix,
            None,
        )
        record["feasibility_after_relaxation"] = feasibility
        analytic = analytical_feasible(feasibility)
        converged = bool(
            record.get("thermodynamic_stability_score", {}).get(
                "converged_at_fmax", False
            )
        )
        record["analytical_feasible_after_relaxation"] = analytic
        converged_count += int(converged)
        analytic_count += int(analytic)
        both_count += int(converged and analytic)

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    accepted_by_formula: dict[str, list[Structure]] = defaultdict(list)
    survivors: list[dict] = []
    duplicates = 0
    for record in sorted(
        records,
        key=lambda item: item.get("thermodynamic_stability_score", {}).get(
            "relaxed_energy_eV_atom", float("inf")
        ),
    ):
        score = record.get("thermodynamic_stability_score", {})
        if not (
            score.get("converged_at_fmax", False)
            and record.get("analytical_feasible_after_relaxation", False)
        ):
            continue
        structure = Structure.from_dict(record["structure"])
        formula = structure.composition.reduced_formula
        duplicate = any(
            matcher.fit(structure, previous)
            for previous in accepted_by_formula[formula]
        )
        if duplicate:
            duplicates += 1
            continue
        accepted_by_formula[formula].append(structure)
        survivors.append(record)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = output_dir / "cif_survivors"
    cif_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "relaxed_all.jsonl"
    survivor_path = output_dir / "relaxed_converged_analytic_unique.jsonl"
    with all_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with survivor_path.open("w") as handle:
        for index, record in enumerate(survivors, start=1):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            Structure.from_dict(record["structure"]).to(
                filename=str(cif_dir / f"candidate_{index:04d}.cif")
            )

    energies = [
        float(record["thermodynamic_stability_score"]["relaxed_energy_eV_atom"])
        for record in survivors
    ]
    summary = {
        "input_shards": [str(path) for path in paths],
        "records": len(records),
        "expected": args.expected,
        "expected_count_met": len(records) == args.expected,
        "duplicate_candidate_ids": len(records)
        - len({record.get("candidate_id") for record in records}),
        "errors": errors,
        "force_converged": converged_count,
        "analytically_feasible": analytic_count,
        "both_converged_and_analytic_before_dedup": both_count,
        "post_relaxation_duplicates_removed": duplicates,
        "unique_survivors": len(survivors),
        "survivor_energy_eV_atom": {
            "minimum": min(energies) if energies else None,
            "median": median(energies) if energies else None,
            "mean": mean(energies) if energies else None,
            "maximum": max(energies) if energies else None,
        },
        "delta_e_hull_evaluated": False,
        "all_output": str(all_path),
        "survivor_output": str(survivor_path),
        "cif_directory": str(cif_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if len(records) != args.expected or errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

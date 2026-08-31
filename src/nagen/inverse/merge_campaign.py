"""Global deduplication and exact re-audit of constrained campaign shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .campaign import analytical_feasible
from .constraints import evaluate_feasibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    input_paths = sorted(Path(args.input_dir).glob("accepted_shard_*.jsonl"))
    if not input_paths:
        raise FileNotFoundError("no accepted_shard_*.jsonl inputs found")
    records: list[dict[str, Any]] = []
    for path in input_paths:
        with path.open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    # Preserve the multi-objective signal that is currently available.  Hull
    # scoring is intentionally not fabricated here and will be joined later.
    records.sort(key=lambda record: record["novelty_score"], reverse=True)

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    seen_by_formula: dict[str, list[Structure]] = defaultdict(list)
    selected: list[dict[str, Any]] = []
    global_duplicates = 0
    reaudit_failures = 0
    for record in records:
        structure = Structure.from_dict(record["structure"])
        feasibility = evaluate_feasibility(
            [str(specie) for specie in structure.species],
            structure.frac_coords,
            structure.lattice.matrix,
            None,
        )
        if not analytical_feasible(feasibility):
            reaudit_failures += 1
            continue
        formula = structure.composition.reduced_formula
        if any(
            matcher.fit(structure, previous)
            for previous in seen_by_formula[formula]
        ):
            global_duplicates += 1
            continue
        seen_by_formula[formula].append(structure)
        record["feasibility"] = feasibility
        record["analytical_feasible"] = True
        selected.append(record)
        if len(selected) == args.limit:
            break

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = output_dir / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "analytically_feasible.jsonl"
    with jsonl_path.open("w") as handle:
        for index, record in enumerate(selected):
            record["pool_rank_by_novelty"] = index + 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            Structure.from_dict(record["structure"]).to(
                filename=str(cif_dir / f"candidate_{index + 1:04d}.cif")
            )

    novelties = [record["novelty_score"] for record in selected]
    atom_counts = [record["N"] for record in selected]
    formulas = [record["reduced_formula"] for record in selected]
    summary = {
        "input_shards": [str(path) for path in input_paths],
        "input_accepted_records": len(records),
        "selected_unique_analytically_feasible": len(selected),
        "requested_limit": args.limit,
        "limit_reached": len(selected) == args.limit,
        "global_duplicates_rejected": global_duplicates,
        "reaudit_failures": reaudit_failures,
        "all_analytic_constraints_pass": all(
            record["analytical_feasible"] for record in selected
        ),
        "thermodynamic_hull_evaluated": False,
        "novelty": {
            "minimum": min(novelties) if novelties else None,
            "median": median(novelties) if novelties else None,
            "mean": mean(novelties) if novelties else None,
            "maximum": max(novelties) if novelties else None,
        },
        "N": {
            "minimum": min(atom_counts) if atom_counts else None,
            "median": median(atom_counts) if atom_counts else None,
            "maximum": max(atom_counts) if atom_counts else None,
        },
        "unique_reduced_formulas": len(set(formulas)),
        "most_common_formulas": Counter(formulas).most_common(20),
        "jsonl": str(jsonl_path),
        "cif_directory": str(cif_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if len(selected) < args.limit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

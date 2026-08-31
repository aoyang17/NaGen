"""Assemble globally unique final survivors with explicit hull scope."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .constraints import evaluate_feasibility


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    records: list[dict] = []
    for source in args.inputs:
        with open(source) as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(
        key=lambda record: (
            record.get("local_hull", {}).get(
                "delta_e_hull_eV_atom", float("inf")
            ),
            -record.get("novelty_score", 0.0),
        )
    )

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    seen: dict[str, list[Structure]] = defaultdict(list)
    unique: list[dict] = []
    duplicates = 0
    for record in records:
        structure = Structure.from_dict(record["structure"])
        formula = structure.composition.reduced_formula
        if any(matcher.fit(structure, previous) for previous in seen[formula]):
            duplicates += 1
            continue
        seen[formula].append(structure)
        hull = record.get("local_hull", {})
        delta = hull.get("delta_e_hull_eV_atom")
        provisional = evaluate_feasibility(
            [str(specie) for specie in structure.species],
            structure.frac_coords,
            structure.lattice.matrix,
            delta,
        )
        record["provisional_data_relative_feasibility"] = provisional
        record["provisional_all_constraints_pass"] = bool(
            hull.get("status") == "evaluated"
            and hull.get("provisional_pass", False)
            and provisional["feasible"]
        )
        unique.append(record)

    passed = [record for record in unique if record["provisional_all_constraints_pass"]]
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "geometry_force_survivors.jsonl"
    pass_path = output_dir / "provisional_hull_pass.jsonl"
    cif_dir = output_dir / "cif_provisional_hull_pass"
    cif_dir.mkdir(parents=True, exist_ok=True)
    with all_path.open("w") as handle:
        for record in unique:
            handle.write(
                json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
            )
    with pass_path.open("w") as handle:
        for index, record in enumerate(passed, start=1):
            handle.write(
                json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
            )
            Structure.from_dict(record["structure"]).to(
                filename=str(cif_dir / f"candidate_{index:04d}.cif")
            )

    summary = {
        "input_records": len(records),
        "global_duplicates_removed": duplicates,
        "unique_geometry_force_survivors": len(unique),
        "provisional_data_relative_hull_pass": len(passed),
        "strict_complete_dft_hull_pass": 0,
        "strict_complete_dft_hull_status": "not_evaluated_missing_consistent_candidate_and_complete_competing_phase_DFT_energies",
        "hull_scope": (
            "CHGNet-0.3.0 candidate relaxed energies against single-point energies "
            "of the 17,774 supplied reference geometries only"
        ),
        "geometry_survivors": str(all_path),
        "provisional_hull_pass_output": str(pass_path),
        "cif_directory": str(cif_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

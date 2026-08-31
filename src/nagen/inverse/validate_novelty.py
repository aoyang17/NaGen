"""Exact composition/StructureMatcher novelty audit against the raw corpus."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    with open(args.candidates) as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]
    candidate_formulas = {record["reduced_formula"] for record in candidates}
    references: dict[str, list[tuple[str, Structure]]] = defaultdict(list)
    reference_rows = 0
    with open(args.reference) as handle:
        for line in handle:
            if not line.strip():
                continue
            reference_rows += 1
            record = json.loads(line)
            formula = Composition(record["formula"]).reduced_formula
            if formula in candidate_formulas:
                references[formula].append(
                    (record["material_id"], Structure.from_dict(record["structure"]))
                )

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    composition_novel = 0
    structure_novel = 0
    matched = 0
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in candidates:
            formula = record["reduced_formula"]
            same_composition = references.get(formula, [])
            composition_in_reference = bool(same_composition)
            if not composition_in_reference:
                composition_novel += 1
            candidate = Structure.from_dict(record["structure"])
            matched_id = None
            for material_id, reference in same_composition:
                if matcher.fit(candidate, reference):
                    matched_id = material_id
                    break
            is_structure_novel = matched_id is None
            if is_structure_novel:
                structure_novel += 1
            else:
                matched += 1
            record["novelty_validation"] = {
                "composition_in_reference": composition_in_reference,
                "same_composition_reference_count": len(same_composition),
                "structure_match_in_reference": matched_id is not None,
                "matched_reference_id": matched_id,
                "structure_novel": is_structure_novel,
                "matcher": {
                    "implementation": "pymatgen.StructureMatcher",
                    "ltol": 0.2,
                    "stol": 0.3,
                    "angle_tol": 5.0,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "reference_rows_scanned": reference_rows,
        "candidate_count": len(candidates),
        "candidate_unique_reduced_formulas": len(candidate_formulas),
        "composition_novel_count": composition_novel,
        "composition_seen_but_structure_novel_count": (
            structure_novel - composition_novel
        ),
        "structure_novel_count": structure_novel,
        "reference_structure_match_count": matched,
        "all_candidates_structure_novel": structure_novel == len(candidates),
        "output": str(output),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

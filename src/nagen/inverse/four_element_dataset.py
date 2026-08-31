"""Build the auditable four-element training and novelty datasets.

The source is opened read-only.  Every Na--Fe--P--O source line receives one
audit record, while all four-element structures (including upstream failures)
remain available to the novelty reference payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch

from .constraints import evaluate_feasibility
from .dataset import _canonical_order, _first_mib_sha256, _parse_structure
from .spec import DEFAULT_SPEC


GEOMETRY_CHECKS = (
    "atom_count_range", "finite_structure", "fractional_coordinate_domain",
    "positive_lattice_determinant", "volume_per_atom_range",
    "minimum_pbc_distances", "p_coordination_eq_4",
    "fe_coordination_in_4_5_6",
)


def reduced_count_key(elements: list[str]) -> str:
    counts = Counter(elements)
    divisor = math.gcd(*(counts[element] for element in DEFAULT_SPEC.elements))
    return "".join(
        f"{element}{counts[element] // divisor}"
        for element in DEFAULT_SPEC.elements
    )


def formula_group_split(
    formulas: list[str], seed: int = 20260826
) -> tuple[list[int], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, formula in enumerate(formulas):
        groups[formula].append(index)
    rng = random.Random(seed)
    tie_break = {formula: rng.random() for formula in groups}
    ordered = sorted(groups, key=lambda f: (-len(groups[f]), tie_break[f], f))
    targets = {"train": 0.8 * len(formulas), "val": 0.1 * len(formulas), "test": 0.1 * len(formulas)}
    assigned: dict[str, str] = {}
    counts = Counter()
    for formula in ordered:
        split = min(
            targets,
            key=lambda name: (
                (counts[name] + len(groups[formula])) / max(targets[name], 1.0),
                name,
            ),
        )
        assigned[formula] = split
        counts[split] += len(groups[formula])
    code = {"train": 0, "val": 1, "test": 2}
    split_codes = [code[assigned[formula]] for formula in formulas]
    formula_sets = {
        name: {formula for formula, selected in assigned.items() if selected == name}
        for name in code
    }
    leakage = sorted(
        (formula_sets["train"] & formula_sets["val"])
        | (formula_sets["train"] & formula_sets["test"])
        | (formula_sets["val"] & formula_sets["test"])
    )
    return split_codes, {
        "seed": seed, "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "counts": dict(counts), "formula_counts": {
            name: len(values) for name, values in formula_sets.items()
        }, "formula_leakage": leakage,
    }


def _packed_payload(records: list[dict[str, Any]], source: str, version: str) -> dict[str, Any]:
    split_codes, split_manifest = formula_group_split(
        [record["reduced_formula"] for record in records]
    )
    offsets = [0]
    for record in records:
        offsets.append(offsets[-1] + len(record["types"]))
    return {
        "version": version,
        "source": str(Path(source).resolve()),
        "source_first_1MiB_sha256": _first_mib_sha256(source),
        "element_to_index": DEFAULT_SPEC.element_to_index,
        "types": torch.from_numpy(np.concatenate([r["types"] for r in records])),
        "frac_coords": torch.from_numpy(np.concatenate([r["frac"] for r in records])),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "lattices": torch.from_numpy(np.stack([r["lattice"] for r in records])),
        "optimized_energy_per_atom": torch.tensor([r["energy"] for r in records]),
        "split": torch.tensor(split_codes, dtype=torch.uint8),
        "ids": [r["id"] for r in records],
        "formulas": [r["reduced_formula"] for r in records],
        "source_indices": torch.tensor([r["source_index"] for r in records]),
        "source_record_sha256": [r["source_hash"] for r in records],
        "split_manifest": split_manifest,
    }


def build_four_element_datasets(
    raw_path: str, quality_output: str, reference_output: str,
    audit_output: str, manifest_output: str,
) -> dict[str, Any]:
    quality: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    total_source = 0
    with open(raw_path, "rb") as handle:
        for source_index, line in enumerate(handle):
            total_source += 1
            record = orjson.loads(line)
            elements, frac, lattice = _parse_structure(record["structure"])
            if set(elements) != set(DEFAULT_SPEC.elements):
                continue
            source_hash = hashlib.sha256(line).hexdigest()
            reduced_formula = reduced_count_key(elements)
            types = np.asarray(
                [DEFAULT_SPEC.element_to_index[element] for element in elements],
                dtype=np.int16,
            )
            order = _canonical_order(types, frac)
            packed = {
                "types": types[order], "frac": frac[order], "lattice": lattice,
                "energy": float(record.get("optimized_energy_per_atom", float("nan"))),
                "id": str(record.get("material_id", source_index)),
                "reduced_formula": reduced_formula, "source_index": source_index,
                "source_hash": source_hash,
            }
            reference.append(packed)
            failures: list[str] = []
            if "error_msg" in record:
                failures.append("upstream_error")
            try:
                result = evaluate_feasibility(elements, frac, lattice)
                failures.extend(name for name in GEOMETRY_CHECKS if not result["checks"][name])
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                result = None
                failures.append(f"invalid_structure:{type(error).__name__}")
            failures = sorted(set(failures))
            if not failures:
                quality.append(packed)
            for reason in failures:
                rejection_counts[reason] += 1
            audit_rows.append({
                "source_index": source_index, "material_id": packed["id"],
                "reduced_formula": reduced_formula, "source_record_sha256": source_hash,
                "accepted_quality": not failures, "rejection_reasons": failures,
                "upstream_error": "error_msg" in record,
            })
    if not quality or not reference:
        raise RuntimeError("four-element extraction produced an empty dataset")
    quality_payload = _packed_payload(quality, raw_path, "naxl-na-fe-p-o-quality-v1")
    reference_payload = _packed_payload(reference, raw_path, "naxl-na-fe-p-o-reference-v1")
    for path, payload in ((quality_output, quality_payload), (reference_output, reference_payload)):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
    audit_target = Path(audit_output)
    audit_target.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_target, "wb") as handle:
        for row in audit_rows:
            handle.write(orjson.dumps(row) + b"\n")
    manifest = {
        "version": "na-fe-p-o-dataset-manifest-v1",
        "source": str(Path(raw_path).resolve()),
        "source_first_1MiB_sha256": _first_mib_sha256(raw_path),
        "n_source_records": total_source, "n_four_element": len(reference),
        "n_quality": len(quality), "quality_minimum_required": 1000,
        "quality_gate_passed": len(quality) >= 1000,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "quality_split": quality_payload["split_manifest"],
        "reference_includes_upstream_errors": True,
        "outputs": {
            "quality": str(Path(quality_output).resolve()),
            "reference": str(Path(reference_output).resolve()),
            "audit": str(audit_target.resolve()),
        },
    }
    manifest_target = Path(manifest_output)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--quality-out", required=True)
    parser.add_argument("--reference-out", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    args = parser.parse_args()
    manifest = build_four_element_datasets(
        args.raw, args.quality_out, args.reference_out,
        args.audit_out, args.manifest_out,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

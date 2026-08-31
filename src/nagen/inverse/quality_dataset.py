"""Build a geometry-quality subset for unconditional N,A,X,L flow training.

The source JSONL is read-only.  Records with a reported relaxation error are
excluded, then the current executable P/TM coordination and pair-distance
definitions are applied exactly.  No family, prototype, parent, or topology
label is used as a conditioning variable.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch

from .constraints import evaluate_feasibility
from .dataset import _canonical_order, _first_mib_sha256, _parse_structure
from .spec import DEFAULT_SPEC


GEOMETRY_CHECKS = (
    "element_domain",
    "positive_atom_count",
    "fractional_coordinate_domain",
    "positive_lattice_determinant",
    "minimum_pbc_distances",
    "p_all_bonded_neighbors_oxygen",
    "tm_all_bonded_neighbors_oxygen",
    "p_coordination_eq_4",
    "tm_coordination_in_4_5_6",
)


def _audit_prefilter(path: str) -> set[int]:
    selected: set[int] = set()
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            tm_histogram = json.loads(row["tm_coord_hist"])
            if (
                float(row["po4_frac"]) == 1.0
                and all(int(count) in {4, 5, 6} for count in tm_histogram)
                and int(row["abnormal_neighbors"]) == 0
            ):
                selected.add(int(row["idx"]))
    return selected


def _evaluate(task: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], list[str]]:
    index, record = task
    elements, frac, lattice = _parse_structure(record["structure"])
    feasibility = evaluate_feasibility(elements, frac, lattice, None)
    failed = [name for name in GEOMETRY_CHECKS if not feasibility["checks"][name]]
    if failed:
        return index, {}, failed
    types = np.asarray(
        [DEFAULT_SPEC.element_to_index[element] for element in elements],
        dtype=np.int16,
    )
    order = _canonical_order(types, frac)
    return index, {
        "types": types[order],
        "frac": frac[order],
        "lattice": lattice,
        "energy": float(record["optimized_energy_per_atom"]),
        "id": str(record["material_id"]),
        "formula": str(record["formula"]),
    }, []


def build_quality_dataset(
    raw_path: str,
    split_path: str,
    audit_csv: str,
    output_path: str,
    workers: int,
) -> dict[str, Any]:
    with open(split_path) as handle:
        split_doc = json.load(handle)
    split_code = {"train": 0, "val": 1, "test": 2}
    index_to_split = {
        int(index): split_code[name]
        for name, indices in split_doc["splits"].items()
        for index in indices
    }
    audit_selected = _audit_prefilter(audit_csv)
    tasks: list[tuple[int, dict[str, Any]]] = []
    rejection_counts: Counter[str] = Counter()
    total = 0
    with open(raw_path, "rb") as handle:
        for index, line in enumerate(handle):
            total += 1
            record = orjson.loads(line)
            if "error_msg" in record:
                rejection_counts["reported_relaxation_error"] += 1
                continue
            if index not in audit_selected:
                rejection_counts["audit_coordination_prefilter"] += 1
                continue
            tasks.append((index, record))

    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        evaluated = pool.imap(_evaluate, tasks, chunksize=8)
        accepted: list[tuple[int, dict[str, Any]]] = []
        for completed, (index, record, failed) in enumerate(evaluated, start=1):
            if failed:
                for name in failed:
                    rejection_counts[name] += 1
            else:
                accepted.append((index, record))
            if completed % 1000 == 0:
                print(
                    json.dumps(
                        {"evaluated": completed, "accepted": len(accepted)}
                    ),
                    flush=True,
                )

    accepted.sort(key=lambda item: item[0])
    all_types: list[np.ndarray] = []
    all_frac: list[np.ndarray] = []
    lattices: list[np.ndarray] = []
    energies: list[float] = []
    splits: list[int] = []
    offsets = [0]
    ids: list[str] = []
    formulas: list[str] = []
    source_indices: list[int] = []
    for index, record in accepted:
        all_types.append(record["types"])
        all_frac.append(record["frac"])
        lattices.append(record["lattice"])
        energies.append(record["energy"])
        splits.append(index_to_split[index])
        offsets.append(offsets[-1] + len(record["types"]))
        ids.append(record["id"])
        formulas.append(record["formula"])
        source_indices.append(index)

    filter_definition = {
        "reported_relaxation_error_absent": True,
        "exact_required_checks": list(GEOMETRY_CHECKS),
        "family_or_parent_filter": False,
        "composition_objective_filter": False,
    }
    payload: dict[str, Any] = {
        "version": "naxl-packed-geometry-quality-v2",
        "source": str(Path(raw_path).resolve()),
        "source_first_1MiB_sha256": _first_mib_sha256(raw_path),
        "split_source": str(Path(split_path).resolve()),
        "audit_source": str(Path(audit_csv).resolve()),
        "filter_definition": filter_definition,
        "element_to_index": DEFAULT_SPEC.element_to_index,
        "types": torch.from_numpy(np.concatenate(all_types)),
        "frac_coords": torch.from_numpy(np.concatenate(all_frac)),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "lattices": torch.from_numpy(np.stack(lattices)),
        "optimized_energy_per_atom": torch.tensor(energies, dtype=torch.float32),
        "split": torch.tensor(splits, dtype=torch.uint8),
        "ids": ids,
        "formulas": formulas,
        "source_indices": torch.tensor(source_indices, dtype=torch.int64),
        "unknown_records": [],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    sizes = torch.diff(payload["offsets"])
    manifest = {
        "version": payload["version"],
        "source": payload["source"],
        "source_first_1MiB_sha256": payload["source_first_1MiB_sha256"],
        "filter_definition": filter_definition,
        "n_source_records": total,
        "n_prefilter_tasks": len(tasks),
        "n_structures": len(accepted),
        "n_atoms": int(payload["types"].numel()),
        "n_min": int(sizes.min()),
        "n_max": int(sizes.max()),
        "split_counts": {
            name: int((payload["split"] == code).sum())
            for name, code in split_code.items()
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "element_to_index": DEFAULT_SPEC.element_to_index,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build_quality_dataset(
        args.raw, args.split, args.audit_csv, args.out, args.workers
    )


if __name__ == "__main__":
    main()

"""Pack the complete NaCathode JSONL for broad fixed-composition Flow training.

This is deliberately chemistry-agnostic: it does not select a target cathode
formula.  A target composition is supplied only at inference through a
condition.  Records are excluded solely when they cannot represent an ordered
periodic atomistic structure (malformed cell/sites, partial occupancy, or an
optional explicit atom-count safety limit).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


CORE_ELEMENTS = ("Na", "Fe", "P", "O")


def parse_structure(record: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    structure = record.get("optimized_structure") or record.get("structure")
    if not isinstance(structure, dict):
        raise ValueError("missing structure")
    lattice = np.asarray(structure["lattice"]["matrix"], dtype=np.float32)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) < 1e-6:
        raise ValueError("invalid lattice")
    elements, frac = [], []
    for site in structure["sites"]:
        species = site["species"]
        if len(species) != 1 or abs(float(species[0].get("occu", 1.0)) - 1.0) > 1e-8:
            raise ValueError("partial_or_mixed_occupancy")
        elements.append(str(species[0]["element"]))
        frac.append(site["abc"])
    coords = np.mod(np.asarray(frac, dtype=np.float32), 1.0)
    if not elements or coords.shape != (len(elements), 3) or not np.isfinite(coords).all():
        raise ValueError("invalid sites")
    return elements, coords, lattice


def canonical_order(types: np.ndarray, frac: np.ndarray) -> np.ndarray:
    q = np.floor(np.mod(frac, 1.0) * 4096.0).astype(np.int64)
    return np.lexsort((q[:, 2], q[:, 1], q[:, 0], types))


def split_for_group(group: str, seed: int) -> int:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") % 100
    return 0 if value < 80 else (1 if value < 90 else 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max-atoms", type=int, default=256,
                        help="explicit compute safety limit; set 0 to disable")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    source = Path(args.input)
    max_atoms = None if args.max_atoms == 0 else args.max_atoms
    if max_atoms is not None and max_atoms < 2:
        parser.error("--max-atoms must be 0 or at least 2")

    elements_seen, rejected = set(), Counter()
    source_hash = hashlib.sha256()
    with source.open("rb") as handle:
        for raw in handle:
            source_hash.update(raw)
            try:
                record = json.loads(raw)
                elements, _, _ = parse_structure(record)
                if max_atoms is not None and len(elements) > max_atoms:
                    raise ValueError("atom_count_limit")
                elements_seen.update(elements)
            except Exception as error:
                rejected[str(error)] += 1
    ordered_elements = list(CORE_ELEMENTS) + sorted(elements_seen - set(CORE_ELEMENTS))
    element_to_index = {element: index + 1 for index, element in enumerate(ordered_elements)}

    all_types, all_frac, lattices, ids, parents, formulas, splits = [], [], [], [], [], [], []
    offsets = [0]
    with source.open() as handle:
        for line_number, raw in enumerate(handle):
            try:
                record = json.loads(raw)
                elements, frac, lattice = parse_structure(record)
                if max_atoms is not None and len(elements) > max_atoms:
                    continue
                types = np.asarray([element_to_index[element] for element in elements], dtype=np.int16)
                order = canonical_order(types, frac)
                types, frac = types[order], frac[order]
            except Exception:
                continue
            # Source dataset plus reduced formula is a conservative provenance group.
            # It prevents the same declared material family from crossing splits when
            # no explicit parent identifier is supplied by the source JSONL.
            group = f"{record.get('source_dataset', 'unknown')}|{record.get('formula', 'unknown')}"
            all_types.append(types); all_frac.append(frac); lattices.append(lattice)
            ids.append(str(record.get("material_id", line_number))); parents.append(group)
            formulas.append(str(record.get("formula", "unknown"))); splits.append(split_for_group(group, args.seed))
            offsets.append(offsets[-1] + len(types))
            if (line_number + 1) % 2000 == 0:
                print(json.dumps({"processed": line_number + 1, "accepted": len(ids)}), flush=True)
    if not ids:
        raise RuntimeError("no ordered periodic structures were accepted")
    payload = {
        "version": "nacathode-broad-conditional-flow-v1",
        "source": str(source.resolve()), "source_sha256": source_hash.hexdigest(),
        "element_to_index": element_to_index,
        "types": torch.from_numpy(np.concatenate(all_types)),
        "frac_coords": torch.from_numpy(np.concatenate(all_frac)),
        "lattices": torch.from_numpy(np.stack(lattices)),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "ids": ids, "parents": parents, "formulas": formulas,
        "split": torch.tensor(splits, dtype=torch.uint8),
        "filter_definition": "All ordered periodic records; no chemistry/formula selection. Excludes malformed/disordered records and only the explicit atom-count safety limit.",
        "split_definition": "deterministic source_dataset|formula provenance-group split, 80/10/10",
        "split_seed": args.seed,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    manifest = {
        "version": payload["version"], "source_sha256": payload["source_sha256"],
        "accepted": len(ids), "rejected": dict(rejected), "max_atoms": max_atoms,
        "n_atoms_min": int(torch.diff(payload["offsets"]).min()),
        "n_atoms_max": int(torch.diff(payload["offsets"]).max()),
        "vocabulary_size": len(element_to_index), "elements": ordered_elements,
        "split_counts": {name: int((payload["split"] == code).sum()) for name, code in (("train", 0), ("val", 1), ("test", 2))},
        "parent_group_count": len(set(parents)),
        "contains_energy_labels": False,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

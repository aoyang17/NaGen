"""Packed Na-Fe-P-O dataset for joint (N, A, X, L) generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch
from torch.utils.data import Dataset

from .spec import DEFAULT_SPEC


def _first_mib_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def _parse_structure(structure: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    lattice = np.asarray(structure["lattice"]["matrix"], dtype=np.float32)
    elements: list[str] = []
    frac: list[list[float]] = []
    for site in structure["sites"]:
        species = site["species"]
        dominant = max(species, key=lambda item: float(item.get("occu", 1.0)))
        elements.append(dominant["element"])
        frac.append(site["abc"])
    coords = np.mod(np.asarray(frac, dtype=np.float32), 1.0)
    return elements, coords, lattice


def _canonical_order(types: np.ndarray, frac: np.ndarray) -> np.ndarray:
    # Stable deterministic site ordering; no family or graph label is used.
    quantized = np.floor(np.mod(frac, 1.0) * 4096.0).astype(np.int64)
    return np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0], types))


def build_packed_dataset(raw_path: str, split_path: str, output_path: str) -> dict[str, Any]:
    with open(split_path) as handle:
        split_doc = json.load(handle)
    split_code = {"train": 0, "val": 1, "test": 2}
    index_to_split: dict[int, int] = {}
    for name, indices in split_doc["splits"].items():
        for index in indices:
            index_to_split[int(index)] = split_code[name]

    element_to_index = DEFAULT_SPEC.element_to_index
    all_types: list[np.ndarray] = []
    all_frac: list[np.ndarray] = []
    lattices: list[np.ndarray] = []
    energies: list[float] = []
    splits: list[int] = []
    offsets = [0]
    ids: list[str] = []
    formulas: list[str] = []
    unknown_records: list[dict[str, Any]] = []

    with open(raw_path, "rb") as handle:
        for index, line in enumerate(handle):
            record = orjson.loads(line)
            elements, frac, lattice = _parse_structure(record["structure"])
            unknown = sorted(set(elements) - set(element_to_index))
            if unknown:
                unknown_records.append({"idx": index, "elements": unknown})
                continue
            types = np.asarray([element_to_index[e] for e in elements], dtype=np.int16)
            order = _canonical_order(types, frac)
            types = types[order]
            frac = frac[order]
            all_types.append(types)
            all_frac.append(frac)
            lattices.append(lattice)
            energies.append(float(record["optimized_energy_per_atom"]))
            splits.append(index_to_split[index])
            offsets.append(offsets[-1] + len(types))
            ids.append(str(record["material_id"]))
            formulas.append(str(record["formula"]))
            if (index + 1) % 2000 == 0:
                print(f"[packed] {index + 1} records", flush=True)

    payload: dict[str, Any] = {
        "version": "naxl-packed-v1",
        "source": os.path.abspath(raw_path),
        "source_first_1MiB_sha256": _first_mib_sha256(raw_path),
        "split_source": os.path.abspath(split_path),
        "element_to_index": element_to_index,
        "types": torch.from_numpy(np.concatenate(all_types)),
        "frac_coords": torch.from_numpy(np.concatenate(all_frac)),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "lattices": torch.from_numpy(np.stack(lattices)),
        "optimized_energy_per_atom": torch.tensor(energies, dtype=torch.float32),
        "split": torch.tensor(splits, dtype=torch.uint8),
        "ids": ids,
        "formulas": formulas,
        "unknown_records": unknown_records,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    manifest = {
        "version": payload["version"],
        "source": payload["source"],
        "source_first_1MiB_sha256": payload["source_first_1MiB_sha256"],
        "n_structures": len(ids),
        "n_atoms": int(payload["types"].numel()),
        "n_min": int(torch.diff(payload["offsets"]).min()),
        "n_max": int(torch.diff(payload["offsets"]).max()),
        "split_counts": {
            name: int((payload["split"] == code).sum()) for name, code in split_code.items()
        },
        "element_to_index": element_to_index,
        "unknown_records": unknown_records,
    }
    with open(output.with_suffix(".manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return payload


class PackedCrystalDataset(Dataset):
    def __init__(self, path: str, split: str = "train") -> None:
        self.data = torch.load(path, map_location="cpu", weights_only=False)
        code = {"train": 0, "val": 1, "test": 2}[split]
        self.indices = torch.where(self.data["split"] == code)[0].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        start = int(self.data["offsets"][index])
        stop = int(self.data["offsets"][index + 1])
        return {
            "index": index,
            "types": self.data["types"][start:stop].long(),
            "frac_coords": self.data["frac_coords"][start:stop].float(),
            "lattice": self.data["lattices"][index].float(),
            "N": stop - start,
        }


def collate_crystals(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    n_max = max(item["N"] for item in batch)
    batch_size = len(batch)
    types = torch.zeros(batch_size, n_max, dtype=torch.long)
    frac = torch.zeros(batch_size, n_max, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, n_max, dtype=torch.bool)
    for row, item in enumerate(batch):
        n_atoms = item["N"]
        types[row, :n_atoms] = item["types"]
        frac[row, :n_atoms] = item["frac_coords"]
        mask[row, :n_atoms] = True
    return {
        "index": torch.tensor([item["index"] for item in batch]),
        "types": types,
        "frac_coords": frac,
        "lattice": torch.stack([item["lattice"] for item in batch]),
        "mask": mask,
        "N": mask.sum(dim=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_packed_dataset(args.raw, args.split, args.out)


if __name__ == "__main__":
    main()

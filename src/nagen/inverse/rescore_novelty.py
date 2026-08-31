"""Recompute descriptor novelty after any geometry-changing stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import torch
from pymatgen.core import Structure

from .novelty import NoveltyIndex, crystal_descriptor
from .spec import DEFAULT_SPEC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--stage", default="after_geometry_update")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.input) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    structures = [Structure.from_dict(record["structure"]) for record in records]
    device = torch.device(args.device)
    n_atoms = torch.tensor([len(structure) for structure in structures], device=device)
    n_max = int(n_atoms.max())
    mask = torch.arange(n_max, device=device)[None, :] < n_atoms[:, None]
    types = torch.zeros(len(records), n_max, dtype=torch.long, device=device)
    frac = torch.zeros(len(records), n_max, 3, device=device)
    lattice = torch.zeros(len(records), 3, 3, device=device)
    element_to_index = DEFAULT_SPEC.element_to_index
    for row, structure in enumerate(structures):
        n = len(structure)
        types[row, :n] = torch.tensor(
            [element_to_index[str(specie)] for specie in structure.species],
            device=device,
        )
        frac[row, :n] = torch.tensor(
            structure.frac_coords, dtype=torch.float32, device=device
        )
        lattice[row] = torch.tensor(
            structure.lattice.matrix, dtype=torch.float32, device=device
        )
    novelty = NoveltyIndex.load(args.novelty_index, device)
    descriptor = crystal_descriptor(types, frac, lattice, mask, novelty.config)
    scores, nearest_indices = novelty.score(descriptor)
    for row, record in enumerate(records):
        record["novelty_score_before_rescore"] = record.get("novelty_score")
        nearest = int(nearest_indices[row])
        record["novelty_score"] = float(scores[row])
        record["nearest_reference_index"] = nearest
        record["nearest_reference_id"] = novelty.ids[nearest]
        record["nearest_reference_formula"] = novelty.formulas[nearest]
        record["novelty_geometry_stage"] = args.stage
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    values = [record["novelty_score"] for record in records]
    summary = {
        "candidate_count": len(records),
        "minimum": min(values),
        "median": median(values),
        "mean": mean(values),
        "maximum": max(values),
        "zero_or_negative": sum(value <= 0.0 for value in values),
        "output": str(output),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

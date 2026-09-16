"""Extract condition-specific known structures from a broad packed Flow dataset.

This is an evaluation/reference operation, never a restriction on the broad
Flow training corpus.  It writes NAXL JSONL accepted by the existing motif,
novelty and reference-energy tools.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--elements", nargs="+", default=("Na", "Fe", "P", "O"))
    parser.add_argument("--require", nargs="*", default=("Na", "Fe", "P", "O"),
                        help="elements each record must contain; pass --require with no values for hull competitors")
    parser.add_argument("--family", default="phosphate")
    args = parser.parse_args()
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    names = {int(index): element for element, index in data["element_to_index"].items()}
    split_code = {"train": 0, "val": 1, "test": 2}
    allowed, required = set(args.elements), set(args.require)
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("x") as handle:
        for index, identifier in enumerate(data["ids"]):
            if args.split != "all" and int(data["split"][index]) != split_code[args.split]:
                continue
            start, stop = int(data["offsets"][index]), int(data["offsets"][index + 1])
            elements = [names[int(value)] for value in data["types"][start:stop]]
            if not set(elements) <= allowed or not required <= set(elements):
                continue
            row = {"material_id": str(identifier), "A": elements,
                   "X_frac": data["frac_coords"][start:stop].numpy(),
                   "L_matrix": data["lattices"][index].numpy(), "family": args.family,
                   "source_formula": data.get("formulas", [None] * len(data["ids"]))[index],
                   "source_parent": data.get("parents", [None] * len(data["ids"]))[index],
                   "source_split": int(data["split"][index])}
            handle.write(json.dumps(row, default=lambda value: value.tolist() if hasattr(value, "tolist") else value) + "\n")
            count += 1
    print(json.dumps({"status": "completed", "count": count, "split": args.split,
                      "allowed_elements": sorted(allowed), "required_elements": sorted(required)}, indent=2))


if __name__ == "__main__":
    main()

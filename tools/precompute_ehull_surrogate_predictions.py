"""Precompute frozen E_hull-surrogate estimates aligned to raw source indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nagen.surrogate.ehull_mace import (
    CrystalGraphDataset,
    collate_graphs,
    load_surrogate,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dataset", required=True)
    parser.add_argument("--surrogate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    graph_path = Path(args.graph_dataset)
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    model, manifest = load_surrogate(args.surrogate, args.device)
    outputs = torch.empty(len(data["ids"]), dtype=torch.float32)
    for split_code in (0, 1, 2):
        dataset = CrystalGraphDataset(data, split_code)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=args.device.startswith("cuda"),
            collate_fn=collate_graphs,
        )
        for batch in loader:
            indices = batch["index"]
            values = model({
                key: value.to(args.device, non_blocking=True)
                for key, value in batch.items()
            })
            outputs[indices] = values.cpu().float()
        print(json.dumps({"split_code": split_code, "count": len(dataset)}), flush=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_outputs = outputs.clone()
    outputs = outputs.clamp_min(0.0)
    payload = {
        "version": "nagen-ehull-surrogate-predictions-v1",
        "source_graph_dataset": str(graph_path.resolve()),
        "surrogate_checkpoint": str(Path(args.surrogate).resolve()),
        "surrogate_checkpoint_version": manifest["version"],
        "count": len(outputs),
        "E_hull_surrogate": outputs,
        "E_hull_surrogate_raw": raw_outputs,
        "mean": float(outputs.mean()),
        "std": float(outputs.std()),
        "raw_min": float(raw_outputs.min()),
        "min": float(outputs.min()),
        "max": float(outputs.max()),
    }
    torch.save(payload, output)
    print(json.dumps({
        key: value for key, value in payload.items()
        if not key.startswith("E_hull_surrogate")
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Auditable exact rejection sampling of generated ``N,A`` proposals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ._io import atomic_json as _atomic_json
from ._io import sha256_file as _sha256
from .composition_sampling import count_signature, exact_composition_passes
from .constraints import composition_metrics
from .generate import load_model, sample_atom_counts, variable_random_source
from .sample import integrate_flow
from .spec import DEFAULT_SPEC


def _record(elements: list[str], type_indices: list[int]) -> dict[str, Any]:
    counts = Counter(elements)
    metrics = composition_metrics(elements)
    return {
        "signature": list(count_signature(elements)),
        "counts": {element: counts[element] for element in DEFAULT_SPEC.elements},
        "N": len(elements),
        "elements": elements,
        "type_indices": type_indices,
        "composition_metrics": metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_model(str(checkpoint_path), device, use_ema=True)
    index_to_element = {
        int(index): element for element, index in checkpoint["element_to_index"].items()
    }
    output = Path(args.out)
    checkpoint_hash = _sha256(checkpoint_path)
    unique: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    proposals = 0
    accepted = 0
    batch_index = 0
    if output.is_file() and args.resume:
        previous = json.loads(output.read_text())
        if previous.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError("cannot resume with a different checkpoint")
        if previous.get("seed") != args.seed or previous.get("ode_steps") != args.ode_steps:
            raise RuntimeError("cannot resume with different sampling settings")
        proposals = int(previous.get("proposals", 0))
        accepted = int(previous.get("accepted_proposals", 0))
        batch_index = int(previous.get("completed_batches", 0))
        for record in previous.get("unique_compositions", []):
            unique[tuple(record["signature"])] = record

    payload: dict[str, Any] = {
        "version": "generated-composition-pool-v1",
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "ema": "ema_model" in checkpoint,
        "ode_steps": args.ode_steps,
        "integrator": "midpoint",
        "maximum_proposals": args.maximum_proposals,
        "maximum_unique": args.maximum_unique,
        "batch_size": args.batch_size,
        "completed_batches": batch_index,
        "proposals": proposals,
        "accepted_proposals": accepted,
        "unique_count": len(unique),
        "status": "running",
        "unique_compositions": list(unique.values()),
    }
    while proposals < args.maximum_proposals and len(unique) < args.maximum_unique:
        current = min(args.batch_size, args.maximum_proposals - proposals)
        generator = torch.Generator(device=device).manual_seed(args.seed + batch_index)
        n_atoms = sample_atom_counts(checkpoint["n_histogram"], current, generator).to(device)
        source, mask = variable_random_source(
            n_atoms, model.config.vocab_size, device, generator
        )
        with torch.no_grad():
            terminal = integrate_flow(
                model, source, mask, steps=args.ode_steps, method="midpoint"
            )
            generated = terminal.atom.argmax(dim=-1) + 1
            generated = generated.masked_fill(~mask, 0)
        for row in range(current):
            indices = generated[row, mask[row]].detach().cpu().tolist()
            elements = [index_to_element[int(index)] for index in indices]
            if exact_composition_passes(elements):
                accepted += 1
                signature = count_signature(elements)
                unique.setdefault(signature, _record(elements, indices))
                if len(unique) >= args.maximum_unique:
                    break
        proposals += current
        batch_index += 1
        status = (
            "complete" if len(unique) >= args.maximum_unique else
            "running"
        )
        payload.update({
            "completed_batches": batch_index, "proposals": proposals,
            "accepted_proposals": accepted, "unique_count": len(unique),
            "status": status, "unique_compositions": list(unique.values()),
        })
        _atomic_json(output, payload)
        print(json.dumps({
            "proposals": proposals, "accepted": accepted,
            "unique": len(unique), "status": status,
        }), flush=True)

    final_status = (
        "complete" if len(unique) >= args.maximum_unique else
        "sufficient_partial" if len(unique) >= 6 else
        "insufficient_unique_compositions"
    )
    payload["status"] = final_status
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--maximum-proposals", type=int, default=200_000)
    parser.add_argument("--maximum-unique", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"], "proposals": report["proposals"],
        "accepted_proposals": report["accepted_proposals"],
        "unique_count": report["unique_count"], "output": args.out,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

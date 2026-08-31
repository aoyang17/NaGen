"""Consistent CHGNet single-point energies for the raw reference corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chgnet.model.model import CHGNet
from pymatgen.core import Structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reference_energy_shard_{args.shard:02d}.jsonl"
    summary_path = output_dir / f"summary_shard_{args.shard:02d}.json"
    model = CHGNet.load(use_device=args.device)
    completed = 0
    errors = 0
    buffer: list[tuple[dict, Structure]] = []

    def flush(output) -> None:
        nonlocal completed, errors, buffer
        if not buffer:
            return
        metadata = [item[0] for item in buffer]
        structures = [item[1] for item in buffer]
        try:
            predictions = model.predict_structure(
                structures, task="e", batch_size=args.batch_size
            )
            if isinstance(predictions, dict):
                predictions = [predictions]
            for meta, structure, prediction in zip(
                metadata, structures, predictions, strict=True
            ):
                composition = structure.composition
                output.write(
                    json.dumps(
                        {
                            **meta,
                            "N": len(structure),
                            "reduced_formula": composition.reduced_formula,
                            "atomic_fractions": {
                                element.symbol: float(amount / composition.num_atoms)
                                for element, amount in composition.items()
                            },
                            "chgnet_energy_eV_atom": float(prediction["e"]),
                            "method": "CHGNet-0.3.0-single-point",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                completed += 1
        except Exception:
            # Isolate any malformed/unsupported structure so one bad record
            # does not discard the rest of a batch.
            for meta, structure in buffer:
                try:
                    prediction = model.predict_structure(structure, task="e")
                    composition = structure.composition
                    output.write(
                        json.dumps(
                            {
                                **meta,
                                "N": len(structure),
                                "reduced_formula": composition.reduced_formula,
                                "atomic_fractions": {
                                    element.symbol: float(
                                        amount / composition.num_atoms
                                    )
                                    for element, amount in composition.items()
                                },
                                "chgnet_energy_eV_atom": float(prediction["e"]),
                                "method": "CHGNet-0.3.0-single-point",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    completed += 1
                except Exception as error:
                    output.write(
                        json.dumps(
                            {
                                **meta,
                                "error": f"{type(error).__name__}: {error}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    errors += 1
        output.flush()
        buffer = []

    with open(args.input) as source, output_path.open("w") as output:
        for index, line in enumerate(source):
            if index % args.num_shards != args.shard or not line.strip():
                continue
            raw = json.loads(line)
            try:
                structure = Structure.from_dict(raw["structure"])
                buffer.append(
                    (
                        {
                            "source_index": index,
                            "material_id": raw["material_id"],
                            "source_formula": raw["formula"],
                        },
                        structure,
                    )
                )
            except Exception as error:
                output.write(
                    json.dumps(
                        {
                            "source_index": index,
                            "material_id": raw.get("material_id"),
                            "error": f"{type(error).__name__}: {error}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errors += 1
            if len(buffer) >= args.batch_size:
                flush(output)
                if completed % 256 < args.batch_size:
                    print(
                        json.dumps(
                            {
                                "shard": args.shard,
                                "completed": completed,
                                "errors": errors,
                            }
                        ),
                        flush=True,
                    )
        flush(output)

    summary = {
        "shard": args.shard,
        "completed": completed,
        "errors": errors,
        "output": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

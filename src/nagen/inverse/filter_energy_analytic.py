"""Retain structures that pass the exact analytic audit after energy guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    records = []
    for source in args.input:
        with open(source) as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    selected = [
        record
        for record in records
        if bool(record.get("analytical_feasible_after_energy_guidance", False))
        != args.invert
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "input_count": len(records),
        "selected_analytic_after_energy_guidance": len(selected),
        "selection_inverted": args.invert,
        "inputs": args.input,
        "output": str(output),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

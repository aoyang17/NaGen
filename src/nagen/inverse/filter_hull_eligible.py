"""Select candidates whose compositions are covered by a reference hull."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    with open(args.input) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    eligible = [
        record
        for record in records
        if record.get("local_hull", {}).get(
            "composition_inside_reference_hull", False
        )
        or record.get("local_hull", {}).get("status") == "evaluated"
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in eligible:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "input_count": len(records),
        "hull_composition_eligible": len(eligible),
        "excluded_outside_reference_composition_hull": len(records) - len(eligible),
        "output": str(output),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

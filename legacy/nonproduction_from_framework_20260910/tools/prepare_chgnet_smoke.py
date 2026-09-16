"""Extract complete prefix records for plumbing tests, never a scientific split."""
import argparse
import json
from itertools import islice
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.raw) as handle:
        rows = [json.loads(line) for line in islice(handle, args.limit)]
    target = [r for r in rows if {"Fe", "P", "O"}.issubset(
        {s["species"][0]["element"] for s in r["structure"]["sites"]})
        and {s["species"][0]["element"] for s in r["structure"]["sites"]} <= {"Na", "Fe", "P", "O"}]
    for name, group in (("train", [r for i,r in enumerate(target) if i%3]),
                        ("candidates", [r for i,r in enumerate(target) if not i%3])):
        with (out/f"{name}.jsonl").open("x") as handle:
            for r in group:
                # Deliberately drop ALL historical energy labels.
                handle.write(json.dumps({"material_id":r["material_id"],"structure":r["structure"]})+"\n")
    (out/"README.txt").write_text("SMOKE ONLY: prefix sample with interleaved split, related structures leak.\n"
        "Do not use for performance claims or final motif calibration. All original energy labels removed.\n")
    print(f"{len(target)} target structures from {len(rows)} prefix records")


if __name__ == "__main__":
    main()

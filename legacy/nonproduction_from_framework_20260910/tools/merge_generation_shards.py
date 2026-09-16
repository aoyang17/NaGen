"""Merge independently seeded UMA-DFlow generation shards deterministically."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True); parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.shard_dir); out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    all_rows, safe_rows, manifests = [], [], []
    for index in range(args.expected_shards):
        shard = root / f"shard_{index}"
        manifest = json.loads((shard / "manifest.json").read_text())
        if manifest.get("status") != "completed": raise ValueError(f"incomplete generation shard {index}")
        manifests.append(manifest); all_rows.extend(read(shard / "generated_all.jsonl")); safe_rows.extend(read(shard / "generated_safe.jsonl"))
    if len(all_rows) != args.expected_count: raise ValueError("generation coverage mismatch")
    ids = [row["material_id"] for row in all_rows]
    if len(set(ids)) != len(ids): raise ValueError("duplicate candidate identifiers across shards")
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("generated_all.jsonl", all_rows), ("generated_safe.jsonl", safe_rows)):
        with (out / name).open("x") as handle:
            for row in rows: handle.write(json.dumps(row) + "\n")
    summary = {"status": "completed", "attempted": len(all_rows), "safe_for_UMA_relax": len(safe_rows),
               "flow_sha256": {item["flow_sha256"] for item in manifests},
               "uma_model_hash": {item["uma"]["checkpoint_sha256"] for item in manifests},
               "shards": [str(root / f"shard_{index}") for index in range(args.expected_shards)]}
    if len(summary["flow_sha256"]) != 1 or len(summary["uma_model_hash"]) != 1: raise ValueError("mixed model provenance")
    summary["flow_sha256"] = next(iter(summary["flow_sha256"])); summary["uma_model_hash"] = next(iter(summary["uma_model_hash"]))
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

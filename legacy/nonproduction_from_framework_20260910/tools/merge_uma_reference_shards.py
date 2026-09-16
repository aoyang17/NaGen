"""Merge complete UMA reference shards, failing closed on protocol drift."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True); parser.add_argument("--pattern", default="reference_*.jsonl")
    parser.add_argument("--expected-shards", type=int, required=True); parser.add_argument("--expected-input-count", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.shard_dir); paths = sorted(root.glob(args.pattern))
    paths = [path for path in paths if not path.name.endswith(".failures.jsonl")]
    if len(paths) != args.expected_shards:
        raise ValueError(f"expected {args.expected_shards} label files, found {len(paths)}")
    rows, manifests = [], []
    for path in paths:
        manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
        if manifest["status"] != "completed" or manifest["failed"]:
            raise ValueError(f"incomplete reference shard: {path}")
        manifests.append(manifest)
        rows.extend(json.loads(line) for line in path.open() if line.strip())
    if sum(item["selected"] for item in manifests) != args.expected_input_count:
        raise ValueError("shard coverage does not equal frozen reference input count")
    if len(rows) != args.expected_input_count:
        raise ValueError("successful labels do not cover frozen reference input count")
    model_hashes = {item["model"]["state_sha256"] for item in manifests}
    protocols = {json.dumps({key: item["configuration"][key] for key in ("steps", "fmax", "task_name")}, sort_keys=True) for item in manifests}
    input_hashes = {item["input_sha256"] for item in manifests}
    if len(model_hashes) != 1 or len(protocols) != 1 or len(input_hashes) != 1:
        raise ValueError("mixed model, relaxation protocol, or frozen reference input")
    rows.sort(key=lambda row: row["id"])
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    summary = {"status": "complete", "count": len(rows), "labels_sha256": digest(out),
               "input_sha256": next(iter(input_hashes)), "model_hash": next(iter(model_hashes)),
               "protocol": json.loads(next(iter(protocols))), "shards": [str(path) for path in paths]}
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Merge complete candidate UMA-relaxation shards with provenance checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path): return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True); parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); root = Path(args.shard_dir); out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    pairs, structures, summaries = [], [], []
    for index in range(args.expected_shards):
        shard = root / f"shard_{index}"
        summary = json.loads((shard / "summary.json").read_text())
        if summary["failed"]: raise ValueError(f"candidate-relaxation failures in shard {index}")
        pairs.extend(rows(shard / "pairs.jsonl")); structures.extend(rows(shard / "relaxed.jsonl")); summaries.append(summary)
    if len(pairs) != args.expected_count or len(structures) != args.expected_count:
        raise ValueError("candidate-relaxation coverage mismatch")
    if len({row["id"] for row in pairs}) != len(pairs): raise ValueError("duplicate relaxed candidate IDs")
    model_hashes = {summary["model"]["state_sha256"] for summary in summaries}
    inputs = {summary["input_sha256"] for summary in summaries}
    profiles = {summary["profile_sha256"] for summary in summaries}
    protocols = {json.dumps({key: summary["configuration"][key] for key in ("steps", "fmax", "task_name")}, sort_keys=True) for summary in summaries}
    if any(len(values) != 1 for values in (model_hashes, inputs, profiles, protocols)):
        raise ValueError("mixed UMA model, input, profile, or protocol")
    out.mkdir(parents=True, exist_ok=False)
    for name, content in (("pairs.jsonl", pairs), ("relaxed.jsonl", structures)):
        with (out / name).open("x") as handle:
            for row in content: handle.write(json.dumps(row) + "\n")
    summary = {"status": "completed", "completed": len(pairs), "failed": 0,
               "converged": sum(row["converged"] for row in pairs), "model": summaries[0]["model"],
               "input_sha256": next(iter(inputs)), "profile_sha256": next(iter(profiles)),
               "protocol": json.loads(next(iter(protocols))),
               "all_gates_passed": {phase: sum(row["gates"][phase]["passed"] for row in pairs) for phase in ("before", "after")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

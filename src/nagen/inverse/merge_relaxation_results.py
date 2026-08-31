"""Merge UMA relaxation pilot shards and apply the final-force semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    shards = [json.loads(Path(path).read_text()) for path in args.inputs]
    records = sorted(
        (record for shard in shards for record in shard["records"]),
        key=lambda record: record["sample_id"],
    )
    ids = [record["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("relaxation shards contain duplicate sample IDs")
    for record in records:
        converged = bool(record["bfgs_converged"])
        record["converged"] = converged
        record["status"] = (
            "converged" if converged and record["fire_converged"] else
            "converged_after_fire_max_steps" if converged else
            "bfgs_not_converged"
        )
    summary = {
        "selected": len(records),
        "relaxation_converged": sum(record["converged"] for record in records),
        "post_relaxation_geometry_pass": sum(
            record["geometry_pass"] for record in records
        ),
        "converged_and_geometry_pass": sum(
            record["converged"] and record["geometry_pass"]
            for record in records
        ),
    }
    output = {
        "version": "uma-relaxation-pilot-merged-v1",
        "sources": [str(Path(path).resolve()) for path in args.inputs],
        "uma_checkpoint": shards[0]["uma_checkpoint"],
        "task_name": shards[0]["task_name"],
        "config": shards[0]["config"],
        "summary": summary,
        "records": records,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

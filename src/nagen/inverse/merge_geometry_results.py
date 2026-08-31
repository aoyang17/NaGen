"""Merge disjoint geometry pilot shards into one auditable result."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


GEOMETRY_CHECKS = (
    "positive_lattice_determinant",
    "volume_per_atom_range",
    "minimum_pbc_distances",
    "p_coordination_eq_4",
    "fe_coordination_in_4_5_6",
)


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
        raise ValueError("geometry result shards contain duplicate sample IDs")
    summary = {
        "n": len(records),
        "geometry_pass": sum(
            all(record["feasibility"]["checks"][key] for key in GEOMETRY_CHECKS)
            for record in records
        ),
        "check_pass_counts": {
            key: sum(record["feasibility"]["checks"][key] for record in records)
            for key in GEOMETRY_CHECKS
        },
        "median_minimum_distance_A": statistics.median(
            record["feasibility"]["values"]["minimum_distance_A"]
            for record in records
        ),
    }
    output = {
        "version": "paired-geometry-baseline-merged-v1",
        "sources": [str(Path(path).resolve()) for path in args.inputs],
        "checkpoint": shards[0]["checkpoint"],
        "seed": shards[0]["seed"],
        "ode_steps": shards[0]["ode_steps"],
        "restoration": shards[0]["restoration"],
        "summary": summary,
        "records": records,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

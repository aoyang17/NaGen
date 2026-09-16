"""Summarize the paired 12-vs-12 solver comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def mean(rows, getter):
    values = [getter(row) for row in rows]
    return statistics.mean(values) if values else None


def load_records(paths):
    records = []
    for path in paths:
        records.extend(json.loads(path.read_text())["records"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir
    lbfgs = load_records(sorted((root / "lbfgs").glob("shard*.json")))
    ipopt = load_records(sorted((root / "ipopt").glob("*.json")))
    concise = []
    for path in sorted((root / "ipopt").glob("shard*.log")):
        shard = int(path.stem.removeprefix("shard"))
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "sample_id" in row:
                concise.append({"shard": shard, **row})
    supplement = json.loads(
        (root / "ipopt" / "shard2_missing_sample2.json").read_text()
    )["records"][0]
    concise.append({
        "shard": 2,
        **{key: supplement[key] for key in (
            "sample_id", "status", "N", "formula", "wall_time_seconds",
            "maximum_constraint_violation", "all_constraints_feasible",
        )},
        "supplemental": True,
    })
    # JSON files shard0/1/3 plus the supplemental file contain 10 detailed
    # records.  The timed-out original shard2 log retains the two remaining
    # records' solver status and primary feasibility metric.
    exact_failure_names = sorted({
        name for row in lbfgs
        for name, passed in row["analytic_constraint_checks"]["checks"].items()
        if not passed
    })
    summary = {
        "design": {
            "paired_samples": 12,
            "same_seeds_compositions_and_screened_sources": True,
            "all_constraints_active_throughout": True,
            "flow_surrogate_uma_enabled": True,
            "gpus_per_solver": 4,
            "original_process_budget_seconds": 1200,
        },
        "lbfgs": {
            "completed_in_original_budget": len(lbfgs),
            "threshold_passes": sum(row["passes_threshold"] for row in lbfgs),
            "analytic_feasible": sum(
                row["analytic_constraint_checks"]["feasible"] for row in lbfgs
            ),
            "all_constraints_feasible": sum(
                row["all_constraints_feasible"] for row in lbfgs
            ),
            "mean_surrogate_e_hull": mean(
                lbfgs, lambda row: row["optimized_e_hull_surrogate"]
            ),
            "mean_minimum_distance_A": mean(
                lbfgs,
                lambda row: row["analytic_constraint_checks"]["values"]["minimum_distance_A"],
            ),
            "mean_distance_violation_count": mean(
                lbfgs,
                lambda row: len(row["analytic_constraint_checks"]["details"]["distance_violations"]),
            ),
            "failed_exact_check_types": exact_failure_names,
            "approx_original_makespan_seconds": max(
                path.stat().st_mtime for path in (root / "lbfgs").glob("shard*.json")
            ) - (root / "pids.txt").stat().st_mtime,
        },
        "ipopt": {
            "completed_in_original_budget": len(concise) - 1,
            "completed_after_supplement": len(concise),
            "status_counts": {
                status: sum(row["status"] == status for row in concise)
                for status in sorted({row["status"] for row in concise})
            },
            "all_constraints_feasible": sum(
                row["all_constraints_feasible"] for row in concise
            ),
            "median_maximum_scaled_constraint_violation": statistics.median(
                row["maximum_constraint_violation"] for row in concise
            ),
            "detailed_records_available": len(ipopt),
            "threshold_passes_detailed": sum(row["passes_threshold"] for row in ipopt),
            "analytic_feasible_detailed": sum(
                row["analytic_constraint_checks"]["feasible"] for row in ipopt
            ),
            "mean_surrogate_e_hull_detailed": mean(
                ipopt, lambda row: row["optimized_e_hull_surrogate"]
            ),
            "mean_minimum_distance_A_detailed": mean(
                ipopt,
                lambda row: row["analytic_constraint_checks"]["values"]["minimum_distance_A"],
            ),
            "mean_distance_violation_count_detailed": mean(
                ipopt,
                lambda row: len(row["analytic_constraint_checks"]["details"]["distance_violations"]),
            ),
            "mean_optimizer_wall_seconds": mean(
                concise, lambda row: row["wall_time_seconds"]
            ),
        },
        "decision": {
            "winner_for_current_shootingflow_inference": "lbfgs",
            "reason": (
                "Neither solver found a fully feasible structure; L-BFGS completed "
                "all 12 within the original budget while IPOPT completed 11, and "
                "IPOPT's full dense PyTorch Jacobian was substantially slower."
            ),
        },
        "ipopt_primary_records": concise,
    }
    target = root / "comparison_summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

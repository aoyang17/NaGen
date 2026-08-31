"""Summarize the locked four-configuration ShootingFlow pilot gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


GEOMETRY_CHECKS = (
    "positive_lattice_determinant", "volume_per_atom_range",
    "minimum_pbc_distances", "p_coordination_eq_4",
    "fe_coordination_in_4_5_6",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.pilot_dir)
    summaries = []
    for index in range(4):
        payload = json.loads((root / f"config_{index}.json").read_text())
        samples = payload["samples"]
        variants = {}
        for name in ("B0", "B1", "B2", "M0"):
            complete = [
                sample for sample in samples
                if name == "B0" or sample[name].get("status") == "complete"
            ]
            exact = [sample[name]["exact"]["feasibility"] for sample in complete]
            variants[name] = {
                "complete": len(complete),
                "analytic_without_hull_pass": sum(
                    all(value for key, value in item["checks"].items()
                        if key != "hull_threshold") for item in exact
                ),
                "geometry_pass": sum(
                    all(item["checks"][key] for key in GEOMETRY_CHECKS)
                    for item in exact
                ),
                "hull_threshold_pass": sum(
                    item["checks"]["hull_threshold"] for item in exact
                ),
                "check_pass_counts": {
                    key: sum(item["checks"][key] for item in exact)
                    for key in GEOMETRY_CHECKS
                },
                "median_E_hull_eV_atom": statistics.median(
                    sample[name]["values"]["E_hull"] for sample in complete
                ),
                "median_novelty": statistics.median(
                    sample[name]["values"]["novelty"] for sample in complete
                ),
                "median_minimum_distance_A": statistics.median(
                    item["values"]["minimum_distance_A"] for item in exact
                ),
            }
            if name != "B0":
                variants[name].update({
                    "median_paired_delta_E_hull_eV_atom": statistics.median(
                        sample[name]["values"]["E_hull"]
                        - sample["B0"]["values"]["E_hull"]
                        for sample in complete
                    ),
                    "median_paired_delta_novelty": statistics.median(
                        sample[name]["values"]["novelty"]
                        - sample["B0"]["values"]["novelty"]
                        for sample in complete
                    ),
                    "median_source_drift": statistics.median(
                        sample[name]["history"][-1]["source_drift"]
                        for sample in complete
                    ),
                })
        summaries.append({
            "config_index": index, "config": payload["config"],
            "objective_scales": payload["objective_scales"],
            "model_hash": payload["model_hash"], "variants": variants,
        })
    eligible = [
        summary for summary in summaries
        if summary["variants"]["M0"]["analytic_without_hull_pass"] > 0
    ]
    report = {
        "version": "shootingflow-pilot-summary-v1",
        "paired_sources": 32,
        "guided_paths": 96,
        "configurations": summaries,
        "eligible_configurations": [item["config_index"] for item in eligible],
        "selected_configuration": None,
        "pilot_gate_passed": bool(eligible),
        "main_256_authorized": bool(eligible),
        "stop_reason": None if eligible else (
            "all four M0 configurations have zero exact analytic-constraint passes; "
            "minimum-distance and P-coordination failures remain"
        ),
        "energy_policy": (
            "E_hull is minimized against a model-hash-matched UMA same-scale "
            "competing-phase hull; 0.150 eV/atom remains a strict "
            "final-candidate constraint"
        ),
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(target), "pilot_gate_passed": report["pilot_gate_passed"],
        "main_256_authorized": report["main_256_authorized"],
        "stop_reason": report["stop_reason"],
    }, indent=2))


if __name__ == "__main__":
    main()

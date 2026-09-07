"""Audit and gate a paired B0/B1/B2/M0 ShootingFlow campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from ._io import atomic_json


VARIANTS = ("B0", "B1", "B2", "M0")


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _complete_variant(record: dict[str, Any], name: str) -> bool:
    return record.get(name, {}).get("status") == "complete"


def _variant_summary(
    records: list[dict[str, Any]], name: str,
) -> dict[str, Any]:
    complete = [record for record in records if _complete_variant(record, name)]
    if not complete:
        return {"complete": 0}
    values = [record[name]["values"] for record in complete]
    exact = [record[name]["exact"] for record in complete]
    summary: dict[str, Any] = {
        "complete": len(complete),
        "fixed_A_preserved": sum(item["fixed_A_preserved"] for item in exact),
        "exact_geometry_pass": sum(item["geometry_pass"] for item in exact),
        "exact_hull_threshold_pass": sum(
            item["feasibility"]["checks"]["hull_threshold"] for item in exact
        ),
        "exact_all_constraints_pass": sum(
            item["feasibility"]["feasible"] for item in exact
        ),
        "median_E_hull_eV_atom": _median(
            [float(value["E_hull"]) for value in values]
        ),
        "median_novelty": _median(
            [float(value["novelty"]) for value in values]
        ),
        "check_pass_counts": {
            key: sum(
                item["feasibility"]["checks"].get(key, False) for item in exact
            )
            for key in sorted(exact[0]["feasibility"]["checks"])
        },
    }
    if name != "B0":
        paired = [record for record in complete if _complete_variant(record, "B0")]
        summary.update({
            "median_paired_delta_E_hull_eV_atom": _median([
                float(record[name]["values"]["E_hull"])
                - float(record["B0"]["values"]["E_hull"])
                for record in paired
            ]),
            "median_paired_delta_novelty": _median([
                float(record[name]["values"]["novelty"])
                - float(record["B0"]["values"]["novelty"])
                for record in paired
            ]),
            "median_source_drift": _median([
                float(record[name]["history"][-1]["source_drift"])
                for record in paired
            ]),
            "median_final_hull_gradient_norm": _median([
                float(record[name]["history"][-1]["hull_gradient_norm"])
                for record in paired
            ]),
            "median_final_novelty_gradient_norm": _median([
                float(record[name]["history"][-1]["novelty_gradient_norm"])
                for record in paired
            ]),
            "median_final_gradient_cosine": _median([
                float(record[name]["history"][-1]["gradient_cosine"])
                for record in paired
            ]),
            "median_final_mgda_hull_weight": _median([
                float(record[name]["history"][-1]["mgda_hull_weight"])
                for record in paired
            ]),
        })
    return summary


def build_report(campaign_directory: Path) -> dict[str, Any]:
    manifest = json.loads((campaign_directory / "manifest.json").read_text())
    record_paths = sorted((campaign_directory / "records").glob("sample_*.json"))
    records = [json.loads(path.read_text()) for path in record_paths]
    expected = int(manifest["count"])
    sample_ids = [int(record["sample_id"]) for record in records]
    expected_ids = list(range(
        int(manifest["start_index"]), int(manifest["start_index"]) + expected
    ))
    unique_ids = len(sample_ids) == len(set(sample_ids))
    exact_ids = sorted(sample_ids) == expected_ids
    variants = {name: _variant_summary(records, name) for name in VARIANTS}
    all_complete = (
        manifest.get("status") == "complete"
        and len(records) == expected
        and unique_ids
        and exact_ids
        and all(
            variants[name].get("complete") == expected for name in VARIANTS
        )
        and all(record.get("status") == "complete" for record in records)
    )
    all_fixed_A = all(
        variants[name].get("fixed_A_preserved") == expected for name in VARIANTS
    )
    scales = [float(value) for value in manifest["objective_scales"]]
    m0 = variants["M0"]
    delta_hull = float(m0.get("median_paired_delta_E_hull_eV_atom", math.inf))
    delta_novelty = float(m0.get("median_paired_delta_novelty", -math.inf))
    hull_improved = delta_hull < 0.0
    novelty_improved = delta_novelty > 0.0
    hull_not_collapsed = delta_hull <= 0.25 * scales[0]
    novelty_not_collapsed = delta_novelty >= -0.25 * scales[1]
    objective_gate = bool(
        (hull_improved and novelty_not_collapsed)
        or (novelty_improved and hull_not_collapsed)
    )
    geometry_gate = bool(
        m0.get("exact_geometry_pass", 0)
        > variants["B0"].get("exact_geometry_pass", 0)
    )
    finite_history = True
    for record in records:
        for name in ("B1", "B2", "M0"):
            for step in record.get(name, {}).get("history", []):
                for key in (
                    "hull_gradient_norm", "novelty_gradient_norm",
                    "gradient_cosine", "mgda_hull_weight", "source_drift",
                ):
                    if not math.isfinite(float(step[key])):
                        finite_history = False
    gate_checks = {
        "complete_paired_campaign": all_complete,
        "fixed_A_preserved": all_fixed_A,
        "finite_gradient_and_drift_history": finite_history,
        "M0_improves_at_least_one_objective_without_other_collapsing": objective_gate,
        "M0_exact_geometry_pass_exceeds_B0": geometry_gate,
    }
    gate_passed = all(gate_checks.values())
    return {
        "version": "paired-shooting-campaign-report-v1",
        "campaign_directory": str(campaign_directory.resolve()),
        "campaign_identity_sha256": manifest["campaign_identity_sha256"],
        "role": manifest["role"],
        "expected_records": expected,
        "observed_records": len(records),
        "objective_scales": scales,
        "configuration": manifest["configuration"],
        "variants": variants,
        "gate_policy": {
            "objective_noncollapse_tolerance": "0.25 pilot standard deviations",
            "geometry_policy": "M0 exact geometry pass count must exceed paired B0",
            "hull_threshold_policy": (
                "E_hull <= 0.150 eV/atom is reported here and remains mandatory "
                "after full UMA relaxation; it is not by itself the G5 improvement gate"
            ),
        },
        "gate_checks": gate_checks,
        "pilot_gate_passed": gate_passed,
        "main_256_authorized": bool(gate_passed and manifest["role"] == "pilot"),
        "frozen_main_inputs": ({
            "objective_scales": scales,
            "configuration": manifest["configuration"],
            "geometry_checkpoint": manifest["geometry_checkpoint"],
            "geometry_checkpoint_sha256": manifest["geometry_checkpoint_sha256"],
            "uma_hull_cache": manifest["uma_hull_cache"],
            "uma_hull_cache_sha256": manifest["uma_hull_cache_sha256"],
            "novelty_index": manifest["novelty_index"],
            "novelty_index_sha256": manifest["novelty_index_sha256"],
        } if gate_passed and manifest["role"] == "pilot" else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-directory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build_report(Path(args.campaign_directory))
    atomic_json(Path(args.out), report)
    print(json.dumps({
        "pilot_gate_passed": report["pilot_gate_passed"],
        "main_256_authorized": report["main_256_authorized"],
        "gate_checks": report["gate_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()

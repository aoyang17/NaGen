"""Report site-level structural quality without weakening the E_hull gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum-site-fraction", type=float, default=0.75)
    parser.add_argument("--maximum-pair-shortfall-fraction", type=float, default=0.10)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    records = []
    for record in payload["records"]:
        feasibility = record["feasibility"]
        sites = feasibility["details"]["coordination"]["sites"]
        p_counts = [site["coordination"] for site in sites if site["element"] == "P"]
        fe_counts = [site["coordination"] for site in sites if site["element"] == "Fe"]
        p_fraction = sum(count == 4 for count in p_counts) / max(1, len(p_counts))
        fe_fraction = sum(count in {4, 5, 6} for count in fe_counts) / max(1, len(fe_counts))
        checks = feasibility["checks"]
        violations = feasibility["details"]["distance_violations"]
        maximum_shortfall = max(
            (
                (item["minimum_A"] - item["distance_A"]) / item["minimum_A"]
                for item in violations
            ),
            default=0.0,
        )
        near_coordination = (
            p_fraction >= args.minimum_site_fraction
            and fe_fraction >= args.minimum_site_fraction
        )
        structurally_rankable = bool(
            record["converged"]
            and checks["positive_lattice_determinant"]
            and near_coordination
            and maximum_shortfall <= args.maximum_pair_shortfall_fraction
        )
        records.append({
            "sample_id": record["sample_id"],
            "relaxation_converged": record["converged"],
            "p_cn4_site_fraction": p_fraction,
            "fe_cn456_site_fraction": fe_fraction,
            "minimum_distance_A": feasibility["values"]["minimum_distance_A"],
            "maximum_pair_shortfall_fraction": maximum_shortfall,
            "volume_per_atom_A3": feasibility["values"]["volume_per_atom_A3"],
            "exact_geometry_pass": record["geometry_pass"],
            "near_coordination": near_coordination,
            "structurally_rankable": structurally_rankable,
            "final_candidate": False,
            "final_candidate_reason": "E_hull hard gate not evaluated/aligned",
        })
    output = {
        "version": "relaxation-structural-tradeoff-v1",
        "source": str(Path(args.input).resolve()),
        "policy": {
            "E_hull_max_eV_atom": 0.150,
            "E_hull_remains_hard": True,
            "minimum_site_fraction": args.minimum_site_fraction,
            "maximum_pair_shortfall_fraction": args.maximum_pair_shortfall_fraction,
            "site_fractions_are_ranking_metrics_not_hard-validity_redefinitions": True,
        },
        "summary": {
            "n": len(records),
            "relaxation_converged": sum(r["relaxation_converged"] for r in records),
            "exact_geometry_pass": sum(r["exact_geometry_pass"] for r in records),
            "near_coordination": sum(r["near_coordination"] for r in records),
            "structurally_rankable": sum(r["structurally_rankable"] for r in records),
            "final_candidates": 0,
        },
        "records": records,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run standalone direct-space IPOPT refinement for one input CIF."""
from __future__ import annotations

import argparse
import json

from nagen.ipopt_opt import DirectIPOPTConfig, optimize_cif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-cif", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--surrogate-checkpoint", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--max-wall-time", type=float, default=900.0)
    parser.add_argument("--print-level", type=int, default=5)
    parser.add_argument("--fractional-trust-radius", type=float, default=0.08)
    parser.add_argument("--lattice-relative-trust-radius", type=float, default=0.08)
    args = parser.parse_args()
    config = DirectIPOPTConfig(
        fractional_trust_radius=args.fractional_trust_radius,
        lattice_relative_trust_radius=args.lattice_relative_trust_radius,
        max_iterations=args.max_iterations,
        max_wall_time_seconds=args.max_wall_time,
        print_level=args.print_level,
    )
    report = optimize_cif(
        args.input, args.output_cif, args.report,
        surrogate_checkpoint=args.surrogate_checkpoint,
        uma_checkpoint=args.uma_checkpoint,
        device=args.device, config=config,
    )
    print(json.dumps({
        key: report[key] for key in (
            "status", "success", "formula", "initial_metrics",
            "final_metrics", "maximum_constraint_violation",
            "violated_constraint_count", "all_exact_checks_feasible",
            "priority_topology_constraints_feasible",
            "geometry_safeguard_alpha", "mechanical_constraints_feasible",
            "all_constraints_feasible",
            "wall_time_seconds",
        )
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

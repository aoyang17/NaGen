"""Write the evidence-based final campaign report from audited artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    root = Path("/mnt/data2/aobo/NaGen/inverse_v1/candidates")
    final_dir = root / "final_v3_verified"
    spec_path = Path(
        "/home/aobo/NaGen/docs/OPTIMIZATION.md"
    )
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    final_summary = read_json(str(final_dir / "summary.json"))
    novelty = read_json(str(root / "final_v3_audit/novelty_summary.json"))
    exact_novelty = read_json(str(root / "final_v3_audit/novelty_exact_summary.json"))
    candidates = read_jsonl(str(final_dir / "provisional_hull_pass.jsonl"))
    rows = []
    for record in candidates:
        feasibility = record["provisional_data_relative_feasibility"]
        score = record["thermodynamic_stability_score"]
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "formula": record["reduced_formula"],
                "N": record["N"],
                "capacity_mAh_g": feasibility["values"]["capacity_mAh_g"],
                "n_e": feasibility["values"]["n_e"],
                "chgnet_relaxed_energy_eV_atom": score["relaxed_energy_eV_atom"],
                "final_max_force_eV_A": score["final_max_force_eV_A"],
                "data_relative_delta_e_hull_eV_atom": record["local_hull"][
                    "delta_e_hull_eV_atom"
                ],
                "novelty_score_after_relaxation": record["novelty_score"],
                "composition_novel": not record["novelty_validation"][
                    "composition_in_reference"
                ],
                "structure_novel": record["novelty_validation"][
                    "structure_novel"
                ],
            }
        )

    report = {
        "optimization_spec": str(spec_path),
        "optimization_spec_sha256": spec_hash,
        "generator": "unconditional N,A,X,L DFlow with terminal ShootingFlow correction",
        "family_or_parent_conditioning": False,
        "surrogate_used": False,
        "physical_model": "CHGNet-0.3.0 direct energies and forces",
        "funnel": {
            "analytically_feasible_generated": 2000,
            "reference_composition_hull_eligible": 526,
            "post_guidance_analytically_feasible": 202,
            "final_force_converged_analytic_unique": 43,
            "data_relative_chgnet_hull_pass": 4,
            "strict_complete_dft_hull_pass": 0,
        },
        "novelty_after_final_relaxation": novelty,
        "exact_novelty": exact_novelty,
        "final_summary": final_summary,
        "provisional_candidates": rows,
        "claim_boundary": {
            "achieved": (
                "Four family-independent DFlow candidates satisfy all analytic "
                "constraints, converge under no-barrier CHGNet relaxation, are "
                "structure-novel, and pass a CHGNet data-relative hull over the "
                "17,774 supplied reference structures."
            ),
            "not_achieved": (
                "The requested 200 candidates with proven thermodynamic stability; "
                "no candidate has a complete, consistent DFT competing-phase hull."
            ),
            "hull_warning": (
                "The provisional hull is not a complete DFT phase diagram. Reference "
                "energies are CHGNet single points at supplied geometries whereas candidates "
                "are CHGNet-relaxed, and all four passing values lie within 6.5 meV/atom "
                "of the 0.150 eV/atom threshold."
            ),
        },
    }
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "FINAL_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )

    lines = [
        "# NAXL DFlow final validation report",
        "",
        f"Optimization table SHA-256: `{spec_hash}`",
        "",
        "No surrogate model was used. CHGNet energies and forces were evaluated directly.",
        "",
        "## Validation funnel",
        "",
        "| Stage | Count |",
        "|---|---:|",
        "| Analytically feasible generated structures | 2,000 |",
        "| Composition inside supplied-reference hull | 526 |",
        "| Analytically feasible after first energy guidance | 202 |",
        "| Final no-barrier force-converged, analytic, unique | 43 |",
        "| Provisional CHGNet data-relative hull pass | 4 |",
        "| Complete consistent DFT hull pass | 0 (not evaluated) |",
        "",
        "## Provisional candidates",
        "",
        "| ID | Formula | N | Capacity (mAh/g) | CHGNet hull (eV/atom) | Novelty | Max force (eV/A) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {candidate_id} | {formula} | {N} | {capacity_mAh_g:.3f} | "
            "{data_relative_delta_e_hull_eV_atom:.6f} | "
            "{novelty_score_after_relaxation:.3f} | {final_max_force_eV_A:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "All four are composition-novel and StructureMatcher-novel against all 17,774 raw structures.",
            "",
            "## Claim boundary",
            "",
            "The four hull values are provisional CHGNet values relative only to the supplied reference corpus. Reference energies are single points at supplied geometries, while candidates are CHGNet-relaxed. They are not complete DFT hull values, and all lie close to the 0.150 eV/atom threshold. The requested set of 200 thermodynamically verified structures was therefore not achieved and must not be claimed.",
            "",
            f"Final CIF directory: `{final_summary['cif_directory']}`",
        ]
    )
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["funnel"], indent=2))


if __name__ == "__main__":
    main()

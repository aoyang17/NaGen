"""Diagnose which pre-release N,A,X,L topology features predict survival."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
from pymatgen.core import Structure

from .constraints import pbc_distance_matrix
from .spec import DEFAULT_SPEC


def read_many(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with open(path) as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def topology_features(record: dict) -> dict[str, float]:
    structure = Structure.from_dict(record["structure"])
    elements = [str(specie) for specie in structure.species]
    counts = Counter(elements)
    distances = pbc_distance_matrix(
        structure.frac_coords, structure.lattice.matrix
    )
    p_indices = [i for i, element in enumerate(elements) if element == "P"]
    tm_indices = [i for i, element in enumerate(elements) if element == "Fe"]
    o_indices = [i for i, element in enumerate(elements) if element == "O"]

    p_bonds: list[float] = []
    tm_bonds: list[float] = []
    p_site_max: list[float] = []
    tm_site_max: list[float] = []
    for center in p_indices:
        bonded = [
            index for index in o_indices
            if distances[center, index] <= DEFAULT_SPEC.p_o_bond_cutoff
        ]
        values = [float(distances[center, index]) for index in bonded]
        p_bonds.extend(values)
        if values:
            p_site_max.append(max(values))
    for center in tm_indices:
        bonded = [
            index for index in o_indices
            if distances[center, index] <= DEFAULT_SPEC.fe_o_bond_cutoff
        ]
        values = [float(distances[center, index]) for index in bonded]
        tm_bonds.extend(values)
        if values:
            tm_site_max.append(max(values))
    score = record.get("thermodynamic_stability_score") or {}
    return {
        "N": float(len(elements)),
        "n_P": float(counts["P"]),
        "n_Fe": float(counts["Fe"]),
        "n_O": float(counts["O"]),
        "p_bond_mean_A": mean(p_bonds) if p_bonds else float("nan"),
        "p_bond_std_A": float(np.std(p_bonds)) if p_bonds else float("nan"),
        "p_worst_site_cutoff_margin_A": (
            DEFAULT_SPEC.p_o_bond_cutoff - max(p_site_max)
            if p_site_max else float("nan")
        ),
        "tm_bond_mean_A": mean(tm_bonds) if tm_bonds else float("nan"),
        "tm_bond_std_A": float(np.std(tm_bonds)) if tm_bonds else float("nan"),
        "tm_worst_site_cutoff_margin_A": (
            DEFAULT_SPEC.fe_o_bond_cutoff - max(tm_site_max)
            if tm_site_max else float("nan")
        ),
        "pre_release_energy_eV_atom": float(
            score.get("final_physical_energy_eV_atom", float("nan"))
        ),
        "pre_release_physical_max_force_eV_A": float(
            score.get("final_physical_max_force_eV_A", float("nan"))
        ),
        "pre_release_biased_max_force_eV_A": float(
            score.get("final_biased_max_force_eV_A", float("nan"))
        ),
        "pre_release_barrier_eV_cell": float(
            score.get("constraint_barrier_energy_eV_cell", float("nan"))
        ),
    }


def auc(values: list[float], labels: list[bool]) -> float | None:
    pairs = [(value, label) for value, label in zip(values, labels) if np.isfinite(value)]
    positives = [value for value, label in pairs if label]
    negatives = [value for value, label in pairs if not label]
    if not positives or not negatives:
        return None
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", nargs="+", required=True)
    parser.add_argument("--final", nargs="+", required=True)
    parser.add_argument("--hull", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    pre_records = read_many(args.pre)
    final_by_id = {record["candidate_id"]: record for record in read_many(args.final)}
    hull_by_id = {record["candidate_id"]: record for record in read_many(args.hull)}
    rows: list[dict] = []
    for record in pre_records:
        candidate_id = record["candidate_id"]
        final = final_by_id.get(candidate_id, {})
        final_score = final.get("thermodynamic_stability_score") or {}
        row = {
            "candidate_id": candidate_id,
            "reduced_formula": record["reduced_formula"],
            **topology_features(record),
            "final_force_converged": bool(final_score.get("converged_at_fmax", False)),
            "final_analytic": bool(final.get("analytical_feasible_after_relaxation", False)),
        }
        row["final_survivor"] = row["final_force_converged"] and row["final_analytic"]
        hull_record = hull_by_id.get(candidate_id, {})
        row["provisional_hull_pass"] = bool(
            hull_record.get("local_hull", {}).get("provisional_pass", False)
        )
        rows.append(row)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")

    labels = [row["final_survivor"] for row in rows]
    feature_names = [
        key for key, value in rows[0].items()
        if isinstance(value, float) and key != "N"
    ] if rows else []
    comparisons = {}
    for feature in feature_names:
        passed = [row[feature] for row in rows if row["final_survivor"] and np.isfinite(row[feature])]
        failed = [row[feature] for row in rows if not row["final_survivor"] and np.isfinite(row[feature])]
        values = [row[feature] for row in rows]
        comparisons[feature] = {
            "survivor_median": median(passed) if passed else None,
            "failure_median": median(failed) if failed else None,
            "auc_higher_predicts_survival": auc(values, labels),
        }
    ranked = sorted(
        comparisons.items(),
        key=lambda item: abs((item[1]["auc_higher_predicts_survival"] or 0.5) - 0.5),
        reverse=True,
    )
    summary = {
        "records": len(rows),
        "final_survivors": sum(labels),
        "provisional_hull_pass": sum(row["provisional_hull_pass"] for row in rows),
        "top_univariate_separators": [
            {"feature": feature, **stats} for feature, stats in ranked[:10]
        ],
        "all_feature_comparisons": comparisons,
        "output": str(output),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["top_univariate_separators"], indent=2))


if __name__ == "__main__":
    main()

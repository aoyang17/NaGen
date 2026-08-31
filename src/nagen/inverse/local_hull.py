"""Data-relative CHGNet convex-hull scoring with explicit scope labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from scipy.optimize import linprog

from .spec import DEFAULT_SPEC


def composition_vector(structure: Structure) -> np.ndarray:
    composition = structure.composition
    return np.asarray(
        [composition[element] / composition.num_atoms for element in DEFAULT_SPEC.elements],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    # At fixed composition only the lowest-energy reference structure can
    # contribute to a convex hull.
    lowest_by_composition: dict[tuple[float, ...], dict] = {}
    reference_rows = 0
    reference_errors = 0
    for path in sorted(Path(args.reference_dir).glob("reference_energy_shard_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "chgnet_energy_eV_atom" not in record:
                    reference_errors += 1
                    continue
                reference_rows += 1
                vector = tuple(
                    round(record["atomic_fractions"].get(element, 0.0), 12)
                    for element in DEFAULT_SPEC.elements
                )
                previous = lowest_by_composition.get(vector)
                if (
                    previous is None
                    or record["chgnet_energy_eV_atom"]
                    < previous["chgnet_energy_eV_atom"]
                ):
                    lowest_by_composition[vector] = record
    if not lowest_by_composition:
        raise RuntimeError("no valid reference energies found")
    reference_vectors = np.asarray(list(lowest_by_composition), dtype=np.float64)
    reference_records = list(lowest_by_composition.values())
    reference_energies = np.asarray(
        [record["chgnet_energy_eV_atom"] for record in reference_records],
        dtype=np.float64,
    )

    with open(args.candidates) as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]
    evaluated = 0
    covered = 0
    composition_covered = 0
    provisional_pass = 0
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in candidates:
            structure = Structure.from_dict(record["structure"])
            target = composition_vector(structure)
            stability_score = record.get("thermodynamic_stability_score") or {}
            energy = stability_score.get("relaxed_energy_eV_atom")
            energy_field = "relaxed_energy_eV_atom"
            if energy is None:
                energy = stability_score.get("final_physical_energy_eV_atom")
                energy_field = "final_physical_energy_eV_atom"
            support = target > 1.0e-12
            eligible = (reference_vectors[:, ~support] <= 1.0e-12).all(axis=1)
            vectors = reference_vectors[eligible]
            energies = reference_energies[eligible]
            eligible_records = [
                reference_records[index]
                for index in np.flatnonzero(eligible).tolist()
            ]
            if len(vectors) == 0:
                result = None
            else:
                # Sum(w)=1 and elemental atomic fractions reproduce the target.
                a_eq = np.vstack((np.ones(len(vectors)), vectors.T))
                b_eq = np.concatenate(([1.0], target))
                result = linprog(
                    energies,
                    A_eq=a_eq,
                    b_eq=b_eq,
                    bounds=(0.0, None),
                    method="highs",
                )
            if result is None or not result.success:
                record["local_hull"] = {
                    "status": "outside_reference_composition_convex_hull",
                    "delta_e_hull_eV_atom": None,
                    "candidate_energy_eV_atom": (
                        None if energy is None else float(energy)
                    ),
                    "candidate_energy_field": (
                        None if energy is None else energy_field
                    ),
                    "eligible_reference_compositions": len(vectors),
                }
            else:
                composition_covered += 1
                if energy is None:
                    record["local_hull"] = {
                        "status": "candidate_energy_missing",
                        "composition_inside_reference_hull": True,
                        "delta_e_hull_eV_atom": None,
                        "reference_hull_energy_eV_atom": float(result.fun),
                        "eligible_reference_compositions": len(vectors),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue
                evaluated += 1
                covered += 1
                raw_delta = float(energy - result.fun)
                delta = max(0.0, raw_delta)
                passed = delta <= DEFAULT_SPEC.hull_max_eV_atom
                provisional_pass += int(passed)
                active = np.flatnonzero(result.x > 1.0e-7)
                decomposition = [
                    {
                        "material_id": eligible_records[index]["material_id"],
                        "reduced_formula": eligible_records[index][
                            "reduced_formula"
                        ],
                        "weight": float(result.x[index]),
                        "energy_eV_atom": float(energies[index]),
                    }
                    for index in active
                ]
                record["local_hull"] = {
                    "status": "evaluated",
                    "method": "CHGNet-0.3.0 data-relative convex hull",
                    "scope": "17,774 provided reference structures only; not a complete DFT phase diagram",
                    "candidate_energy_eV_atom": float(energy),
                    "candidate_energy_field": energy_field,
                    "reference_hull_energy_eV_atom": float(result.fun),
                    "raw_energy_minus_reference_hull_eV_atom": raw_delta,
                    "delta_e_hull_eV_atom": delta,
                    "threshold_eV_atom": DEFAULT_SPEC.hull_max_eV_atom,
                    "provisional_pass": passed,
                    "decomposition": decomposition,
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "reference_rows": reference_rows,
        "reference_errors": reference_errors,
        "unique_reference_compositions": len(lowest_by_composition),
        "candidate_count": len(candidates),
        "candidate_energy_evaluated": evaluated,
        "composition_inside_reference_hull": composition_covered,
        "composition_outside_reference_hull": len(candidates)
        - composition_covered,
        "inside_reference_composition_hull": covered,
        "outside_reference_composition_hull": evaluated - covered,
        "provisional_delta_e_hull_pass": provisional_pass,
        "scope_warning": (
            "This is a CHGNet data-relative hull over the provided structures, "
            "not a complete DFT competing-phase hull."
        ),
        "output": str(output),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

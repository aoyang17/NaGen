"""Finite-reference same-model convex envelope, not an asserted complete hull."""
from collections import Counter
import hashlib
import json
import numpy as np
from scipy.optimize import linprog


ELEMENTS = ("Na", "Fe", "P", "O")


def fractions(elements):
    counts = Counter(elements)
    if not counts or not set(counts).issubset(ELEMENTS):
        raise ValueError("hull supports nonempty Na/Fe/P/O compositions only")
    return np.array([counts[e] / len(elements) for e in ELEMENTS])


class ReferenceHull:
    """Each reference must have id, elements, energy_eV_atom and model hash.

    Convex coefficients sum to one by the composition constraints. Missing
    compositional coverage returns unknown, never fabricated elemental energies.
    A finite candidate-independent reference set is frozen before evaluation.
    """
    def __init__(self, records, model_hash):
        if not records:
            raise ValueError("empty hull reference")
        if any(r["model_hash"] != model_hash for r in records):
            raise ValueError("mixed model energy scales in hull reference")
        self.records = records
        self.matrix = np.stack([fractions(r["elements"]) for r in records], axis=1)
        self.energies = np.array([r["energy_eV_atom"] for r in records])
        if not np.isfinite(self.energies).all():
            raise ValueError("nonfinite hull reference energy")
        self.hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()

    def evaluate(self, elements, energy):
        if not np.isfinite(energy):
            raise ValueError("nonfinite candidate energy")
        solution = linprog(self.energies, A_eq=self.matrix, b_eq=fractions(elements),
                           bounds=(0, None), method="highs")
        result = {"reference_scope": "finite_reference_set_not_complete_phase_diagram",
                  "reference_sha256": self.hash, "reference_count": len(self.records)}
        if not solution.success:
            return {**result, "status": "composition_not_covered", "delta_e_reference_eV_atom": None,
                    "e_above_reference_hull_eV_atom": None}
        delta = float(energy - solution.fun)
        return {**result, "status": "evaluated", "reference_energy_eV_atom": float(solution.fun),
                "delta_e_reference_eV_atom": delta,
                "e_above_reference_hull_eV_atom": max(0., delta),
                "below_reference_hull": delta < -1e-6,
                "decomposition": [{"id": r["id"], "fraction": float(w)}
                                  for r, w in zip(self.records, solution.x) if w > 1e-8]}

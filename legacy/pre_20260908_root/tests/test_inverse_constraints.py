from __future__ import annotations

import unittest
import numpy as np

from nagen.inverse.constraints import composition_metrics, evaluate_feasibility, theoretical_capacity
from nagen.inverse.spec import DEFAULT_SPEC


def formula(scale: int = 1) -> list[str]:
    return (["Na", "Fe", "P"] + ["O"] * 4) * scale


class ConstraintTests(unittest.TestCase):
    def test_spec_is_exactly_four_elements_and_locked_cutoffs(self) -> None:
        self.assertEqual(DEFAULT_SPEC.elements, ("Na", "Fe", "P", "O"))
        self.assertEqual(DEFAULT_SPEC.p_o_bond_cutoff, 2.00)
        self.assertEqual(DEFAULT_SPEC.fe_o_bond_cutoff, 2.50)

    def test_composition_and_fe_valence_boundaries(self) -> None:
        metrics = composition_metrics(formula())
        self.assertTrue(metrics["domain_ok"])
        self.assertEqual(metrics["fe_valence"], 2.0)
        oxygen_boundary = composition_metrics(["Na"] * 2 + ["Fe"] * 2 + ["P"] * 2 + ["O"] * 7)
        self.assertEqual(oxygen_boundary["y"], 0.5)
        self.assertTrue(oxygen_boundary["domain_ok"])

    def test_composition_rejects_missing_element_and_y_above_boundary(self) -> None:
        self.assertFalse(composition_metrics(["Na", "P"] + ["O"] * 4)["domain_ok"])
        outside = composition_metrics(["Na", "Fe", "P"] + ["O"] * 3)
        self.assertEqual(outside["y"], 1.0)
        self.assertFalse(outside["domain_ok"])

    def test_capacity_is_cell_scale_invariant(self) -> None:
        q1, _, n1 = theoretical_capacity(formula())
        q2, _, n2 = theoretical_capacity(formula(4))
        self.assertAlmostEqual(q1, q2, places=10)
        self.assertAlmostEqual(n2, 4 * n1)
        self.assertGreaterEqual(q1, 130.0)

    def test_missing_hull_never_passes_final_feasibility(self) -> None:
        elements = formula(4)
        lattice = np.eye(3) * (15.0 * len(elements)) ** (1 / 3)
        frac = np.linspace(0.01, 0.99, len(elements) * 3).reshape(-1, 3)
        result = evaluate_feasibility(elements, frac, lattice)
        self.assertFalse(result["checks"]["hull_threshold"])
        self.assertFalse(result["feasible"])


if __name__ == "__main__":
    unittest.main()

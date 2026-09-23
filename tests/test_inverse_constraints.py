from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from nagen.inverse.constraints import composition_metrics, evaluate_feasibility, theoretical_capacity
from nagen.selection.pipeline import Crystal, Conditioning, hard_gates
from nagen.inverse.spec import (
    DEFAULT_SPEC,
    OptimizationSpec,
    parse_fe_coordination_options,
)


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

    def test_fe_coordination_options_accept_any_nonempty_subset(self) -> None:
        self.assertEqual(parse_fe_coordination_options("4"), (4,))
        self.assertEqual(parse_fe_coordination_options("6,4"), (4, 6))
        self.assertEqual(parse_fe_coordination_options("4,5,6"), (4, 5, 6))
        self.assertEqual(
            OptimizationSpec(fe_coordination_options=(4, 6)).fe_coordination_options,
            (4, 6),
        )
        for invalid in ("", "3", "4,x"):
            with self.assertRaises(ValueError):
                parse_fe_coordination_options(invalid)

    def test_hard_gates_enforce_custom_fe_coordination_set(self) -> None:
        elements = ["Na", "Fe", "P", "O", "O", "O", "O"]
        crystal = Crystal(
            "custom-cn",
            elements,
            np.arange(len(elements) * 3).reshape(-1, 3) / (len(elements) * 3),
            np.eye(3) * 20.0,
        )
        profile = {
            "min_structures": 1,
            "training_ids": ["train-0"],
            "cutoffs": {"P": 2.0, "Fe": 2.5},
            "profiles": {
                "phosphate:P:4": {
                    "structures": 1, "bond_min": 1.4, "bond_max": 1.8,
                    "angle_reference": [109.5] * 6,
                },
                "phosphate:Fe:6": {
                    "structures": 1, "bond_min": 1.6, "bond_max": 2.4,
                    "angle_reference": [90.0] * 15,
                },
            },
        }
        rows = [
            {"site": 2, "element": "P", "cn": 4, "inside": True,
             "oxygen_indices": [3, 4, 5, 6], "bond_min": 1.5, "bond_max": 1.6,
             "off_center": 0.0, "bond_distortion": 0.0,
             "angles": [109.5] * 6},
            {"site": 1, "element": "Fe", "cn": 6, "inside": True,
             "oxygen_indices": [3, 4, 5, 6], "bond_min": 1.8, "bond_max": 2.2,
             "off_center": 0.0, "bond_distortion": 0.0,
             "angles": [90.0] * 15},
        ]
        condition = Conditioning({"Na": 1, "Fe": 1, "P": 1, "O": 4})
        forces = np.zeros((len(elements), 3))
        with patch("nagen.selection.pipeline.neighbors", return_value=[]), \
             patch("nagen.selection.pipeline.motifs", return_value=rows):
            allowed = hard_gates(
                crystal, condition, profile, forces=forces,
                motif_backend="native", fe_coordination_options=(4, 5, 6),
            )
            disallowed = hard_gates(
                crystal, condition, profile, forces=forces,
                motif_backend="native", fe_coordination_options=(4, 5),
            )
        self.assertTrue(allowed["passed"])
        self.assertFalse(disallowed["passed"])
        self.assertFalse(disallowed["sites"][1]["allowed_coordination"])

    def test_missing_hull_never_passes_final_feasibility(self) -> None:
        elements = formula(4)
        lattice = np.eye(3) * (15.0 * len(elements)) ** (1 / 3)
        frac = np.linspace(0.01, 0.99, len(elements) * 3).reshape(-1, 3)
        result = evaluate_feasibility(elements, frac, lattice)
        self.assertFalse(result["checks"]["hull_threshold"])
        self.assertFalse(result["feasible"])


if __name__ == "__main__":
    unittest.main()

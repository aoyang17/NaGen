import unittest
from collections import Counter
import numpy as np

from nagen.selection.pipeline import (
    Crystal, Conditioning, neighbors, motifs, calibrate, hard_gates,
    robust_rank, diversity_select,
)


class SelectionTests(unittest.TestCase):
    def test_vectorized_neighbors_match_explicit_images(self):
        from itertools import product
        rng = np.random.default_rng(17)
        c = Crystal('skew', ['O']*5, rng.random((5, 3)),
                    np.array([[2., 0, 0], [3., 2., 0], [.2, .3, 2.]]))
        for i in range(5):
            expected = []
            for j in range(5):
                for shift in product(range(-4, 5), repeat=3):
                    if i == j and shift == (0, 0, 0):
                        continue
                    d = np.linalg.norm((c.frac[j]-c.frac[i]+shift) @ c.lattice)
                    if d <= 2.5+1e-10:
                        expected.append((j, round(float(d), 9)))
            actual = [(j, round(d, 9)) for j, _, d in neighbors(c, i, 2.5)]
            self.assertEqual(sorted(actual), sorted(expected))

    def test_periodic_self_images_and_skew(self):
        c = Crystal("tiny", ["O"], np.zeros((1, 3)), np.eye(3))
        self.assertEqual(len(neighbors(c, 0, 1.01)), 6)
        c.lattice = np.array([[1., 0, 0], [4., 1, 0], [0, 0, 1]])
        self.assertEqual(len(neighbors(c, 0, 1.01)), 6)

    def test_po4_inside_and_outside(self):
        vectors = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]) * .9
        c = Crystal("tetra", ["P"] + ["O"] * 4,
                    np.vstack([np.zeros(3), vectors]) / 10, np.eye(3) * 10)
        row = motifs(c)[0]
        self.assertTrue(row["inside"])
        self.assertEqual(row["cn"], 4)
        self.assertAlmostEqual(row["off_center"], 0)
        c.frac[0] += [.12, 0, 0]
        self.assertFalse(motifs(c)[0]["inside"])

    def test_condition_and_missing_measurements_fail_closed(self):
        c = Crystal("c", ["Na", "Fe", "P"] + ["O"] * 4,
                    np.arange(21).reshape(7,3) / 23, np.eye(3) * 10)
        condition = Conditioning(dict(Counter(c.elements)))
        result = hard_gates(c, condition, calibrate([c], 1))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["final_force"])
        self.assertFalse(result["checks"]["external_motif"])
        self.assertTrue(result["checks"]["charge"])
        result = hard_gates(c, condition, calibrate([c], 1), forces=np.full((7,3), .04))
        self.assertFalse(result["checks"]["final_force"])  # norm, not component maximum
        with self.assertRaises(ValueError):
            Conditioning({"Na": 1.5, "Fe": 1, "P": 1, "O": 4})

    def test_bad_metric_cannot_be_hidden_by_good_energy(self):
        rows = [dict(id=str(i), passed=True, group="same", metrics={"energy": e, "local": q})
                for i, (e,q) in enumerate([(-1000,10),(-2,2),(-1,1)])]
        rows.append(dict(id="failed", passed=False, group="same", metrics={"energy": -1e9, "local": 0}))
        ranked = robust_rank(rows, ["energy", "local"])["same"]
        self.assertEqual(ranked[0]["id"], "1")
        self.assertEqual(len(ranked), 3)
        rows[0]["metrics"]["energy"] = float("nan")
        with self.assertRaises(ValueError):
            robust_rank(rows, ["energy", "local"])

    def test_diversity_respects_quality_and_training_duplicates(self):
        rows = [{"id": str(i)} for i in range(4)]
        x = np.array([0, .01, .5, 100])
        distances = np.abs(x[:,None] - x)
        chosen = diversity_select(rows, distances, [.1,.1,.5,100], 3, 3, .02)
        self.assertEqual([r["id"] for r in chosen], ["0", "2"])
        chosen = diversity_select(rows, distances, [0,.1,.5,100], 3, 3, .02)
        self.assertEqual([r["id"] for r in chosen], ["1", "2"])

    def test_composition_supercell_invariant(self):
        c = Crystal("x", ["Na", "Fe", "P"] + ["O"] * 4, None, None)
        d = Crystal("y", c.elements * 2, None, None)
        self.assertEqual(c.composition, d.composition)

    def test_energy_only_source_gradient(self):
        import torch
        from nagen.selection.optimization import optimize_source
        snapshots = optimize_source([torch.tensor([2.])],
            lambda x: (x, torch.eye(3)), lambda x, cell: (x*x).sum(), steps=5, lr=.1)
        self.assertEqual(len(snapshots), 6)
        self.assertLess(snapshots[-1]["energy_eV_atom"], snapshots[0]["energy_eV_atom"])


if __name__ == "__main__":
    unittest.main()

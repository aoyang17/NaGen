import unittest
import numpy as np
from nagen.selection.pipeline import Crystal
from nagen.selection.hull import ReferenceHull
from nagen.selection.diversity import descriptor


class HullTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"id": e, "elements": [e], "energy_eV_atom": 0., "model_hash": "x"}
                        for e in ("Na", "Fe", "P", "O")]
        self.records.append({"id": "oxide", "elements": ["Fe", "O"],
                             "energy_eV_atom": -2., "model_hash": "x"})

    def test_decomposition_and_below_reference(self):
        hull = ReferenceHull(self.records, "x")
        r = hull.evaluate(["Na", "Fe", "O"], -1.)
        self.assertAlmostEqual(r["reference_energy_eV_atom"], -4/3)
        self.assertAlmostEqual(r["delta_e_reference_eV_atom"], 1/3)
        r = hull.evaluate(["Fe", "O"], -3.)
        self.assertEqual(r["delta_e_reference_eV_atom"], -1)
        self.assertEqual(r["e_above_reference_hull_eV_atom"], 0)
        self.assertTrue(r["below_reference_hull"])

    def test_missing_composition_and_mixed_models(self):
        hull = ReferenceHull(self.records[-1:], "x")
        self.assertEqual(hull.evaluate(["Na"], 0)["status"], "composition_not_covered")
        with self.assertRaises(ValueError):
            ReferenceHull(self.records, "wrong")


class FingerprintTests(unittest.TestCase):
    def test_invariances(self):
        c = Crystal("a", ["Na", "Fe", "P", "O"],
                    np.array([[.1,.2,.3],[.7,.2,.4],[.4,.6,.8],[.7,.8,.2]]), np.eye(3)*6.)
        original = descriptor(c)
        perm = [2,0,3,1]
        q, _ = np.linalg.qr(np.array([[1.,2,3],[2,0,1],[0,1,2]]))
        changed = Crystal("b", [c.elements[i] for i in perm], (c.frac[perm]+.43)%1, c.lattice @ q)
        np.testing.assert_allclose(descriptor(changed), original, atol=1e-12)
        supercell = Crystal("super", c.elements*2,
            np.vstack([c.frac/[2,1,1],(c.frac+[1,0,0])/[2,1,1]]), np.diag([2,1,1])@c.lattice)
        np.testing.assert_allclose(descriptor(supercell), original, atol=1e-12)
        basis = np.array([[1,2,0],[0,1,0],[0,0,1]])
        changed = Crystal("basis", c.elements, (c.frac@np.linalg.inv(basis))%1, basis@c.lattice)
        np.testing.assert_allclose(descriptor(changed), original, atol=1e-12)


class FixedConditionTests(unittest.TestCase):
    def test_midpoint_keeps_atom_channel_and_source_gradients(self):
        import torch
        from nagen.inverse.model import CrystalState
        from nagen.selection.generate import fixed_rollout
        atom = torch.tensor([[[1., 0.], [0., 1.]]])
        seen = []

        class Field:
            lattice_mean = torch.zeros(6)
            lattice_std = torch.ones(6)

            def __call__(self, a, x, h, t, mask):
                seen.append(a.detach().clone())
                return CrystalState(torch.ones_like(a)*100, x*.01, h*.01)

        frac = torch.full((1,2,3), .2, requires_grad=True)
        lattice = torch.zeros(1,6,requires_grad=True)
        x, h = fixed_rollout(Field(), atom, frac, lattice, torch.ones(1,2,dtype=torch.bool), 3)
        gradients = torch.autograd.grad(x.sum()+h.sum(), (frac,lattice))
        self.assertTrue(all(torch.equal(a,atom) for a in seen))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))


if __name__ == "__main__":
    unittest.main()

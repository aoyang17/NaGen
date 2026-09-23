import unittest
from types import SimpleNamespace
import torch

from nagen.inverse.guidance import _coordination_assignment, terminal_constraint_terms
from nagen.inverse.model import (
    CrystalState, CrystalVectorField, FlowConfig, flow_matching_loss,
)
from nagen.inverse.spec import DEFAULT_SPEC


class TerminalGuidanceTests(unittest.TestCase):
    def test_fe_only_terms_have_finite_gradients(self):
        torch.manual_seed(7)
        model = SimpleNamespace(lattice_mean=torch.zeros(6), lattice_std=torch.ones(6))
        offsets = {element: index for index, element in enumerate(DEFAULT_SPEC.elements)}
        elements = ["P", "Fe"] + ["O"] * 12 + ["Na"] * 9
        atom_indices = torch.tensor([[offsets[element] for element in elements]])
        atom = torch.nn.functional.one_hot(atom_indices, num_classes=4).float()
        frac = torch.rand(1, len(elements), 3, requires_grad=True)
        lattice = torch.zeros(1, 6, requires_grad=True)
        mask = torch.ones(1, len(elements), dtype=torch.bool)
        terms = terminal_constraint_terms(model, CrystalState(atom, frac, lattice), mask)
        loss = sum(value for name, value in terms.items() if name != "minimum_distance_A")
        loss.backward()
        self.assertEqual(set(terms), {
            "distance", "p_coordination", "fe_coordination",
            "p_geometry", "fe_geometry", "minimum_distance_A",
            "poly_center", "poly_face", "poly_coplanar",
        })
        self.assertTrue(torch.isfinite(frac.grad).all())
        self.assertTrue(torch.isfinite(lattice.grad).all())
        self.assertGreater(float(frac.grad.norm()), 0.0)

    def test_population_weighted_fe_coordination_has_finite_gradients(self):
        torch.manual_seed(8)
        model = SimpleNamespace(lattice_mean=torch.zeros(6), lattice_std=torch.ones(6))
        offsets = {element: index for index, element in enumerate(DEFAULT_SPEC.elements)}
        elements = ["P", "Fe"] + ["O"] * 12 + ["Na"] * 9
        atom_indices = torch.tensor([[offsets[element] for element in elements]])
        atom = torch.nn.functional.one_hot(atom_indices, num_classes=4).float()
        frac = torch.rand(1, len(elements), 3, requires_grad=True)
        lattice = torch.zeros(1, 6, requires_grad=True)
        mask = torch.ones(1, len(elements), dtype=torch.bool)
        terms = terminal_constraint_terms(
            model,
            CrystalState(atom, frac, lattice),
            mask,
            fe_coordination_prior=(0.084, 0.1025, 0.8135),
        )
        terms["fe_coordination"].backward()
        self.assertTrue(torch.isfinite(frac.grad).all())
        self.assertTrue(torch.isfinite(lattice.grad).all())

    def test_assignment_honors_custom_fe_coordination_subsets(self):
        offsets = {element: index for index, element in enumerate(DEFAULT_SPEC.elements)}
        types = torch.tensor([[offsets["Fe"]] + [offsets["O"]] * 6])
        mask = torch.ones(1, 7, dtype=torch.bool)
        distances = torch.ones(1, 7, 7) * 2.2
        distances[0, 0, 1:5] = 2.0
        distances[0, 0, 5] = 2.49
        distances[0, 0, 6] = 10.0
        for options, expected in (
            ((4,), 4),
            ((5,), 5),
            ((4, 6), 4),
            ((5, 6), 5),
        ):
            assignment = _coordination_assignment(
                types,
                mask,
                distances,
                fe_coordination_options=options,
            )
            self.assertEqual(int(assignment[0, 0].sum()), expected)

    def test_geometry_validity_auxiliary_has_finite_gradient(self):
        torch.manual_seed(11)
        model = CrystalVectorField(
            FlowConfig(hidden_dim=16, layers=1, heads=4, radial_basis=4, time_dim=8),
            torch.zeros(6), torch.ones(6),
        )
        elements = (["Na", "Fe", "P"] + ["O"] * 4) * 4
        types = torch.tensor([[
            DEFAULT_SPEC.element_to_index[element] for element in elements
        ]])
        frac = torch.rand(1, len(elements), 3)
        lattice = torch.zeros(1, 6)
        mask = torch.ones(1, len(elements), dtype=torch.bool)
        loss, metrics = flow_matching_loss(
            model, types, frac, lattice, mask,
            geometry_only=True,
            geometry_validity_weight=1.0,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(metrics["geometry_validity_loss"]))
        self.assertGreater(float(metrics["geometry_validity_loss"]), 0.0)
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()

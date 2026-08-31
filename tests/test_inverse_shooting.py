from __future__ import annotations

import unittest
import torch

from nagen.inverse.model import CrystalState, CrystalVectorField, FlowConfig
from nagen.inverse.multiobjective_shooting import (
    CrystalDesignProblem, ShootingConfig, ShootingFlowRunner, closed_form_mgda,
)


class MGDATests(unittest.TestCase):
    def test_parallel_opposite_and_zero_gradients(self) -> None:
        alpha, combined = closed_form_mgda(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))
        self.assertEqual(alpha, 1.0)
        torch.testing.assert_close(combined, torch.tensor([1.0, 0.0]))
        alpha, combined = closed_form_mgda(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
        self.assertAlmostEqual(alpha, 0.5)
        torch.testing.assert_close(combined, torch.zeros(2))
        alpha, combined = closed_form_mgda(torch.zeros(2), torch.zeros(2))
        self.assertEqual(alpha, 0.5)
        torch.testing.assert_close(combined, torch.zeros(2))

    def test_source_only_update_freezes_model_and_reintegrates(self) -> None:
        torch.manual_seed(4)
        model = CrystalVectorField(
            FlowConfig(hidden_dim=16, layers=1, heads=4, radial_basis=4, time_dim=8),
            torch.zeros(6), torch.ones(6),
        ).eval()
        elements = [0, 1, 2] + [3] * 4
        indices = torch.tensor([elements * 4])
        atom = torch.nn.functional.one_hot(indices, num_classes=4).float() * 2.0
        mask = torch.ones(1, atom.shape[1], dtype=torch.bool)
        source = CrystalState(atom, torch.rand(1, atom.shape[1], 3), torch.randn(1, 6))

        def energy(frac, lattice):
            return frac.square().mean((1, 2)) + 0.01 * lattice.square().mean((1, 2))

        problem = CrystalDesignProblem(model, mask, energy, -0.2)
        result = ShootingFlowRunner(model, problem).run(
            atom.shape[1], atom, source,
            ShootingConfig(guidance_steps=1, ode_steps=1, learning_rate=0.001),
        )
        self.assertEqual(result.status, "complete")
        self.assertTrue(torch.equal(result.z0_star.atom, source.atom))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(len(result.history), 1)

    def test_restoration_precedes_objectives(self) -> None:
        torch.manual_seed(5)
        model = CrystalVectorField(
            FlowConfig(hidden_dim=16, layers=1, heads=4, radial_basis=4, time_dim=8),
            torch.zeros(6), torch.ones(6),
        ).eval()
        indices = torch.tensor([[0, 1, 2, 3, 3, 3, 3] * 4])
        atom = torch.nn.functional.one_hot(indices, num_classes=4).float() * 2.0
        mask = torch.ones(1, atom.shape[1], dtype=torch.bool)
        source = CrystalState(atom, torch.rand(1, atom.shape[1], 3), torch.randn(1, 6))
        calls = 0

        def energy(frac, lattice):
            nonlocal calls
            calls += 1
            return frac.square().mean((1, 2)) + 0.01 * lattice.square().mean((1, 2))

        result = ShootingFlowRunner(
            model, CrystalDesignProblem(model, mask, energy, -0.2)
        ).run(
            atom.shape[1], atom, source,
            ShootingConfig(
                restoration_steps=2,
                guidance_steps=1,
                ode_steps=1,
                learning_rate=0.001,
                use_novelty=False,
            ),
        )
        self.assertEqual([item["phase"] for item in result.history], [
            "restoration", "restoration", "multiobjective",
        ])
        self.assertEqual(calls, 1)

    def test_constraint_margin_is_validated_and_forwarded(self) -> None:
        model = CrystalVectorField(
            FlowConfig(hidden_dim=16, layers=1, heads=4, radial_basis=4, time_dim=8),
            torch.zeros(6), torch.ones(6),
        ).eval()
        indices = torch.tensor([[0, 1, 2, 3, 3, 3, 3]])
        atom = torch.nn.functional.one_hot(indices, num_classes=4).float()
        mask = torch.ones_like(indices, dtype=torch.bool)
        terminal = CrystalState(atom, torch.rand(1, 7, 3), torch.zeros(1, 6))
        with self.assertRaises(ValueError):
            CrystalDesignProblem(model, mask, lambda x, cell: x.sum((1, 2)), 0.0,
                                 constraint_margin_A=-0.01)
        default = CrystalDesignProblem(
            model, mask, lambda x, cell: x.sum((1, 2)), 0.0,
            constraint_margin_A=0.0,
        ).constraint_terms(terminal)["distance"]
        buffered = CrystalDesignProblem(
            model, mask, lambda x, cell: x.sum((1, 2)), 0.0,
            constraint_margin_A=0.10,
        ).constraint_terms(terminal)["distance"]
        self.assertGreaterEqual(float(buffered), float(default))


if __name__ == "__main__":
    unittest.main()

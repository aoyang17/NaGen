from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from nagen.inverse.calibrated_mp_hull import (
    CalibratedMPHullCache,
    build_calibrated_cache,
    candidate_correction_eV_atom,
    candidate_parameters,
)
from nagen.inverse.joint_dflow import (
    JointCrystalProblem,
    JointDFlowConfig,
    JointDFlowSolver,
)
from nagen.inverse.differentiable_mp_hull import DifferentiableMP2020Hull
from nagen.inverse.model import CrystalState, CrystalVectorField, FlowConfig
from nagen.inverse.novelty import crystal_descriptor
from nagen.inverse.relaxation_surrogate import (
    RelaxationMember,
    RelaxationSurrogateEnsemble,
    soft_crystal_descriptor,
)
from nagen.inverse.relax_aware_dflow import (
    RelaxAwareConfig,
    RelaxAwareIPOPTSolver,
    RelaxAwareProblem,
)


class CalibratedMPHullTests(unittest.TestCase):
    def test_differentiable_hull_planes_match_calibrated_cache(self) -> None:
        phase_cache = Path(
            "/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/cache/"
            "mp_phase_entries.json"
        )
        if not phase_cache.exists():
            self.skipTest("project phase cache is unavailable")
        hull = DifferentiableMP2020Hull.from_phase_cache(phase_cache)
        probabilities = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.],
                                       [0., 0., 1., 0.], [0., 0., 0., 1.],
                                       [0., 0., 0., 1.], [0., 0., 0., 1.],
                                       [0., 0., 0., 1.]]], requires_grad=True)
        mask = torch.ones(1, 7, dtype=torch.bool)
        energy = torch.tensor([-6.0], requires_grad=True)
        raw = hull.raw_e_hull(energy, probabilities, mask)
        cache_payload = build_calibrated_cache(
            phase_cache, [{"Na": 1, "Fe": 1, "P": 1, "O": 4}]
        )
        cache_path = Path(tempfile.mkdtemp()) / "cache.json"
        cache_path.write_text(json.dumps(cache_payload))
        expected = CalibratedMPHullCache(cache_path).raw_e_hull("NaFePO4", -6.0)
        self.assertAlmostEqual(float(raw), expected, places=5)
        gradients = torch.autograd.grad(raw.sum(), (energy, probabilities))
        self.assertTrue(all(torch.isfinite(item).all() for item in gradients))

    def test_candidate_metadata_matches_reference_protocol(self) -> None:
        parameters = candidate_parameters({"Na": 1, "Fe": 1, "P": 1, "O": 4})
        self.assertEqual(parameters["run_type"], "GGA+U")
        self.assertEqual(parameters["hubbards"], {"Fe": 5.3})
        self.assertEqual(
            parameters["potcar_symbols"],
            ["PAW_PBE Na_pv", "PAW_PBE Fe_pv", "PAW_PBE P", "PAW_PBE O"],
        )
        self.assertLess(candidate_correction_eV_atom("NaFePO4"), 0.0)

    def test_phase_entries_are_not_corrected_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase = root / "phase.json"
            output = root / "cache.json"
            entries = [
                {"entry_id": "Na", "composition": {"Na": 1}, "energy_eV": 0.0},
                {"entry_id": "Fe", "composition": {"Fe": 1}, "energy_eV": 0.0},
                {"entry_id": "P", "composition": {"P": 1}, "energy_eV": 0.0},
                {"entry_id": "O", "composition": {"O": 2}, "energy_eV": 0.0},
                {
                    "entry_id": "compound", "composition": {
                        "Na": 1, "Fe": 1, "P": 1, "O": 4,
                    }, "energy_eV": -14.0, "correction_eV": -7.0,
                },
            ]
            phase.write_text(json.dumps({
                "compatibility": "MaterialsProject2020Compatibility",
                "entries": entries,
            }))
            payload = build_calibrated_cache(
                phase, [{"Na": 1, "Fe": 1, "P": 1, "O": 4}]
            )
            output.write_text(json.dumps(payload))
            cache = CalibratedMPHullCache(output)
            reference, _ = cache.constants("NaFePO4")
            self.assertAlmostEqual(reference, -2.0)
            self.assertEqual(
                payload["mp_entries_energy_semantics"],
                "already_compatible_energy_eV_no_recorrection",
            )


class JointDFlowTests(unittest.TestCase):
    def test_hard_hull_forward_has_analytic_soft_composition_gradient(self) -> None:
        model = CrystalVectorField(
            FlowConfig(
                hidden_dim=16, layers=1, heads=4,
                radial_basis=4, time_dim=8,
            ),
            torch.zeros(6), torch.ones(6),
        ).eval()
        mask = torch.ones(1, 4, dtype=torch.bool)
        atom = torch.tensor([[
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ]], requires_grad=True)
        terminal = CrystalState(atom, torch.rand(1, 4, 3), torch.zeros(1, 6))
        hull = DifferentiableMP2020Hull(torch.tensor([
            [0.0, 0.0, 0.0, -1.0],
        ]))

        def factory(numbers):
            def energy(frac, lattice):
                return 0.0 * (frac.sum((1, 2)) + lattice.sum((1, 2)))
            return energy

        problem = JointCrystalProblem(
            model, mask, factory(torch.tensor([11, 26, 15, 8])),
            0.0, 0.0, None,
            energy_factory=factory,
            differentiable_hull=hull,
            atomic_number_lookup=torch.tensor([11, 26, 15, 8]),
        )
        raw = problem.values(terminal)["raw_E_hull"]
        hard = torch.eye(4).reshape(1, 4, 4)
        expected = hull.raw_e_hull(torch.zeros(1), hard, mask).mean()
        torch.testing.assert_close(raw.detach(), expected.detach())
        gradient = torch.autograd.grad(raw, atom)[0]
        self.assertGreater(float(gradient.norm()), 0.0)

    def test_joint_problem_optimizes_atom_fraction_and_lattice_noise(self) -> None:
        torch.manual_seed(17)
        model = CrystalVectorField(
            FlowConfig(
                hidden_dim=16, layers=1, heads=4,
                radial_basis=4, time_dim=8,
            ),
            torch.zeros(6), torch.ones(6),
        ).eval()
        z0 = CrystalState(
            torch.randn(1, 28, 4), torch.rand(1, 28, 3), torch.randn(1, 6)
        )
        mask = torch.ones(1, 28, dtype=torch.bool)

        def factory(numbers):
            def energy(frac, lattice):
                return (
                    frac.square().mean((1, 2))
                    + 0.01 * lattice.square().mean((1, 2))
                    + 0.0 * numbers.float().sum()
                )
            return energy

        problem = JointCrystalProblem(
            model, mask, factory(torch.full((28,), 8)), 0.0, 0.0, None,
            energy_factory=factory,
            differentiable_hull=DifferentiableMP2020Hull(torch.tensor([
                [0.0, 0.0, 0.0, -1.0],
            ])),
            atomic_number_lookup=torch.tensor([11, 26, 15, 8]),
        )
        result = JointDFlowSolver(model, problem).solve(
            z0, JointDFlowConfig(steps=1, ode_steps=1, hull_threshold_eV_atom=10.0)
        )
        self.assertEqual(
            result.solver_stats["optimized_source_channels"],
            ["z_A", "z_X", "z_L"],
        )
        self.assertEqual(result.z0_star.atom.shape, z0.atom.shape)
        self.assertIn("composition", result.history[0]["constraints"])

    def test_soft_descriptor_matches_hard_descriptor_for_one_hot_types(self) -> None:
        types = torch.tensor([[1, 2, 3, 4]])
        probabilities = torch.nn.functional.one_hot(types - 1, num_classes=4).float()
        frac = torch.rand(1, 4, 3)
        lattice = torch.eye(3)[None] * 5.0
        mask = torch.ones(1, 4, dtype=torch.bool)
        expected = crystal_descriptor(types, frac, lattice, mask)
        actual = soft_crystal_descriptor(probabilities, frac, lattice, mask)
        torch.testing.assert_close(actual, expected)

    def test_relaxation_surrogate_propagates_to_soft_species_and_geometry(self) -> None:
        dimension = 331
        member = RelaxationMember(dimension, hidden=16)
        surrogate = RelaxationSurrogateEnsemble(
            [member], torch.zeros(dimension), torch.ones(dimension),
            torch.zeros(2), torch.ones(2),
        )
        probabilities = torch.softmax(torch.randn(1, 4, 4), dim=-1).requires_grad_(True)
        frac = torch.rand(1, 4, 3, requires_grad=True)
        lattice = (torch.eye(3)[None] * 5.0).requires_grad_(True)
        mask = torch.ones(1, 4, dtype=torch.bool)
        output = surrogate.forward_structure(probabilities, frac, lattice, mask)
        loss = output["final_energy_mean"].sum()
        gradients = torch.autograd.grad(loss, (probabilities, frac, lattice))
        self.assertTrue(all(torch.isfinite(item).all() for item in gradients))

    def test_relax_aware_ipopt_optimizes_atom_fraction_and_lattice_noise(self) -> None:
        torch.manual_seed(11)
        model = CrystalVectorField(
            FlowConfig(hidden_dim=16, layers=1, heads=4, radial_basis=4, time_dim=8),
            torch.zeros(6), torch.ones(6),
        ).eval()
        dimension = 331
        surrogate = RelaxationSurrogateEnsemble(
            [RelaxationMember(dimension, hidden=16)],
            torch.zeros(dimension), torch.ones(dimension),
            torch.zeros(2), torch.ones(2),
        ).eval()
        hull = DifferentiableMP2020Hull(torch.tensor([
            [0.0, 0.0, 0.0, -1.0],
        ]))
        mask = torch.ones(1, 8, dtype=torch.bool)
        problem = RelaxAwareProblem(model, mask, surrogate, hull, None)
        z0 = CrystalState(
            torch.randn(1, 8, 4), torch.rand(1, 8, 3), torch.randn(1, 6)
        )
        result = RelaxAwareIPOPTSolver(model, problem).solve(
            z0, RelaxAwareConfig(steps=1, ode_steps=1)
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.z0_star.atom.shape, z0.atom.shape)
        self.assertGreaterEqual(len(result.history), 1)
        self.assertIn("composition", result.history[0])
        self.assertIn("return_status", result.solver_stats)

    def test_one_joint_problem_updates_only_source_geometry(self) -> None:
        torch.manual_seed(7)
        model = CrystalVectorField(
            FlowConfig(
                hidden_dim=16, layers=1, heads=4,
                radial_basis=4, time_dim=8,
            ),
            torch.zeros(6), torch.ones(6),
        ).eval()
        indices = torch.tensor([[0] * 4 + [1] * 4 + [2] * 4 + [3] * 16])
        atom = torch.nn.functional.one_hot(indices, num_classes=4).float() * 2.0
        mask = torch.ones_like(indices, dtype=torch.bool)
        z0 = CrystalState(atom, torch.rand(1, 28, 3), torch.randn(1, 6))

        def energy(frac, lattice):
            return frac.square().mean((1, 2)) + 0.01 * lattice.square().mean((1, 2))

        problem = JointCrystalProblem(
            model, mask, energy,
            mp_hull_reference_eV_atom=-1.0,
            mp2020_candidate_correction_eV_atom=-0.2,
            novelty_index=None,
        )
        result = JointDFlowSolver(model, problem).solve(
            z0,
            JointDFlowConfig(
                steps=2, ode_steps=1, learning_rate=0.001,
                novelty_weight=0.01, hull_threshold_eV_atom=10.0,
            ),
        )
        self.assertEqual(result.status, "complete")
        self.assertTrue(torch.equal(result.z0_star.atom, z0.atom))
        self.assertGreaterEqual(len(result.history), 1)
        self.assertIn(
            result.solver_stats["return_status"],
            {"Solve_Succeeded", "Solved_To_Acceptable_Level", "Maximum_Iterations_Exceeded"},
        )
        for row in result.history:
            self.assertEqual(
                set(row["constraints"]),
                {"distance", "p_coordination", "fe_coordination", "volume"},
            )
            self.assertIn("E_hull", row)
            self.assertIn("hull_constraint_value_eV_atom", row)
            self.assertIn("hull_constraint_satisfied", row)
            self.assertIn("hull_constraint_gradient_norm", row)
            self.assertIn("novelty", row)
            self.assertIn("evaluation", row)
            self.assertNotIn("phase", row)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(result.solver_stats["hull_constraint_satisfied"])

    def test_infeasible_hull_is_not_reported_complete(self) -> None:
        torch.manual_seed(9)
        model = CrystalVectorField(
            FlowConfig(
                hidden_dim=16, layers=1, heads=4,
                radial_basis=4, time_dim=8,
            ),
            torch.zeros(6), torch.ones(6),
        ).eval()
        indices = torch.tensor([[0] * 2 + [1] * 2 + [2] * 2 + [3] * 8])
        atom = torch.nn.functional.one_hot(indices, num_classes=4).float() * 2.0
        mask = torch.ones_like(indices, dtype=torch.bool)
        z0 = CrystalState(atom, torch.rand(1, 14, 3), torch.randn(1, 6))

        def irreducibly_high_energy(frac, lattice):
            # Keep a valid autograd path while making the calibrated hull
            # constraint impossible for this synthetic problem.
            return 1.0 + 0.0 * (
                frac.sum((1, 2)) + lattice.sum((1, 2))
            )

        problem = JointCrystalProblem(
            model, mask, irreducibly_high_energy,
            mp_hull_reference_eV_atom=0.0,
            mp2020_candidate_correction_eV_atom=0.0,
            novelty_index=None,
        )
        result = JointDFlowSolver(model, problem).solve(
            z0,
            JointDFlowConfig(
                steps=1, ode_steps=1, hull_threshold_eV_atom=0.150,
            ),
        )
        self.assertEqual(result.status, "hull_constraint_violated")
        self.assertFalse(result.solver_stats["hull_constraint_satisfied"])
        self.assertGreater(
            result.solver_stats["selected_hull_constraint_eV_atom"], 0.150
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pymatgen.core import Lattice, Structure

from nagen.inverse.relax_uma import RelaxationConfig, relax_structure, unique_structure_indices
from nagen.inverse.joint_dflow_relax_campaign import final_acceptance


class FakeOptimizer:
    result = True

    def __init__(self, target, **kwargs):
        self.target = target

    def run(self, fmax, steps):
        return self.result


class FalseOptimizer(FakeOptimizer):
    result = False


class RelaxationTests(unittest.TestCase):
    def atoms(self, force: float = 0.0):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
        calculator = SinglePointCalculator(
            atoms, energy=0.0, forces=np.array([[force, 0.0, 0.0]]), stress=np.zeros(6)
        )
        return atoms, calculator

    def test_fire_max_steps_can_be_recovered_by_converged_bfgs(self) -> None:
        atoms, calculator = self.atoms(0.0)
        with tempfile.TemporaryDirectory() as directory:
            result = relax_structure(
                atoms, calculator, directory, RelaxationConfig(fire_steps=1, bfgs_steps=1),
                fire_factory=FalseOptimizer, bfgs_factory=FakeOptimizer,
            )
        self.assertTrue(result.converged)
        self.assertFalse(result.fire_converged)
        self.assertEqual(result.status, "converged_after_fire_max_steps")

    def test_both_optimizers_converged_is_success(self) -> None:
        atoms, calculator = self.atoms(0.0)
        with tempfile.TemporaryDirectory() as directory:
            result = relax_structure(
                atoms, calculator, directory, RelaxationConfig(fire_steps=1, bfgs_steps=1),
                fire_factory=FakeOptimizer, bfgs_factory=FakeOptimizer,
            )
        self.assertTrue(result.converged)

    def test_bfgs_max_steps_is_final_failure(self) -> None:
        atoms, calculator = self.atoms(0.0)
        with tempfile.TemporaryDirectory() as directory:
            result = relax_structure(
                atoms, calculator, directory, RelaxationConfig(fire_steps=1, bfgs_steps=1),
                fire_factory=FakeOptimizer, bfgs_factory=FalseOptimizer,
            )
        self.assertFalse(result.converged)
        self.assertEqual(result.status, "bfgs_not_converged")

    def test_structure_matcher_removes_reference_and_candidate_duplicates(self) -> None:
        lattice = Lattice.cubic(5)
        reference = Structure(lattice, ["Na"], [[0, 0, 0]])
        duplicate = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        novel = Structure(lattice, ["Fe"], [[0, 0, 0]])
        indices = unique_structure_indices([duplicate, novel, novel.copy()], [reference])
        self.assertEqual(indices, [1])

    def test_final_acceptance_requires_all_four_gates(self) -> None:
        feasible = {
            "feasible": True,
            "checks": {"hull_threshold": True},
        }
        accepted = final_acceptance(True, -6.0, -6.1, feasible, 1.0e-6)
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["status"], "accepted")

        increased = final_acceptance(True, -6.0, -5.9, feasible, 1.0e-6)
        self.assertFalse(increased["accepted"])
        self.assertEqual(increased["status"], "energy_increased")

        hull_failed = {
            "feasible": False,
            "checks": {"hull_threshold": False},
        }
        rejected = final_acceptance(True, -6.0, -6.1, hull_failed, 1.0e-6)
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["status"], "constraints_violated")


if __name__ == "__main__":
    unittest.main()

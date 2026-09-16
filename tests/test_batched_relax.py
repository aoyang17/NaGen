from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE
import numpy as np

from nagen.selection.batched_relax import fire_batch


class Harmonic(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        p = self.atoms.positions
        self.results = {"energy": .5 * float(np.sum(p * p)), "forces": -p,
                        "stress": np.zeros(6)}


def test_batched_fire_matches_independent_ase_trajectories():
    original = [Atoms("H", positions=[[x, 0, 0]], cell=np.eye(3) * 20, pbc=True) for x in (0., .2, 2., 20.)]
    sequential = []
    for item in original:
        atoms = item.copy()
        atoms.calc = Harmonic()
        opt = FIRE(atoms, logfile=None, maxstep=.1)
        opt.run(fmax=.03, steps=60)
        sequential.append((atoms.positions.copy(), opt.nsteps))
    batch_sizes = []

    def evaluate(atoms):
        batch_sizes.append(len(atoms))
        rows = []
        for a in atoms:
            force = -a.positions.copy()
            rows.append({"energy_eV_atom": .5 * float(np.sum(a.positions ** 2)) / len(a),
                         "forces_eV_A": force, "stress_eV_A3": np.zeros((3, 3)),
                         "force_max_eV_A": float(np.linalg.norm(force, axis=1).max())})
        return rows

    batch = [a.copy() for a in original]
    result = fire_batch(evaluate, batch, fmax=.03, steps=60)
    for a, row, (positions, steps) in zip(batch, result, sequential):
        np.testing.assert_allclose(a.positions, positions, rtol=0, atol=1e-14)
        assert row["steps"] == steps
    assert result[0]["steps"] == 0
    assert result[-1]["steps"] == 60 and not result[-1]["converged"]
    assert batch_sizes[0] == 4 and min(batch_sizes) == 1

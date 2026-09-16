from pathlib import Path

import casadi as ca
import numpy as np
from pymatgen.core import Structure

from nagen.ipopt_opt.direct import (
    DirectIPOPTConfig,
    build_fixed_topology,
    canonical_lattice,
    geometry_constraints,
    lattice_parameters,
)


def shooting_structure() -> Structure:
    return Structure.from_file(
        Path(__file__).parents[1] / "data" / "shooting_08.cif"
    )


def test_canonical_lattice_preserves_metric_and_volume():
    original = shooting_structure().lattice.matrix
    canonical = canonical_lattice(original)
    np.testing.assert_allclose(
        canonical @ canonical.T, original @ original.T, atol=1.0e-10
    )
    assert np.linalg.det(canonical) > 0
    np.testing.assert_allclose(np.linalg.det(canonical), abs(np.linalg.det(original)))


def test_shooting_08_initial_geometry_is_ipopt_feasible():
    structure = shooting_structure()
    elements = [str(site.specie) for site in structure]
    frac = np.asarray(structure.frac_coords)
    lattice = canonical_lattice(structure.lattice.matrix)
    topology = build_fixed_topology(elements, frac, lattice)
    x0 = np.concatenate((frac.reshape(-1), lattice_parameters(lattice)))
    variable = ca.MX.sym("x", len(x0))
    constraints, names = geometry_constraints(
        variable, topology, len(elements), DirectIPOPTConfig()
    )
    values = np.asarray(ca.Function("geometry", [variable], [constraints])(x0)).ravel()
    assert len(names) == 1517
    assert topology.maximum_shared_oxygen_count == 2
    assert len(topology.adjacencies) == 3
    assert all(count == 4 for count in topology.coordination_counts.values())
    assert values.max() <= 0

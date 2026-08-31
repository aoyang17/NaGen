"""Post-generation UMA relaxation with strict convergence semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RelaxationConfig:
    fire_fmax_eV_A: float = 0.05
    fire_steps: int = 1500
    bfgs_fmax_eV_A: float = 0.01
    bfgs_steps: int = 1500


@dataclass
class RelaxationResult:
    atoms: Any
    converged: bool
    status: str
    final_fmax_eV_A: float
    fire_converged: bool
    bfgs_converged: bool


def maximum_force(atoms: Any) -> float:
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return float(np.linalg.norm(forces, axis=1).max(initial=0.0))


def relax_structure(
    atoms: Any, calculator: Any, output_directory: str,
    config: RelaxationConfig = RelaxationConfig(),
    fire_factory: Callable[..., Any] | None = None,
    bfgs_factory: Callable[..., Any] | None = None,
) -> RelaxationResult:
    from ase.filters import ExpCellFilter
    from ase.optimize import BFGS, FIRE
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    atoms.calc = calculator
    fire_factory = fire_factory or FIRE
    bfgs_factory = bfgs_factory or BFGS
    cell_filter = ExpCellFilter(atoms)
    fire = fire_factory(
        cell_filter, trajectory=str(output / "fire.traj"),
        logfile=str(output / "fire.log"),
    )
    fire_reported = bool(fire.run(fmax=config.fire_fmax_eV_A, steps=config.fire_steps))
    fire_converged = fire_reported and maximum_force(atoms) <= config.fire_fmax_eV_A
    bfgs = bfgs_factory(
        atoms, trajectory=str(output / "bfgs.traj"),
        logfile=str(output / "bfgs.log"),
    )
    bfgs_reported = bool(bfgs.run(fmax=config.bfgs_fmax_eV_A, steps=config.bfgs_steps))
    final_fmax = maximum_force(atoms)
    bfgs_converged = bfgs_reported and final_fmax <= config.bfgs_fmax_eV_A
    # FIRE is the coarse cell-and-position preconditioner.  Reaching its step
    # limit is diagnostic, not a final failure, when the subsequent fine BFGS
    # stage reaches the locked final force tolerance.
    converged = bool(bfgs_converged)
    status = (
        "converged" if converged and fire_converged else
        "converged_after_fire_max_steps" if converged else
        "bfgs_not_converged"
    )
    return RelaxationResult(
        atoms, converged, status, final_fmax, fire_converged, bfgs_converged
    )


def unique_structure_indices(structures: list[Any], references: list[Any]) -> list[int]:
    """Exact StructureMatcher de-duplication against references and candidates."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    accepted: list[int] = []
    accepted_structures: list[Any] = []
    for index, structure in enumerate(structures):
        if any(matcher.fit(structure, reference) for reference in references):
            continue
        if any(matcher.fit(structure, previous) for previous in accepted_structures):
            continue
        accepted.append(index)
        accepted_structures.append(structure)
    return accepted

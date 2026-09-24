"""Translate a validated constraint configuration into runtime policies."""

from __future__ import annotations

from dataclasses import dataclass

from shootingcsp.inverse.spec import OptimizationSpec
from .schema import ConstraintConfig


def to_optimization_spec(config: ConstraintConfig) -> OptimizationSpec:
    composition = config.composition
    pbc = config.pbc
    return OptimizationSpec(
        elements=composition.elements,
        fe_valence_min=config.valence.fe_min,
        fe_valence_max=config.valence.fe_max,
        n_primitive_min=config.cell.primitive_atom_count.min,
        n_primitive_max=config.cell.primitive_atom_count.max,
        volume_per_atom_min_A3=config.cell.volume_per_atom_A3.min,
        volume_per_atom_max_A3=config.cell.volume_per_atom_A3.max,
        pair_minima=dict(pbc.pair_minima_A),
        generic_min_radius_fraction=pbc.generic_radius_fraction,
        p_o_bond_cutoff=config.coordination.P.cutoff_A,
        fe_o_bond_cutoff=config.coordination.Fe.cutoff_A,
        capacity_min_mAh_g=config.capacity.min_mAh_g,
        hull_max_eV_atom=config.energy.hull_max_eV_atom,
        p_coordination_options=config.coordination.P.allowed,
        fe_coordination_options=config.coordination.Fe.allowed,
    )


def conditioning_counts(config: ConstraintConfig) -> dict[str, int]:
    return {element: int(config.composition.fixed_counts.get(element, 0)) for element in config.composition.elements}


def optimization_weights(config: ConstraintConfig) -> dict[str, float]:
    return config.optimization.as_dict()


__all__ = ["conditioning_counts", "optimization_weights", "to_optimization_spec"]

"""Executable constants frozen by ``docs/OPTIMIZATION.md``.

Historical multi-transition-metal settings are intentionally absent from the
new Na--Fe--P--O experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OptimizationSpec:
    elements: tuple[str, ...] = ("Na", "Fe", "P", "O")
    fe_valence_min: float = 2.0
    fe_valence_max: float = 4.5
    y_min: float = 0.0
    y_max: float = 0.5
    n_primitive_min: int = 23
    n_primitive_max: int = 184
    volume_per_atom_min_A3: float = 10.5
    volume_per_atom_max_A3: float = 20.5
    pair_minima: dict[str, float] = field(default_factory=lambda: {
        "P-O": 1.40, "Fe-O": 1.55, "Na-O": 2.00,
        "O-O": 2.00, "P-P": 2.60, "Na-Na": 2.20,
    })
    atomic_masses: dict[str, float] = field(default_factory=lambda: {
        "Na": 22.98976928, "Fe": 55.845, "P": 30.973761998, "O": 15.999,
    })
    covalent_radii: dict[str, float] = field(default_factory=lambda: {
        "Na": 1.66, "Fe": 1.32, "P": 1.07, "O": 0.66,
    })
    generic_min_radius_fraction: float = 0.75
    p_o_bond_cutoff: float = 2.00
    fe_o_bond_cutoff: float = 2.50
    capacity_min_mAh_g: float = 130.0
    hull_max_eV_atom: float = 0.150
    faraday_C_mol: float = 96485.33212

    @property
    def element_to_index(self) -> dict[str, int]:
        return {element: index + 1 for index, element in enumerate(self.elements)}

    @property
    def index_to_element(self) -> dict[int, str]:
        return {index: element for element, index in self.element_to_index.items()}


DEFAULT_SPEC = OptimizationSpec()

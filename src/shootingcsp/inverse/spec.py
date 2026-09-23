"""Executable constants frozen by ``docs/OPTIMIZATION.md``.

Historical multi-transition-metal settings are intentionally absent from the
new Na--Fe--P--O experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


FE_COORDINATION_CHOICES = (4, 5, 6)
DEFAULT_FE_COORDINATION_OPTIONS = FE_COORDINATION_CHOICES


def normalize_fe_coordination_options(options: Iterable[int] | int) -> tuple[int, ...]:
    """Return a sorted, non-empty subset of the supported Fe CN choices."""
    if isinstance(options, int) and not isinstance(options, bool):
        values = [options]
    else:
        try:
            values = list(options)
        except TypeError as error:
            raise ValueError("Fe coordination options must be an iterable of integers") from error
    if not values:
        raise ValueError("at least one Fe coordination option is required")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("Fe coordination options must be integers")
    normalized = tuple(sorted(set(values)))
    unknown = sorted(set(normalized) - set(FE_COORDINATION_CHOICES))
    if unknown:
        raise ValueError(
            f"unsupported Fe coordination options {unknown}; "
            f"choose from {FE_COORDINATION_CHOICES}"
        )
    return normalized


def parse_fe_coordination_options(value: str | Iterable[int] | int) -> tuple[int, ...]:
    """Parse ``4,5`` / ``4 5`` CLI input or normalize an iterable."""
    if not isinstance(value, str):
        return normalize_fe_coordination_options(value)
    text = value.strip().replace(";", ",").replace(" ", ",")
    if not text:
        raise ValueError("Fe coordination options cannot be empty")
    parts = [part for part in text.split(",") if part]
    try:
        parsed = [int(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"invalid Fe coordination option list: {value!r}") from error
    return normalize_fe_coordination_options(parsed)


def format_fe_coordination_options(options: Iterable[int]) -> str:
    return ",".join(str(value) for value in normalize_fe_coordination_options(options))


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
    fe_coordination_options: tuple[int, ...] = DEFAULT_FE_COORDINATION_OPTIONS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fe_coordination_options",
            normalize_fe_coordination_options(self.fe_coordination_options),
        )

    @property
    def element_to_index(self) -> dict[str, int]:
        return {element: index + 1 for index, element in enumerate(self.elements)}

    @property
    def index_to_element(self) -> dict[int, str]:
        return {index: element for element, index in self.element_to_index.items()}


DEFAULT_SPEC = OptimizationSpec()

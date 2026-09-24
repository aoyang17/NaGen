"""Load, normalize, validate, and hash ShootingCSP constraint JSON."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from .schema import ConstraintConfig

SCHEMA_VERSION = "shootingcsp.constraints.v1"
ELEMENTS = ("Na", "Fe", "P", "O")
PAIR_ORDER = {element: index for index, element in enumerate(ELEMENTS)}


def default_constraint_path() -> Path:
    return Path(files("shootingcsp.config").joinpath("na_fe_po_default.json"))


def default_constraint_mapping() -> dict[str, Any]:
    return json.loads(default_constraint_path().read_text())


def _canonical_pair(key: str) -> str:
    pieces = key.replace("–", "-").replace("_", "-").split("-")
    if len(pieces) != 2 or any(piece not in ELEMENTS for piece in pieces):
        raise ValueError(f"invalid pair-minimum key: {key!r}")
    first, second = sorted(pieces, key=PAIR_ORDER.__getitem__)
    return f"{first}-{second}"


def _validate_range(name: str, value: Mapping[str, Any], *, integer: bool = False) -> None:
    minimum, maximum = value["min"], value["max"]
    if integer and (type(minimum) is not int or type(maximum) is not int):
        raise ValueError(f"{name}: min/max must be integers")
    if minimum > maximum:
        raise ValueError(f"{name}: min must not exceed max")


def validate_constraint_mapping(value: Mapping[str, Any]) -> None:
    """Validate semantic constraints that JSON Schema cannot express alone."""
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {value.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    composition = value["composition"]
    elements = tuple(composition["elements"])
    unknown = sorted(set(elements) - set(ELEMENTS))
    if unknown:
        raise ValueError(f"composition: unsupported elements {unknown}")
    if len(set(elements)) != len(elements):
        raise ValueError("composition: elements must be unique")
    fixed = {str(k): int(v) for k, v in composition["fixed_counts"].items()}
    unknown_fixed = sorted(set(fixed) - set(elements))
    if unknown_fixed:
        raise ValueError(f"composition.fixed_counts contains elements outside composition: {unknown_fixed}")
    if any(count < 0 for count in fixed.values()):
        raise ValueError("composition.fixed_counts values must be nonnegative")
    if composition["require_all_elements"] and any(fixed.get(element, 0) < 1 for element in elements):
        raise ValueError("composition: require_all_elements is true but a fixed count is zero")

    valence = value["valence"]
    if valence["fe_min"] > valence["fe_max"]:
        raise ValueError("valence: fe_min must not exceed fe_max")
    if value["capacity"]["min_mAh_g"] < 0:
        raise ValueError("capacity.min_mAh_g must be nonnegative")
    _validate_range("cell.volume_per_atom_A3", value["cell"]["volume_per_atom_A3"])
    _validate_range("cell.primitive_atom_count", value["cell"]["primitive_atom_count"], integer=True)

    pbc = value["pbc"]
    if pbc["generic_radius_fraction"] <= 0:
        raise ValueError("pbc.generic_radius_fraction must be positive")
    normalized_pairs: dict[str, float] = {}
    for key, threshold in pbc["pair_minima_A"].items():
        canonical = _canonical_pair(str(key))
        if canonical in normalized_pairs:
            raise ValueError(f"pbc.pair_minima_A contains duplicate pair {canonical}")
        if float(threshold) <= 0:
            raise ValueError(f"pbc.pair_minima_A[{key}] must be positive")
        normalized_pairs[canonical] = float(threshold)

    for element, rule in value["coordination"].items():
        allowed = tuple(sorted({int(item) for item in rule["allowed"]}))
        if not allowed:
            raise ValueError(f"coordination.{element}.allowed cannot be empty")
        if element == "Fe" and not set(allowed).issubset({4, 5, 6}):
            raise ValueError("coordination.Fe.allowed must be a subset of {4,5,6}")
        if element == "P" and any(item < 1 or item > 8 for item in allowed):
            raise ValueError("coordination.P.allowed values must be in 1..8")
        if float(rule["cutoff_A"]) <= 0:
            raise ValueError(f"coordination.{element}.cutoff_A must be positive")

    polyhedra = value["polyhedra"]
    if int(polyhedra["fe_shared_oxygen_max"]) < 0:
        raise ValueError("polyhedra.fe_shared_oxygen_max must be nonnegative")
    angle = float(polyhedra["minimum_face_angle_deg"])
    if not 0 <= angle <= 180:
        raise ValueError("polyhedra.minimum_face_angle_deg must be in [0,180]")

    if float(value["energy"]["hull_max_eV_atom"]) < 0:
        raise ValueError("energy.hull_max_eV_atom must be nonnegative")
    if float(value["force"]["max_eV_A"]) < 0:
        raise ValueError("force.max_eV_A must be nonnegative")

    relaxation = value["relaxation"]
    for key in ("maxstep_A", "max_force_eV_A", "refine_trigger_force_eV_A", "refine_trigger_hull_eV_atom"):
        if float(relaxation[key]) < 0:
            raise ValueError(f"relaxation.{key} must be nonnegative")
    for key in ("max_steps", "refine_max_steps"):
        if int(relaxation[key]) < 0:
            raise ValueError(f"relaxation.{key} must be nonnegative")

    novelty = value["novelty"]
    unknown_strip = sorted(set(novelty["strip_elements"]) - set(elements))
    if unknown_strip:
        raise ValueError(f"novelty.strip_elements contains unavailable elements: {unknown_strip}")
    if float(novelty["minimum_descriptor_distance"]) < 0:
        raise ValueError("novelty.minimum_descriptor_distance must be nonnegative")

    for key, weight in value["optimization"].items():
        if float(weight) < 0:
            raise ValueError(f"optimization.{key} must be nonnegative")

    source = value["source_precheck"]
    _validate_range("source_precheck.volume_per_atom_A3", source["volume_per_atom_A3"])
    if float(source["max_cell_condition_number"]) <= 0:
        raise ValueError("source_precheck.max_cell_condition_number must be positive")
    if float(source["minimum_distance_A"]) <= 0:
        raise ValueError("source_precheck.minimum_distance_A must be positive")


def normalize_constraint_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(value))
    normalized_pairs = {
        _canonical_pair(str(key)): float(threshold)
        for key, threshold in data["pbc"]["pair_minima_A"].items()
    }
    data["pbc"]["pair_minima_A"] = dict(sorted(normalized_pairs.items()))
    for element in ("P", "Fe"):
        data["coordination"][element]["allowed"] = sorted(
            {int(item) for item in data["coordination"][element]["allowed"]}
        )
    return data


def load_constraint_mapping(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else default_constraint_path()
    value = json.loads(source.read_text())
    validate_constraint_mapping(value)
    return normalize_constraint_mapping(value)


def load_constraint_config(path: str | Path | None = None) -> ConstraintConfig:
    return ConstraintConfig.from_mapping(load_constraint_mapping(path))


def constraint_config_sha256(config: ConstraintConfig | Mapping[str, Any]) -> str:
    value = config.as_dict() if isinstance(config, ConstraintConfig) else normalize_constraint_mapping(config)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "constraint_config_sha256",
    "default_constraint_mapping",
    "default_constraint_path",
    "load_constraint_config",
    "load_constraint_mapping",
    "normalize_constraint_mapping",
    "validate_constraint_mapping",
]

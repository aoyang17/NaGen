"""Typed model for the ShootingCSP constraint configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RangePolicy:
    enabled: bool
    min: float
    max: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RangePolicy":
        return cls(enabled=bool(value.get("enabled", True)), min=float(value["min"]), max=float(value["max"]))


@dataclass(frozen=True)
class IntRangePolicy:
    enabled: bool
    min: int
    max: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IntRangePolicy":
        return cls(enabled=bool(value.get("enabled", True)), min=int(value["min"]), max=int(value["max"]))


@dataclass(frozen=True)
class CompositionPolicy:
    enabled: bool
    elements: tuple[str, ...]
    require_all_elements: bool
    fixed_counts: Mapping[str, int]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompositionPolicy":
        return cls(
            enabled=bool(value["enabled"]),
            elements=tuple(str(item) for item in value["elements"]),
            require_all_elements=bool(value["require_all_elements"]),
            fixed_counts={str(k): int(v) for k, v in value["fixed_counts"].items()},
        )


@dataclass(frozen=True)
class ValencePolicy:
    enabled: bool
    fe_min: float
    fe_max: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ValencePolicy":
        return cls(enabled=bool(value["enabled"]), fe_min=float(value["fe_min"]), fe_max=float(value["fe_max"]))


@dataclass(frozen=True)
class CapacityPolicy:
    enabled: bool
    min_mAh_g: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CapacityPolicy":
        return cls(enabled=bool(value["enabled"]), min_mAh_g=float(value["min_mAh_g"]))


@dataclass(frozen=True)
class CellPolicy:
    volume_per_atom_A3: RangePolicy
    primitive_atom_count: IntRangePolicy

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CellPolicy":
        return cls(
            volume_per_atom_A3=RangePolicy.from_mapping(value["volume_per_atom_A3"]),
            primitive_atom_count=IntRangePolicy.from_mapping(value["primitive_atom_count"]),
        )


@dataclass(frozen=True)
class PBCPolicy:
    enabled: bool
    generic_radius_fraction: float
    pair_minima_A: Mapping[str, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PBCPolicy":
        return cls(
            enabled=bool(value["enabled"]),
            generic_radius_fraction=float(value["generic_radius_fraction"]),
            pair_minima_A={str(k): float(v) for k, v in value["pair_minima_A"].items()},
        )


@dataclass(frozen=True)
class CoordinationRule:
    enabled: bool
    allowed: tuple[int, ...]
    cutoff_A: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CoordinationRule":
        return cls(
            enabled=bool(value["enabled"]),
            allowed=tuple(sorted({int(item) for item in value["allowed"]})),
            cutoff_A=float(value["cutoff_A"]),
        )


@dataclass(frozen=True)
class CoordinationPolicy:
    P: CoordinationRule
    Fe: CoordinationRule

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CoordinationPolicy":
        return cls(P=CoordinationRule.from_mapping(value["P"]), Fe=CoordinationRule.from_mapping(value["Fe"]))


@dataclass(frozen=True)
class PolyhedraPolicy:
    enabled: bool
    require_center_inside: bool
    fe_shared_oxygen_max: int
    forbid_coplanar_faces: bool
    minimum_face_angle_deg: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolyhedraPolicy":
        return cls(
            enabled=bool(value["enabled"]),
            require_center_inside=bool(value["require_center_inside"]),
            fe_shared_oxygen_max=int(value["fe_shared_oxygen_max"]),
            forbid_coplanar_faces=bool(value["forbid_coplanar_faces"]),
            minimum_face_angle_deg=float(value["minimum_face_angle_deg"]),
        )


@dataclass(frozen=True)
class OxygenCoveragePolicy:
    enabled: bool
    require_all: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OxygenCoveragePolicy":
        return cls(enabled=bool(value["enabled"]), require_all=bool(value["require_all"]))


@dataclass(frozen=True)
class EnergyPolicy:
    enabled: bool
    hull_max_eV_atom: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EnergyPolicy":
        return cls(enabled=bool(value["enabled"]), hull_max_eV_atom=float(value["hull_max_eV_atom"]))


@dataclass(frozen=True)
class ForcePolicy:
    enabled: bool
    max_eV_A: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ForcePolicy":
        return cls(enabled=bool(value["enabled"]), max_eV_A=float(value["max_eV_A"]))


@dataclass(frozen=True)
class RelaxationPolicy:
    enabled: bool
    maxstep_A: float
    max_steps: int
    max_force_eV_A: float
    refine_max_steps: int
    refine_trigger_force_eV_A: float
    refine_trigger_hull_eV_atom: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RelaxationPolicy":
        return cls(
            enabled=bool(value["enabled"]),
            maxstep_A=float(value["maxstep_A"]),
            max_steps=int(value["max_steps"]),
            max_force_eV_A=float(value["max_force_eV_A"]),
            refine_max_steps=int(value["refine_max_steps"]),
            refine_trigger_force_eV_A=float(value["refine_trigger_force_eV_A"]),
            refine_trigger_hull_eV_atom=float(value["refine_trigger_hull_eV_atom"]),
        )


@dataclass(frozen=True)
class NoveltyPolicy:
    enabled: bool
    strip_elements: tuple[str, ...]
    require_structure_nonmatch: bool
    require_topology_nonmatch: bool
    minimum_descriptor_distance: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NoveltyPolicy":
        return cls(
            enabled=bool(value["enabled"]),
            strip_elements=tuple(str(item) for item in value["strip_elements"]),
            require_structure_nonmatch=bool(value["require_structure_nonmatch"]),
            require_topology_nonmatch=bool(value["require_topology_nonmatch"]),
            minimum_descriptor_distance=float(value["minimum_descriptor_distance"]),
        )


@dataclass(frozen=True)
class OptimizationPolicy:
    energy_weight: float
    distance_weight: float
    p_coordination_weight: float
    fe_coordination_weight: float
    volume_weight: float
    source_prior_weight: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OptimizationPolicy":
        return cls(**{key: float(value[key]) for key in (
            "energy_weight", "distance_weight", "p_coordination_weight",
            "fe_coordination_weight", "volume_weight", "source_prior_weight",
        )})

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SourcePrecheckPolicy:
    enabled: bool
    volume_per_atom_A3: RangePolicy
    max_cell_condition_number: float
    minimum_distance_A: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourcePrecheckPolicy":
        return cls(
            enabled=bool(value["enabled"]),
            volume_per_atom_A3=RangePolicy.from_mapping(value["volume_per_atom_A3"]),
            max_cell_condition_number=float(value["max_cell_condition_number"]),
            minimum_distance_A=float(value["minimum_distance_A"]),
        )


@dataclass(frozen=True)
class ConstraintConfig:
    schema_version: str
    name: str
    composition: CompositionPolicy
    valence: ValencePolicy
    capacity: CapacityPolicy
    cell: CellPolicy
    pbc: PBCPolicy
    coordination: CoordinationPolicy
    polyhedra: PolyhedraPolicy
    oxygen_coverage: OxygenCoveragePolicy
    energy: EnergyPolicy
    force: ForcePolicy
    relaxation: RelaxationPolicy
    novelty: NoveltyPolicy
    optimization: OptimizationPolicy
    source_precheck: SourcePrecheckPolicy

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConstraintConfig":
        return cls(
            schema_version=str(value["schema_version"]),
            name=str(value.get("name", "unnamed")),
            composition=CompositionPolicy.from_mapping(value["composition"]),
            valence=ValencePolicy.from_mapping(value["valence"]),
            capacity=CapacityPolicy.from_mapping(value["capacity"]),
            cell=CellPolicy.from_mapping(value["cell"]),
            pbc=PBCPolicy.from_mapping(value["pbc"]),
            coordination=CoordinationPolicy.from_mapping(value["coordination"]),
            polyhedra=PolyhedraPolicy.from_mapping(value["polyhedra"]),
            oxygen_coverage=OxygenCoveragePolicy.from_mapping(value["oxygen_coverage"]),
            energy=EnergyPolicy.from_mapping(value["energy"]),
            force=ForcePolicy.from_mapping(value["force"]),
            relaxation=RelaxationPolicy.from_mapping(value["relaxation"]),
            novelty=NoveltyPolicy.from_mapping(value["novelty"]),
            optimization=OptimizationPolicy.from_mapping(value["optimization"]),
            source_precheck=SourcePrecheckPolicy.from_mapping(value["source_precheck"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "CapacityPolicy", "CellPolicy", "CompositionPolicy", "ConstraintConfig",
    "CoordinationPolicy", "CoordinationRule", "EnergyPolicy", "ForcePolicy",
    "IntRangePolicy", "NoveltyPolicy", "OptimizationPolicy",
    "OxygenCoveragePolicy", "PBCPolicy", "PolyhedraPolicy", "RangePolicy",
    "RelaxationPolicy", "SourcePrecheckPolicy", "ValencePolicy",
]

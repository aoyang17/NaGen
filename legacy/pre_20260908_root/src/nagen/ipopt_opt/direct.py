"""Standalone direct-space IPOPT crystal refinement.

The atom count and species are immutable.  Fractional coordinates and the six
independent entries of a canonical right-handed lattice are optimized.  The
trained E_hull surrogate provides the property objective, UMA provides only
mechanical constraints, and CasADi expresses the sparse analytic geometry
constraints and calls IPOPT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import casadi as ca
import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from scipy.spatial import ConvexHull

from nagen.inverse.constraints import (
    bond_cutoff,
    evaluate_feasibility,
    minimum_distance,
)
from nagen.surrogate.ehull_mace import (
    load_surrogate,
    structure_to_model_batch,
)


@dataclass(frozen=True)
class DirectIPOPTConfig:
    energy_scale_eV_atom: float = 0.15
    hull_max_eV_atom: float = 0.15
    force_tolerance_eV_A: float = 0.01
    stress_tolerance_GPa: float = 0.10
    force_constraint_scale_eV_A: float = 1.0
    stress_constraint_scale_eV_A3: float = 0.01
    fractional_trust_radius: float = 0.08
    lattice_relative_trust_radius: float = 0.08
    coordination_margin_A: float = 0.01
    coordination_constraint_scale_A: float = 0.01
    polyhedron_volume_margin_A3: float = 1.0e-4
    polyhedron_constraint_scale_A3: float = 1.0e-4
    fe_adjacency_min_angle_deg: float = 30.0
    fe_adjacency_constraint_scale: float = 0.01
    max_iterations: int = 40
    max_wall_time_seconds: float = 900.0
    ipopt_tolerance: float = 1.0e-4
    print_level: int = 5

    def __post_init__(self) -> None:
        if self.energy_scale_eV_atom <= 0:
            raise ValueError("energy scale must be positive")
        if self.force_constraint_scale_eV_A <= 0 or self.stress_constraint_scale_eV_A3 <= 0:
            raise ValueError("mechanical constraint scales must be positive")
        if not 0 <= self.coordination_margin_A < 0.1:
            raise ValueError("coordination margin must be in [0, 0.1) A")
        if self.coordination_constraint_scale_A <= 0:
            raise ValueError("coordination constraint scale must be positive")
        if self.polyhedron_constraint_scale_A3 <= 0:
            raise ValueError("polyhedron constraint scale must be positive")
        if not 0 < self.fe_adjacency_min_angle_deg < 90:
            raise ValueError("Fe adjacency angle must be between 0 and 90 degrees")
        if self.fe_adjacency_constraint_scale <= 0:
            raise ValueError("Fe adjacency constraint scale must be positive")


def canonical_lattice(matrix: np.ndarray) -> np.ndarray:
    """Return a lower-triangular cell with the same row-vector Gram matrix."""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    canonical = np.linalg.cholesky(matrix @ matrix.T)
    if np.linalg.det(canonical) <= 0:
        raise ValueError("cannot construct a right-handed canonical lattice")
    return canonical


def lattice_parameters(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return matrix[[0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2]]


def lattice_from_parameters_numpy(values: np.ndarray) -> np.ndarray:
    a, bx, by, cx, cy, cz = np.asarray(values, dtype=np.float64)
    return np.asarray(((a, 0.0, 0.0), (bx, by, 0.0), (cx, cy, cz)))


def _nearest_shift(delta: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    best_shift: np.ndarray | None = None
    best = float("inf")
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shift = np.asarray((i, j, k), dtype=np.float64)
                squared = float(np.dot((delta + shift) @ lattice, (delta + shift) @ lattice))
                if squared < best:
                    best = squared
                    best_shift = shift
    assert best_shift is not None
    return best_shift


@dataclass(frozen=True)
class PairRecord:
    i: int
    j: int
    shift: tuple[float, float, float]
    minimum_A: float


@dataclass(frozen=True)
class CoordinationRecord:
    center: int
    oxygen: int
    shift: tuple[float, float, float]
    cutoff_A: float
    bonded: bool


@dataclass(frozen=True)
class FaceRecord:
    center: int
    vertices: tuple[tuple[int, tuple[float, float, float]], ...]


@dataclass(frozen=True)
class AdjacencyRecord:
    first: int
    second: int
    first_face: FaceRecord
    second_face: FaceRecord
    shared_oxygens: tuple[int, ...]


@dataclass(frozen=True)
class FixedTopology:
    pairs: tuple[PairRecord, ...]
    coordination: tuple[CoordinationRecord, ...]
    faces: tuple[FaceRecord, ...]
    adjacencies: tuple[AdjacencyRecord, ...]
    coordination_counts: dict[int, int]
    maximum_shared_oxygen_count: int


def build_fixed_topology(
    elements: list[str], frac: np.ndarray, lattice: np.ndarray,
) -> FixedTopology:
    frac = np.asarray(frac, dtype=np.float64)
    pairs: list[PairRecord] = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            shift = _nearest_shift(frac[j] - frac[i], lattice)
            pairs.append(PairRecord(
                i, j, tuple(shift.tolist()), minimum_distance(elements[i], elements[j])
            ))

    oxygen_indices = [i for i, element in enumerate(elements) if element == "O"]
    coordination: list[CoordinationRecord] = []
    neighbor_images: dict[int, list[tuple[int, tuple[float, float, float]]]] = {}
    neighbor_sets: dict[int, set[int]] = {}
    for center, element in enumerate(elements):
        if element not in {"P", "Fe"}:
            continue
        cutoff = bond_cutoff(element, "O")
        local: list[tuple[int, tuple[float, float, float]]] = []
        for oxygen in oxygen_indices:
            shift_array = _nearest_shift(frac[oxygen] - frac[center], lattice)
            shift = tuple(shift_array.tolist())
            vector = (frac[oxygen] - frac[center] + shift_array) @ lattice
            bonded = float(np.linalg.norm(vector)) <= cutoff + 1.0e-10
            coordination.append(CoordinationRecord(
                center, oxygen, shift, cutoff, bonded
            ))
            if bonded:
                local.append((oxygen, shift))
        count = len(local)
        if element == "P" and count != 4:
            raise ValueError(f"P site {center} has coordination {count}, expected 4")
        if element == "Fe" and count not in {4, 5, 6}:
            raise ValueError(f"Fe site {center} has coordination {count}, expected 4, 5, or 6")
        neighbor_images[center] = local
        neighbor_sets[center] = {oxygen for oxygen, _ in local}

    fe_indices = [i for i, element in enumerate(elements) if element == "Fe"]
    maximum_shared = 0
    for row, first in enumerate(fe_indices):
        for second in fe_indices[row + 1 :]:
            maximum_shared = max(
                maximum_shared, len(neighbor_sets[first] & neighbor_sets[second])
            )
    if maximum_shared > 2:
        raise ValueError(
            f"initial Fe polyhedra share {maximum_shared} oxygen atoms; maximum is 2"
        )

    faces: list[FaceRecord] = []
    faces_by_center: dict[int, list[FaceRecord]] = {}
    for center, neighbors in neighbor_images.items():
        points = np.stack([
            (frac[oxygen] - frac[center] + np.asarray(shift)) @ lattice
            for oxygen, shift in neighbors
        ])
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ordered = [neighbors[int(index)] for index in simplex]
            vectors = points[simplex]
            phi = float(np.dot(np.cross(vectors[1] - vectors[0], vectors[2] - vectors[0]), -vectors[0]) / 6.0)
            if phi < 0:
                ordered[1], ordered[2] = ordered[2], ordered[1]
            face = FaceRecord(center, tuple(ordered))
            faces.append(face)
            faces_by_center.setdefault(center, []).append(face)

    def inward_normal(face: FaceRecord) -> np.ndarray:
        vectors = [
            (frac[oxygen] - frac[face.center] + np.asarray(shift)) @ lattice
            for oxygen, shift in face.vertices
        ]
        return np.cross(vectors[1] - vectors[0], vectors[2] - vectors[0])

    def facing_face(center: int, direction: np.ndarray) -> FaceRecord:
        # Faces were oriented inward above, hence the outward normal is -normal.
        return max(
            faces_by_center[center],
            key=lambda face: float(np.dot(-inward_normal(face), direction)
                                   / np.linalg.norm(inward_normal(face))),
        )

    adjacencies: list[AdjacencyRecord] = []
    for row, first in enumerate(fe_indices):
        for second in fe_indices[row + 1:]:
            shared = tuple(sorted(neighbor_sets[first] & neighbor_sets[second]))
            if not shared:
                continue
            shift = _nearest_shift(frac[second] - frac[first], lattice)
            direction = (frac[second] - frac[first] + shift) @ lattice
            adjacencies.append(AdjacencyRecord(
                first, second,
                facing_face(first, direction),
                facing_face(second, -direction),
                shared,
            ))
    return FixedTopology(
        tuple(pairs), tuple(coordination), tuple(faces), tuple(adjacencies),
        {center: len(values) for center, values in neighbor_images.items()},
        maximum_shared,
    )


def _casadi_lattice(x: ca.MX, offset: int) -> ca.MX:
    a, bx, by, cx, cy, cz = [x[offset + index] for index in range(6)]
    return ca.vertcat(
        ca.horzcat(a, 0, 0),
        ca.horzcat(bx, by, 0),
        ca.horzcat(cx, cy, cz),
    )


def _casadi_relative(
    x: ca.MX, lattice: ca.MX, i: int, j: int,
    shift: tuple[float, float, float],
) -> ca.MX:
    delta = ca.vertcat(*[
        x[3 * j + axis] - x[3 * i + axis] + shift[axis]
        for axis in range(3)
    ])
    return lattice.T @ delta


def geometry_constraints(
    x: ca.MX, topology: FixedTopology, n_atoms: int,
    config: DirectIPOPTConfig,
) -> tuple[ca.MX, list[str]]:
    lattice = _casadi_lattice(x, 3 * n_atoms)
    values: list[ca.MX] = []
    names: list[str] = []
    volume = ca.det(lattice)
    values.extend(((10.5 * n_atoms - volume) / (5.0 * n_atoms),
                   (volume - 20.5 * n_atoms) / (5.0 * n_atoms)))
    names.extend(("volume_lower", "volume_upper"))
    for pair in topology.pairs:
        vector = _casadi_relative(x, lattice, pair.i, pair.j, pair.shift)
        values.append((pair.minimum_A**2 - ca.dot(vector, vector)) / pair.minimum_A**2)
        names.append(f"minimum_distance:{pair.i}:{pair.j}")
    for record in topology.coordination:
        vector = _casadi_relative(
            x, lattice, record.center, record.oxygen, record.shift
        )
        squared = ca.dot(vector, vector)
        if record.bonded:
            boundary = record.cutoff_A - config.coordination_margin_A
            values.append(
                (squared - boundary**2)
                / (2.0 * boundary * config.coordination_constraint_scale_A)
            )
            names.append(f"bond_upper:{record.center}:{record.oxygen}")
        else:
            boundary = record.cutoff_A + config.coordination_margin_A
            values.append(
                (boundary**2 - squared)
                / (2.0 * boundary * config.coordination_constraint_scale_A)
            )
            names.append(f"nonbond_lower:{record.center}:{record.oxygen}")
    for face_index, face in enumerate(topology.faces):
        vectors = [
            _casadi_relative(x, lattice, face.center, oxygen, shift)
            for oxygen, shift in face.vertices
        ]
        phi = ca.dot(ca.cross(vectors[1] - vectors[0], vectors[2] - vectors[0]), -vectors[0]) / 6.0
        values.append(
            (config.polyhedron_volume_margin_A3 - phi)
            / config.polyhedron_constraint_scale_A3
        )
        names.append(f"polyhedron_center:{face.center}:{face_index}")
    cosine_limit_squared = float(np.cos(np.deg2rad(
        config.fe_adjacency_min_angle_deg
    )) ** 2)
    for adjacency in topology.adjacencies:
        normals = []
        for face in (adjacency.first_face, adjacency.second_face):
            vectors = [
                _casadi_relative(x, lattice, face.center, oxygen, shift)
                for oxygen, shift in face.vertices
            ]
            normals.append(ca.cross(
                vectors[1] - vectors[0], vectors[2] - vectors[0]
            ))
        norm_product = ca.dot(normals[0], normals[0]) * ca.dot(normals[1], normals[1])
        cosine_squared = ca.dot(normals[0], normals[1]) ** 2 / (norm_product + 1.0e-24)
        values.append(
            (cosine_squared - cosine_limit_squared)
            / config.fe_adjacency_constraint_scale
        )
        names.append(f"fe_polyhedron_adjacency:{adjacency.first}:{adjacency.second}")
    return ca.vertcat(*values), names


def fixed_topology_checks(
    elements: list[str], frac: np.ndarray, lattice: np.ndarray,
    topology: FixedTopology, config: DirectIPOPTConfig,
) -> dict[str, Any]:
    """Exact, unscaled acceptance checks for the immutable local topology."""
    frac = np.asarray(frac, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64)
    oxygen_indices = [i for i, element in enumerate(elements) if element == "O"]
    expected: dict[int, set[int]] = {}
    for record in topology.coordination:
        if record.bonded:
            expected.setdefault(record.center, set()).add(record.oxygen)
    observed: dict[int, set[int]] = {}
    site_details = []
    for center in sorted(expected):
        cutoff = bond_cutoff(elements[center], "O")
        neighbors = set()
        for oxygen in oxygen_indices:
            shift = _nearest_shift(frac[oxygen] - frac[center], lattice)
            distance = np.linalg.norm(
                (frac[oxygen] - frac[center] + shift) @ lattice
            )
            if distance <= cutoff + 1.0e-8:
                neighbors.add(oxygen)
        observed[center] = neighbors
        site_details.append({
            "site": center,
            "element": elements[center],
            "expected_oxygens": sorted(expected[center]),
            "observed_oxygens": sorted(neighbors),
            "preserved": neighbors == expected[center],
        })

    center_margins = []
    for face in topology.faces:
        vectors = [
            (frac[oxygen] - frac[face.center]
             + _nearest_shift(frac[oxygen] - frac[face.center], lattice)) @ lattice
            for oxygen, _ in face.vertices
        ]
        phi = float(np.dot(
            np.cross(vectors[1] - vectors[0], vectors[2] - vectors[0]),
            -vectors[0],
        ) / 6.0)
        center_margins.append(phi)

    adjacency_details = []
    for adjacency in topology.adjacencies:
        normals = []
        for face in (adjacency.first_face, adjacency.second_face):
            vectors = [
                (frac[oxygen] - frac[face.center]
                 + _nearest_shift(frac[oxygen] - frac[face.center], lattice)) @ lattice
                for oxygen, _ in face.vertices
            ]
            normals.append(np.cross(
                vectors[1] - vectors[0], vectors[2] - vectors[0]
            ))
        cosine = abs(float(np.dot(normals[0], normals[1]))) / (
            float(np.linalg.norm(normals[0]) * np.linalg.norm(normals[1])) + 1.0e-30
        )
        angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
        adjacency_details.append({
            "fe_sites": [adjacency.first, adjacency.second],
            "shared_oxygens": list(adjacency.shared_oxygens),
            "angle_deg": angle,
            "passes": angle + 1.0e-8 >= config.fe_adjacency_min_angle_deg,
        })

    fe_indices = [i for i, element in enumerate(elements) if element == "Fe"]
    maximum_shared = 0
    for row, first in enumerate(fe_indices):
        for second in fe_indices[row + 1:]:
            maximum_shared = max(
                maximum_shared,
                len(observed.get(first, set()) & observed.get(second, set())),
            )
    minimum_phi = min(center_margins, default=float("inf"))
    return {
        "coordination_preserved": all(item["preserved"] for item in site_details),
        "coordination_sites": site_details,
        "polyhedron_center_passes": bool(
            minimum_phi >= config.polyhedron_volume_margin_A3 - 1.0e-10
        ),
        "minimum_center_face_volume_A3": minimum_phi,
        "center_face_volume_count": len(center_margins),
        "fe_adjacency_passes": all(item["passes"] for item in adjacency_details),
        "fe_adjacencies": adjacency_details,
        "maximum_shared_oxygen_count": maximum_shared,
        "shared_oxygen_limit_passes": maximum_shared <= 2,
    }


def _load_uma_without_checksum(checkpoint: str, device: str):
    """Load the validated second-order UMA path without rereading 11 GB for SHA256."""
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(checkpoint, inference_settings="traineval", device=device)
    initialized = False

    def atomic_data(numbers, positions, cell, target_device):
        numbers = numbers.to(target_device)
        positions = positions.to(target_device)
        cell = cell.to(target_device)
        count = numbers.numel()
        return AtomicData(
            pos=positions, atomic_numbers=numbers, cell=cell,
            pbc=torch.ones(1, 3, dtype=torch.bool, device=target_device),
            natoms=torch.tensor([count], dtype=torch.long, device=target_device),
            edge_index=torch.empty(2, 0, dtype=torch.long, device=target_device),
            cell_offsets=torch.empty(0, 3, dtype=positions.dtype, device=target_device),
            nedges=torch.zeros(1, dtype=torch.long, device=target_device),
            charge=torch.zeros(1, dtype=torch.long, device=target_device),
            spin=torch.zeros(1, dtype=torch.long, device=target_device),
            fixed=torch.zeros(count, dtype=torch.long, device=target_device),
            tags=torch.zeros(count, dtype=torch.long, device=target_device),
            batch=torch.zeros(count, dtype=torch.long, device=target_device),
            dataset="omat",
        )

    def evaluate(numbers, frac, lattice):
        nonlocal initialized
        cell = lattice.to(device)
        numbers = numbers.to(device)
        positions = torch.einsum("bnd,bdk->bnk", frac.to(device), cell).squeeze(0)
        count = numbers.numel()
        if not initialized:
            predictor.predict(atomic_data(
                numbers.detach().cpu(), positions.detach().cpu(),
                cell.detach().cpu(), "cpu",
            ))
            initialized = True
        for head in predictor.model.module.output_heads.values():
            head.train(True)
        prediction = predictor.predict(atomic_data(numbers, positions, cell, device))
        energy = prediction.get("energy")
        forces = prediction.get("forces")
        stress = prediction.get("stress")
        if energy is None or forces is None or stress is None:
            raise RuntimeError("UMA prediction lacks energy, forces, or stress")
        if not forces.requires_grad or not stress.requires_grad:
            raise RuntimeError("UMA mechanical outputs do not retain a derivative graph")
        return energy.reshape(-1) / count, forces, stress.reshape(-1, 3, 3)

    return evaluate


class TorchPropertyProblem:
    """Surrogate objective plus UMA mechanical constraint callback."""

    def __init__(
        self, *, elements: list[str], initial_frac: np.ndarray,
        initial_lattice: np.ndarray, surrogate_checkpoint: str,
        uma_checkpoint: str, device: str, config: DirectIPOPTConfig,
    ) -> None:
        self.n_atoms = len(elements)
        self.n_variables = 3 * self.n_atoms + 6
        self.device = torch.device(device)
        self.config = config
        self.atomic_numbers = torch.tensor(
            [{"Na": 11, "Fe": 26, "P": 15, "O": 8}[element] for element in elements],
            dtype=torch.long, device=self.device,
        )
        self.uma = _load_uma_without_checksum(uma_checkpoint, device)
        self.surrogate, self.surrogate_payload = load_surrogate(
            surrogate_checkpoint, self.device
        )
        for parameter in self.surrogate.parameters():
            parameter.requires_grad_(False)
        initial_structure = Structure(
            Lattice(initial_lattice), elements, initial_frac
        )
        graph = structure_to_model_batch(
            initial_structure,
            self.surrogate_payload["element_to_index"],
            self.surrogate.config.cutoff_A,
            self.surrogate.config.max_neighbors,
        )
        self.graph = {
            key: value.to(self.device) for key, value in graph.items()
        }
        self.n_outputs = self.n_atoms + 4
        self.cache_values_x: np.ndarray | None = None
        self.cache_values: np.ndarray | None = None
        self.cache_jacobian_x: np.ndarray | None = None
        self.cache_jacobian: np.ndarray | None = None
        self.last_metrics: dict[str, float] = {}
        self.evaluations = 0
        self.jacobian_evaluations = 0

    def tensors(self, vector: np.ndarray):
        variable = torch.as_tensor(
            vector, dtype=torch.float32, device=self.device
        ).requires_grad_(True)
        frac = variable[: 3 * self.n_atoms].reshape(1, self.n_atoms, 3)
        a, bx, by, cx, cy, cz = variable[3 * self.n_atoms :]
        zero = variable.new_zeros(())
        lattice = torch.stack((
            torch.stack((a, zero, zero)),
            torch.stack((bx, by, zero)),
            torch.stack((cx, cy, cz)),
        )).unsqueeze(0)
        return variable, frac, lattice

    def _forward(self, vector: np.ndarray):
        variable, frac, lattice = self.tensors(vector)
        graph = dict(self.graph)
        graph["frac"] = frac[0]
        graph["lattice"] = lattice
        prediction = self.surrogate(graph).reshape(())
        _, forces, stress = self.uma(self.atomic_numbers, frac, lattice)
        objective = (prediction / self.config.energy_scale_eV_atom).square()
        force_norms = torch.sqrt(forces.square().sum(dim=-1) + 1.0e-16)
        force_max = force_norms.amax()
        stress_fro = torch.linalg.vector_norm(stress.reshape(-1), ord=2)
        stress_tolerance = self.config.stress_tolerance_GPa / 160.21766208
        force_constraints = (
            force_norms - self.config.force_tolerance_eV_A
        ) / self.config.force_constraint_scale_eV_A
        outputs = torch.cat((
            objective.reshape(1),
            ((prediction - self.config.hull_max_eV_atom)
             / self.config.energy_scale_eV_atom).reshape(1),
            (-prediction / self.config.energy_scale_eV_atom).reshape(1),
            force_constraints.reshape(-1),
            ((stress_fro - stress_tolerance)
             / self.config.stress_constraint_scale_eV_A3).reshape(1),
        ))
        metrics = {
            "objective": float(objective.detach()),
            "predicted_e_hull_eV_atom": float(prediction.detach()),
            "force_max_eV_A": float(force_max.detach()),
            "stress_fro_eV_A3": float(stress_fro.detach()),
            "stress_fro_GPa": float(stress_fro.detach()) * 160.21766208,
        }
        return outputs, variable, metrics

    def values(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if self.cache_values_x is not None and np.array_equal(vector, self.cache_values_x):
            assert self.cache_values is not None
            return self.cache_values
        outputs, _, metrics = self._forward(vector)
        values = outputs.detach().double().cpu().numpy()
        self.cache_values_x = vector.copy()
        self.cache_values = values
        self.last_metrics = metrics
        self.evaluations += 1
        return values

    def jacobian(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if self.cache_jacobian_x is not None and np.array_equal(vector, self.cache_jacobian_x):
            assert self.cache_jacobian is not None
            return self.cache_jacobian
        outputs, variable, metrics = self._forward(vector)
        rows = []
        for index, value in enumerate(outputs):
            rows.append(torch.autograd.grad(
                value, variable, retain_graph=index + 1 < outputs.numel()
            )[0])
        jacobian = torch.stack(rows).detach().double().cpu().numpy()
        self.cache_jacobian_x = vector.copy()
        self.cache_jacobian = jacobian
        self.last_metrics = metrics
        self.evaluations += 1
        self.jacobian_evaluations += 1
        return jacobian


class _JacobianCallback(ca.Callback):
    def __init__(self, name: str, problem: TorchPropertyProblem) -> None:
        self.problem = problem
        super().__init__()
        self.construct(name)

    def get_n_in(self): return 2
    def get_n_out(self): return 1
    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(
            self.problem.n_variables if index == 0 else self.problem.n_outputs, 1
        )
    def get_sparsity_out(self, index):
        return ca.Sparsity.dense(self.problem.n_outputs, self.problem.n_variables)
    def eval(self, arguments):
        return [ca.DM(self.problem.jacobian(np.asarray(arguments[0]).reshape(-1)))]


class _ValueCallback(ca.Callback):
    def __init__(self, name: str, problem: TorchPropertyProblem) -> None:
        self.problem = problem
        self.jacobian_callback = None
        super().__init__()
        self.construct(name)

    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(self.problem.n_variables, 1)
    def get_sparsity_out(self, index):
        return ca.Sparsity.dense(self.problem.n_outputs, 1)
    def eval(self, arguments):
        return [ca.DM(self.problem.values(np.asarray(arguments[0]).reshape(-1)))]
    def has_jacobian(self): return True
    def get_jacobian(self, name, inames, onames, opts):
        self.jacobian_callback = _JacobianCallback(name, self.problem)
        return self.jacobian_callback


def _input_bounds(
    frac: np.ndarray, lattice_params: np.ndarray, config: DirectIPOPTConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.concatenate((frac.reshape(-1), lattice_params))
    lower = x0.copy()
    upper = x0.copy()
    lower[: frac.size] -= config.fractional_trust_radius
    upper[: frac.size] += config.fractional_trust_radius
    # Remove the translational null mode without restricting internal relaxation.
    lower[:3] = x0[:3]
    upper[:3] = x0[:3]
    scale = np.maximum(np.abs(lattice_params), 1.0)
    lower[frac.size :] -= config.lattice_relative_trust_radius * scale
    upper[frac.size :] += config.lattice_relative_trust_radius * scale
    # Positive diagonal bounds keep the canonical cell right handed.
    for index in (0, 2, 5):
        absolute = frac.size + index
        lower[absolute] = max(lower[absolute], 1.0e-3)
    return lower, upper


def optimize_cif(
    input_cif: str | Path, output_cif: str | Path, report_json: str | Path,
    *, surrogate_checkpoint: str | Path, uma_checkpoint: str | Path,
    device: str = "cuda",
    config: DirectIPOPTConfig = DirectIPOPTConfig(),
) -> dict[str, Any]:
    started = time.monotonic()
    structure = Structure.from_file(str(input_cif))
    elements = [str(site.specie) for site in structure]
    initial_lattice = canonical_lattice(structure.lattice.matrix)
    initial_frac = np.asarray(structure.frac_coords, dtype=np.float64)
    initial_static = evaluate_feasibility(
        elements, initial_frac, initial_lattice, delta_e_hull_eV_atom=None
    )
    required = [
        "element_support_exact", "atom_count_range", "finite_structure",
        "positive_lattice_determinant", "volume_per_atom_range",
        "composition_domain", "fe_average_valence", "specific_capacity",
        "minimum_pbc_distances", "p_coordination_eq_4",
        "fe_coordination_in_4_5_6",
    ]
    failed = [name for name in required if not initial_static["checks"][name]]
    if failed:
        raise ValueError(f"initial structure fails required checks: {failed}")
    topology = build_fixed_topology(elements, initial_frac, initial_lattice)
    property_problem = TorchPropertyProblem(
        elements=elements, initial_frac=initial_frac,
        initial_lattice=initial_lattice,
        surrogate_checkpoint=str(surrogate_checkpoint),
        uma_checkpoint=str(uma_checkpoint),
        device=device, config=config,
    )
    n_atoms = len(elements)
    params0 = lattice_parameters(initial_lattice)
    x0 = np.concatenate((initial_frac.reshape(-1), params0))
    lower, upper = _input_bounds(initial_frac, params0, config)
    symbol = ca.MX.sym("x", property_problem.n_variables)
    callback = _ValueCallback("direct_uma_properties", property_problem)
    properties = callback(symbol)
    geometry, geometry_names = geometry_constraints(symbol, topology, n_atoms, config)
    geometry_function = ca.Function("direct_geometry", [symbol], [geometry])
    constraints = ca.vertcat(properties[1:], geometry)
    constraint_names = [
        "surrogate_hull_upper", "surrogate_hull_nonnegative",
        *[f"force:{index}" for index in range(n_atoms)],
        "stress", *geometry_names,
    ]
    solver = ca.nlpsol("direct_ipopt", "ipopt", {
        "x": symbol, "f": properties[0], "g": constraints,
    }, {
        "print_time": True,
        "ipopt.print_level": config.print_level,
        "ipopt.sb": "yes",
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.max_iter": config.max_iterations,
        "ipopt.max_wall_time": config.max_wall_time_seconds,
        "ipopt.tol": config.ipopt_tolerance,
        "ipopt.acceptable_tol": 1.0e-3,
        "ipopt.acceptable_iter": 3,
        "ipopt.fixed_variable_treatment": "make_parameter",
    })
    initial_values = property_problem.values(x0)
    initial_metrics = dict(property_problem.last_metrics)
    solution = solver(
        x0=x0, lbx=lower, ubx=upper,
        lbg=np.full(len(constraint_names), -np.inf),
        ubg=np.zeros(len(constraint_names)),
    )
    raw_optimized = np.asarray(solution["x"]).reshape(-1)
    raw_values = property_problem.values(raw_optimized)
    raw_metrics = dict(property_problem.last_metrics)

    # IPOPT may return a useful but infeasible point after an iteration/time
    # limit. Never emit such a point if it changes coordination or violates a
    # polyhedron constraint: retain the furthest feasible point on the segment
    # from the initial structure to the returned coupled-NLP candidate.
    def geometry_values(vector: np.ndarray) -> np.ndarray:
        return np.asarray(geometry_function(vector)).reshape(-1)

    def output_geometry_feasible(vector: np.ndarray) -> bool:
        if geometry_values(vector).max(initial=-np.inf) > 1.0e-7:
            return False
        candidate_frac = vector[: 3 * n_atoms].reshape(n_atoms, 3)
        candidate_lattice = lattice_from_parameters_numpy(vector[3 * n_atoms:])
        exact = evaluate_feasibility(
            elements, candidate_frac % 1.0, candidate_lattice,
            delta_e_hull_eV_atom=None,
        )
        if not all(exact["checks"][name] for name in required):
            return False
        topology_checks = fixed_topology_checks(
            elements, candidate_frac, candidate_lattice, topology, config
        )
        return bool(
            topology_checks["coordination_preserved"]
            and topology_checks["polyhedron_center_passes"]
            and topology_checks["fe_adjacency_passes"]
            and topology_checks["shared_oxygen_limit_passes"]
        )

    raw_geometry = geometry_values(raw_optimized)
    geometry_safeguard_alpha = 1.0
    optimized = raw_optimized
    if not output_geometry_feasible(raw_optimized):
        direction = raw_optimized - x0
        feasible_alpha = 0.0
        infeasible_alpha = 1.0
        for alpha in np.linspace(1.0, 0.0, 201):
            if output_geometry_feasible(x0 + alpha * direction):
                feasible_alpha = float(alpha)
                infeasible_alpha = min(1.0, feasible_alpha + 0.005)
                break
        for _ in range(30):
            alpha = 0.5 * (feasible_alpha + infeasible_alpha)
            if output_geometry_feasible(x0 + alpha * direction):
                feasible_alpha = alpha
            else:
                infeasible_alpha = alpha
        geometry_safeguard_alpha = feasible_alpha
        optimized = x0 + feasible_alpha * direction

    final_values = property_problem.values(optimized)
    final_metrics = dict(property_problem.last_metrics)
    final_frac = optimized[: 3 * n_atoms].reshape(n_atoms, 3) % 1.0
    final_lattice = lattice_from_parameters_numpy(optimized[3 * n_atoms :])
    final_structure = Structure(Lattice(final_lattice), elements, final_frac)
    output_cif = Path(output_cif)
    report_json = Path(report_json)
    output_cif.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    final_structure.to(filename=str(output_cif))
    final_exact = evaluate_feasibility(
        elements, final_frac, final_lattice,
        delta_e_hull_eV_atom=final_metrics["predicted_e_hull_eV_atom"],
    )
    initial_topology_checks = fixed_topology_checks(
        elements, initial_frac, initial_lattice, topology, config
    )
    final_topology_checks = fixed_topology_checks(
        elements, optimized[: 3 * n_atoms].reshape(n_atoms, 3),
        final_lattice, topology, config,
    )
    final_geometry = geometry_values(optimized)
    constraint_vector = np.concatenate((final_values[1:], final_geometry))
    violations = np.maximum(constraint_vector, 0.0)
    worst = np.argsort(violations)[::-1][:10]
    report: dict[str, Any] = {
        "version": "nagen-direct-ipopt-v1",
        "solver": "casadi-ipopt",
        "status": solver.stats().get("return_status", "unknown"),
        "success": bool(solver.stats().get("success", False)),
        "input_cif": str(Path(input_cif).resolve()),
        "output_cif": str(output_cif.resolve()),
        "N": n_atoms,
        "formula": final_structure.composition.reduced_formula,
        "fixed_elements": elements,
        "config": asdict(config),
        "surrogate_checkpoint": str(Path(surrogate_checkpoint).resolve()),
        "uma_checkpoint": str(Path(uma_checkpoint).resolve()),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "raw_ipopt_metrics": raw_metrics,
        "geometry_safeguard_alpha": geometry_safeguard_alpha,
        "initial_property_constraint_vector": initial_values[1:].tolist(),
        "maximum_constraint_violation": float(violations.max(initial=0.0)),
        "violated_constraint_count": int(np.count_nonzero(violations > 1.0e-6)),
        "worst_constraints": [
            {"name": constraint_names[int(index)],
             "scaled_value": float(constraint_vector[index]),
             "violation": float(violations[index])}
            for index in worst if violations[index] > 0
        ],
        "initial_exact_checks": initial_static,
        "final_exact_checks": final_exact,
        "initial_priority_topology_checks": initial_topology_checks,
        "final_priority_topology_checks": final_topology_checks,
        "fixed_topology": {
            "coordination_counts": topology.coordination_counts,
            "maximum_shared_oxygen_count": topology.maximum_shared_oxygen_count,
            "pair_constraint_count": len(topology.pairs),
            "coordination_constraint_count": len(topology.coordination),
            "polyhedron_face_constraint_count": len(topology.faces),
            "fe_adjacency_constraint_count": len(topology.adjacencies),
        },
        "property_evaluations": property_problem.evaluations,
        "property_jacobian_evaluations": property_problem.jacobian_evaluations,
        "wall_time_seconds": time.monotonic() - started,
        "frac_coords": final_frac.tolist(),
        "lattice": final_lattice.tolist(),
    }
    report["all_exact_checks_feasible"] = bool(final_exact["feasible"])
    report["priority_topology_constraints_feasible"] = bool(
        final_topology_checks["coordination_preserved"]
        and final_topology_checks["polyhedron_center_passes"]
        and final_topology_checks["fe_adjacency_passes"]
        and final_topology_checks["shared_oxygen_limit_passes"]
    )
    report["mechanical_constraints_feasible"] = bool(
        final_metrics["force_max_eV_A"] <= config.force_tolerance_eV_A
        and final_metrics["stress_fro_GPa"] <= config.stress_tolerance_GPa
    )
    report["all_ipopt_constraints_feasible"] = bool(violations.max(initial=0.0) <= 1.0e-4)
    report["all_constraints_feasible"] = bool(
        report["all_exact_checks_feasible"]
        and report["priority_topology_constraints_feasible"]
        and report["mechanical_constraints_feasible"]
        and report["all_ipopt_constraints_feasible"]
    )
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report

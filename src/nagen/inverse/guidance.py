"""Differentiable hard-constraint guidance in D-Flow source space."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .constraints import minimum_distance
from .lattice import decode_lattice, unstandardize_lattice
from .model import CrystalState, CrystalVectorField
from .sample import integrate_flow
from .spec import DEFAULT_SPEC


@dataclass(frozen=True)
class GuidanceWeights:
    distance: float = 20.0
    p_coordination: float = 3.0
    fe_coordination: float = 3.0
    source_prior: float = 0.02


@dataclass(frozen=True)
class TerminalWeights:
    """Weights for single-shooting optimization of the generated endpoint."""

    distance: float = 30.0
    p_coordination: float = 12.0
    fe_coordination: float = 12.0
    p_geometry: float = 4.0
    fe_geometry: float = 4.0
    displacement: float = 0.01
    lattice_displacement: float = 0.02


def _constant_tables(device: torch.device, dtype: torch.dtype):
    spec = DEFAULT_SPEC
    elements = spec.elements
    minima = torch.tensor(
        [[minimum_distance(a, b, spec) for b in elements] for a in elements],
        device=device,
        dtype=dtype,
    )
    return minima


def soft_constraint_terms(
    model: CrystalVectorField,
    terminal: CrystalState,
    mask: torch.Tensor,
    temperature: float = 0.35,
    bond_temperature_A: float = 0.10,
    probabilities_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Smooth penalties corresponding to the exact post-generation checks."""
    dtype = terminal.atom.dtype
    if probabilities_override is None:
        probabilities = torch.softmax(terminal.atom / temperature, dim=-1)
        probabilities = probabilities * mask.unsqueeze(-1)
    else:
        probabilities = probabilities_override * mask.unsqueeze(-1)
    n_atoms = mask.sum(dim=1).to(dtype)
    physical_lattice = unstandardize_lattice(
        terminal.lattice, model.lattice_mean, model.lattice_std
    )
    lattice = decode_lattice(physical_lattice, mask.sum(dim=1))
    delta = terminal.frac[:, :, None, :] - terminal.frac[:, None, :, :]
    delta = delta - torch.floor(delta + 0.5)
    cart = torch.einsum("bijd,bdk->bijk", delta, lattice)
    distances = torch.linalg.vector_norm(cart, dim=-1).clamp_min(1e-6)
    valid_pairs = mask[:, :, None] & mask[:, None, :]
    upper = torch.triu(
        torch.ones(mask.shape[1], mask.shape[1], dtype=torch.bool, device=mask.device),
        diagonal=1,
    )
    upper_pairs = valid_pairs & upper

    minima = _constant_tables(mask.device, dtype)
    expected_minimum = torch.einsum(
        "biv,vw,bjw->bij", probabilities, minima, probabilities
    )
    distance_violation = F.relu(expected_minimum - distances)
    per_structure_distance = (
        distance_violation.square().masked_fill(~upper_pairs, 0.0).sum(dim=(1, 2))
        / n_atoms.clamp_min(1.0)
    )
    maximum_violation = distance_violation.masked_fill(~upper_pairs, 0.0).amax(
        dim=(1, 2)
    )
    distance_loss = (per_structure_distance + maximum_violation.square()).mean()

    element_to_offset = {element: i for i, element in enumerate(DEFAULT_SPEC.elements)}
    p_probability = probabilities[..., element_to_offset["P"]]
    o_probability = probabilities[..., element_to_offset["O"]]
    fe_probability = probabilities[..., element_to_offset["Fe"]]
    po_bonds = torch.sigmoid(
        (DEFAULT_SPEC.p_o_bond_cutoff - distances) / bond_temperature_A
    ) * o_probability[:, None, :]
    fe_o_bonds = torch.sigmoid(
        (DEFAULT_SPEC.fe_o_bond_cutoff - distances) / bond_temperature_A
    ) * o_probability[:, None, :]
    diagonal = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    po_bonds = po_bonds.masked_fill(diagonal | ~valid_pairs, 0.0)
    fe_o_bonds = fe_o_bonds.masked_fill(diagonal | ~valid_pairs, 0.0)
    p_coordination = po_bonds.sum(dim=-1)
    fe_coordination = fe_o_bonds.sum(dim=-1)
    p_coordination_loss = (
        p_probability * (p_coordination - 4.0).square()
    ).sum() / p_probability.sum().clamp_min(1.0)
    fe_interval_violation = F.relu(4.0 - fe_coordination).square() + F.relu(
        fe_coordination - 6.0
    ).square()
    fe_coordination_loss = (
        fe_probability * fe_interval_violation
    ).sum() / fe_probability.sum().clamp_min(1.0)
    volume = torch.linalg.det(lattice)
    volume_per_atom = volume / n_atoms.clamp_min(1.0)
    volume_loss = (
        F.relu(DEFAULT_SPEC.volume_per_atom_min_A3 - volume_per_atom).square()
        + F.relu(volume_per_atom - DEFAULT_SPEC.volume_per_atom_max_A3).square()
    ).mean()
    return {
        "distance": distance_loss,
        "p_coordination": p_coordination_loss,
        "fe_coordination": fe_coordination_loss,
        "volume": volume_loss,
    }


def _source_prior(source: CrystalState, mask: torch.Tensor) -> torch.Tensor:
    atom_values = source.atom[mask]
    frac_values = source.frac[mask]
    # Uniform fractional coordinates have mean 0.5 and variance 1/12.
    atom_prior = atom_values.mean().square() + (atom_values.std() - 1.0).square()
    frac_prior = (frac_values.mean() - 0.5).square() + (
        frac_values.std() - (1.0 / 12.0) ** 0.5
    ).square()
    lattice_prior = source.lattice.mean().square() + (
        source.lattice.std() - 1.0
    ).square()
    return atom_prior + frac_prior + lattice_prior


def _periodic_distances_27(
    frac: torch.Tensor, lattice: torch.Tensor
) -> torch.Tensor:
    """Differentiable counterpart of the exact 27-image verifier."""
    delta = frac[:, :, None, :] - frac[:, None, :, :]
    delta = delta - torch.floor(delta + 0.5)
    translations = torch.tensor(
        [
            (i, j, k)
            for i in (-1.0, 0.0, 1.0)
            for j in (-1.0, 0.0, 1.0)
            for k in (-1.0, 0.0, 1.0)
        ],
        dtype=frac.dtype,
        device=frac.device,
    )
    images = delta.unsqueeze(-2) + translations
    cart = torch.einsum("bijtd,bdk->bijtk", images, lattice)
    return torch.linalg.vector_norm(cart, dim=-1).amin(dim=-1).clamp_min(1e-7)


def _coordination_assignment(
    type_index: torch.Tensor,
    mask: torch.Tensor,
    distances: torch.Tensor,
) -> torch.Tensor:
    """Build a fixed, capacity-aware center--oxygen assignment.

    The assignment is an inference-only auxiliary variable derived from
    ``A, X, L``; it is not an additional generated design variable.  Each P
    receives four distinct oxygens.  Each TM receives the locally least-cost
    count in {4, 5, 6}.  Oxygen sharing up to degree two is preferred, while a
    deterministic fallback keeps every center assigned when a sampled
    composition does not have enough degree-two oxygen slots.
    """
    offsets = {element: i for i, element in enumerate(DEFAULT_SPEC.elements)}
    detached = distances.detach().cpu()
    types_cpu = type_index.detach().cpu()
    mask_cpu = mask.detach().cpu()
    assignment = torch.zeros_like(distances, dtype=torch.bool, device="cpu")
    for batch_index in range(mask.shape[0]):
        oxygens = [
            index
            for index in range(mask.shape[1])
            if bool(mask_cpu[batch_index, index])
            and int(types_cpu[batch_index, index]) == offsets["O"]
        ]
        centers: list[tuple[int, int]] = []
        for center in range(mask.shape[1]):
            if not bool(mask_cpu[batch_index, center]):
                continue
            element_index = int(types_cpu[batch_index, center])
            if element_index == offsets["P"]:
                centers.append((center, 4))
                continue
            element = DEFAULT_SPEC.elements[element_index]
            if element != "Fe":
                continue
            oxygen_distances = sorted(
                float(detached[batch_index, center, oxygen]) for oxygen in oxygens
            )
            cutoff = DEFAULT_SPEC.fe_o_bond_cutoff
            # Select the allowed coordination count requiring the smallest
            # endpoint correction.  A tiny tie-break favours lower CN and
            # therefore avoids inventing unnecessary bonds.
            costs: list[tuple[float, int]] = []
            for count in (4, 5, 6):
                if len(oxygen_distances) < count:
                    continue
                included = sum(
                    max(0.0, value - cutoff) ** 2
                    for value in oxygen_distances[:count]
                )
                excluded = (
                    max(0.0, cutoff - oxygen_distances[count]) ** 2
                    if len(oxygen_distances) > count
                    else 0.0
                )
                costs.append((included + excluded + 1.0e-6 * count, count))
            if costs:
                centers.append((center, min(costs)[1]))

        oxygen_degree = {oxygen: 0 for oxygen in oxygens}
        selected = {center: set() for center, _ in centers}
        # Round-robin allocation prevents an early center from consuming all
        # low-degree oxygen slots before other centers receive any neighbours.
        maximum_count = max((count for _, count in centers), default=0)
        for slot in range(maximum_count):
            active = [(center, count) for center, count in centers if slot < count]
            active.sort(
                key=lambda item: min(
                    (
                        float(detached[batch_index, item[0], oxygen])
                        for oxygen in oxygens
                        if oxygen not in selected[item[0]]
                    ),
                    default=1.0e6,
                )
            )
            for center, _ in active:
                candidates = [
                    oxygen
                    for oxygen in oxygens
                    if oxygen not in selected[center] and oxygen_degree[oxygen] < 2
                ]
                if not candidates:
                    candidates = [
                        oxygen for oxygen in oxygens if oxygen not in selected[center]
                    ]
                if not candidates:
                    continue
                oxygen = min(
                    candidates,
                    key=lambda index: (
                        float(detached[batch_index, center, index])
                        + 0.35 * oxygen_degree[index],
                        oxygen_degree[index],
                        index,
                    ),
                )
                assignment[batch_index, center, oxygen] = True
                selected[center].add(oxygen)
                oxygen_degree[oxygen] += 1
    return assignment.to(distances.device)


def _assigned_coordination_losses(
    distances: torch.Tensor,
    assignment: torch.Tensor,
    phosphorus: torch.Tensor,
    transition_metal: torch.Tensor,
    oxygen: torch.Tensor,
    margin_A: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable bond and polyhedron losses for a fixed assignment."""
    zero = distances.sum() * 0.0
    p_coordination = zero
    tm_coordination = zero
    p_geometry = zero
    tm_geometry = zero
    p_sites = 0
    tm_sites = 0
    for batch_index in range(distances.shape[0]):
        for center in range(distances.shape[1]):
            is_p = bool(phosphorus[batch_index, center])
            is_tm = bool(transition_metal[batch_index, center])
            if not (is_p or is_tm):
                continue
            neighbours = torch.nonzero(
                assignment[batch_index, center], as_tuple=False
            ).flatten()
            count = int(neighbours.numel())
            if count == 0:
                continue
            included = distances[batch_index, center, neighbours]
            lower = (
                DEFAULT_SPEC.pair_minima["P-O"]
                if is_p
                else DEFAULT_SPEC.pair_minima["Fe-O"]
            )
            cutoff = (
                DEFAULT_SPEC.p_o_bond_cutoff
                if is_p
                else DEFAULT_SPEC.fe_o_bond_cutoff
            )
            site_coordination = F.relu(lower + margin_A - included).square().sum()
            site_coordination = site_coordination + F.relu(
                included - (cutoff - margin_A)
            ).square().sum()
            unassigned_oxygen = oxygen[batch_index] & ~assignment[batch_index, center]
            if bool(unassigned_oxygen.any()):
                nearest_excluded = distances[
                    batch_index, center, unassigned_oxygen
                ].amin()
                site_coordination = site_coordination + F.relu(
                    cutoff + margin_A - nearest_excluded
                ).square()

            neighbour_pairs = distances[
                batch_index,
                neighbours[:, None],
                neighbours[None, :],
            ]
            pair_upper = torch.triu(
                torch.ones(
                    count,
                    count,
                    dtype=torch.bool,
                    device=distances.device,
                ),
                diagonal=1,
            )
            pair_values = neighbour_pairs[pair_upper].sort().values
            bond_target = 1.55 if is_p else 2.00
            site_geometry = 0.25 * (included - bond_target).square().sum()
            if count == 4:
                target = torch.full_like(
                    pair_values, bond_target * (8.0 / 3.0) ** 0.5
                )
            elif count == 5:
                target = torch.tensor(
                    [bond_target * 2.0**0.5] * 6
                    + [bond_target * 3.0**0.5] * 3
                    + [2.0 * bond_target],
                    dtype=distances.dtype,
                    device=distances.device,
                )
            elif count == 6:
                target = torch.tensor(
                    [bond_target * 2.0**0.5] * 12 + [2.0 * bond_target] * 3,
                    dtype=distances.dtype,
                    device=distances.device,
                )
            else:
                target = pair_values.detach()
            if pair_values.numel():
                site_geometry = site_geometry + (pair_values - target).square().mean()
            if is_p:
                p_coordination = p_coordination + site_coordination
                p_geometry = p_geometry + site_geometry
                p_sites += 1
            else:
                tm_coordination = tm_coordination + site_coordination
                tm_geometry = tm_geometry + site_geometry
                tm_sites += 1
    return (
        p_coordination / max(1, p_sites),
        tm_coordination / max(1, tm_sites),
        p_geometry / max(1, p_sites),
        tm_geometry / max(1, tm_sites),
    )


def polyhedron_constraint_terms(
    model: CrystalVectorField,
    terminal: CrystalState,
    mask: torch.Tensor,
    bond_temperature_A: float = 0.10,
    coplanar_temperature: float = 0.10,
    theta_min_deg: float = 30.0,
) -> dict[str, torch.Tensor]:
    """Differentiable P/Fe--O center and Fe-polyhedron adjacency penalties.

    Neighbor identities are selected from detached distances (an auxiliary
    discrete assignment); all losses are evaluated on differentiable vectors.
    This avoids pretending that top-k/neighbor selection itself has a gradient.
    """
    dtype = terminal.frac.dtype
    device = terminal.frac.device
    physical_lattice = unstandardize_lattice(
        terminal.lattice, model.lattice_mean, model.lattice_std
    )
    lattice = decode_lattice(physical_lattice, mask.sum(dim=1))
    frac = terminal.frac
    delta = frac[:, :, None, :] - frac[:, None, :, :]
    translations = torch.tensor(
        [(i, j, k) for i in (-1.0, 0.0, 1.0)
         for j in (-1.0, 0.0, 1.0) for k in (-1.0, 0.0, 1.0)],
        dtype=dtype, device=device,
    )
    images = delta.unsqueeze(-2) + translations
    vectors = torch.einsum("bijtd,bdk->bijtk", images, lattice)
    distances = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(1.0e-7)
    nearest = distances.detach().argmin(dim=-1)
    vectors = vectors.gather(
        -2, nearest[..., None, None].expand(*nearest.shape, 1, 3)
    ).squeeze(-2)
    distances = distances.gather(-1, nearest[..., None]).squeeze(-1)

    type_index = terminal.atom.argmax(dim=-1)
    offsets = {element: i for i, element in enumerate(DEFAULT_SPEC.elements)}
    oxygen = (type_index == offsets["O"]) & mask
    phosphorus = (type_index == offsets["P"]) & mask
    iron = (type_index == offsets["Fe"]) & mask
    center_loss = frac.new_zeros(())
    face_loss = frac.new_zeros(())
    coplanar_loss = frac.new_zeros(())
    center_count = 0
    pair_count = 0
    cos_limit = math.cos(math.radians(theta_min_deg))
    for b in range(frac.shape[0]):
        oxygen_indices = torch.where(oxygen[b])[0]
        if oxygen_indices.numel() == 0:
            continue
        centers = torch.where((phosphorus[b] | iron[b]))[0]
        memberships: dict[int, torch.Tensor] = {}
        vectors_by_center: dict[int, torch.Tensor] = {}
        for center in centers.tolist():
            cutoff = (DEFAULT_SPEC.p_o_bond_cutoff
                      if bool(phosphorus[b, center])
                      else DEFAULT_SPEC.fe_o_bond_cutoff)
            d = distances[b, center, oxygen_indices]
            weights = torch.sigmoid((cutoff - d) / bond_temperature_A)
            weights = weights / weights.sum().clamp_min(1.0e-8)
            displacement = vectors[b, center, oxygen_indices]
            center_loss = center_loss + (
                (weights[:, None] * displacement).sum(dim=0).square().sum()
            )
            center_count += 1
            memberships[center] = weights
            vectors_by_center[center] = displacement
        if centers.numel() < 2:
            continue
        fe_centers = [int(x) for x in centers.tolist() if bool(iron[b, x])]
        for index, first in enumerate(fe_centers):
            for second in fe_centers[index + 1:]:
                w_first = memberships[first]
                w_second = memberships[second]
                shared = w_first * w_second
                shared_count = shared.sum()
                face_loss = face_loss + F.softplus(
                    (shared_count - 2.0) / bond_temperature_A
                ).square()
                pair_count += 1
                if oxygen_indices.numel() < 2:
                    continue
                top = shared.detach().topk(2).indices
                u = vectors_by_center[first][top]
                v = vectors_by_center[second][top]
                normal_first = torch.linalg.cross(u[0], u[1])
                normal_second = torch.linalg.cross(v[0], v[1])
                denom = (
                    torch.linalg.vector_norm(normal_first)
                    * torch.linalg.vector_norm(normal_second)
                ).clamp_min(1.0e-8)
                cosine = (normal_first * normal_second).sum().abs() / denom
                coplanar_loss = coplanar_loss + F.softplus(
                    (cosine - cos_limit) / coplanar_temperature
                ).square()
    return {
        "poly_center": center_loss / max(1, center_count),
        "poly_face": face_loss / max(1, pair_count),
        "poly_coplanar": coplanar_loss / max(1, pair_count),
    }


def terminal_constraint_terms(
    model: CrystalVectorField,
    terminal: CrystalState,
    mask: torch.Tensor,
    margin_A: float = 0.035,
    coordination_assignment: torch.Tensor | None = None,
    fe_coordination_prior: tuple[float, float, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Piecewise-smooth penalties closely matching the discrete hard checks.

    Atom identities are fixed. P is driven to tetrahedral O4 and each
    transition metal to octahedral O6 (one allowed member of {4, 5, 6}). The
    next-nearest oxygen is explicitly pushed outside the bonding cutoff,
    avoiding the ambiguity of a soft coordination-number sum.
    """
    dtype = terminal.frac.dtype
    type_index = terminal.atom.argmax(dim=-1)
    physical_lattice = unstandardize_lattice(
        terminal.lattice, model.lattice_mean, model.lattice_std
    )
    lattice = decode_lattice(physical_lattice, mask.sum(dim=1))
    distances = _periodic_distances_27(terminal.frac, lattice)
    valid_pairs = mask[:, :, None] & mask[:, None, :]
    upper = torch.triu(
        torch.ones(mask.shape[1], mask.shape[1], dtype=torch.bool, device=mask.device),
        diagonal=1,
    )
    pair_mask = valid_pairs & upper

    minima = _constant_tables(mask.device, dtype)
    pair_minima = minima[type_index[:, :, None], type_index[:, None, :]]
    violations = F.relu(pair_minima + margin_A - distances)
    valid_violations = violations.masked_fill(~pair_mask, 0.0)
    n_atoms = mask.sum(dim=1).to(dtype).clamp_min(1.0)
    distance_per_structure = valid_violations.square().sum(dim=(1, 2)) / n_atoms
    distance_max = valid_violations.amax(dim=(1, 2)).square()
    # A global pair average is dominated by the many already-safe long-range
    # pairs in large cells. Keep the maximum and add a worst-decile term so a
    # cluster of collisions cannot be diluted by system size.
    pair_values = valid_violations[pair_mask].square()
    worst_count = max(1, math.ceil(0.10 * pair_values.numel()))
    distance_cvar = pair_values.topk(worst_count).values.mean()
    distance_loss = (distance_per_structure + distance_max).mean() + distance_cvar

    offsets = {element: i for i, element in enumerate(DEFAULT_SPEC.elements)}
    oxygen = (type_index == offsets["O"]) & mask
    phosphorus = (type_index == offsets["P"]) & mask
    iron = (type_index == offsets["Fe"]) & mask

    if coordination_assignment is not None:
        p_loss, tm_loss, p_geometry, tm_geometry = _assigned_coordination_losses(
            distances,
            coordination_assignment,
            phosphorus,
            iron,
            oxygen,
            margin_A,
        )
    else:
        oxygen_distances = distances.masked_fill(~oxygen[:, None, :], 1.0e3)
        oxygen_distances = oxygen_distances.masked_fill(
            torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None],
            1.0e3,
        )
        sorted_oxygen = oxygen_distances.sort(dim=-1).values

        def exact_count_site_loss(
            center_mask: torch.Tensor,
            count: int,
            lower: float,
            upper_cutoff: float,
        ) -> torch.Tensor:
            included_distances = sorted_oxygen[..., :count]
            excluded_distance = sorted_oxygen[..., count]
            included = F.relu(lower + margin_A - included_distances).square()
            included = included + F.relu(
                included_distances - (upper_cutoff - margin_A)
            ).square()
            excluded = F.relu(
                upper_cutoff + margin_A - excluded_distance
            ).square()
            return included.sum(dim=-1) + excluded

        def worst_site_loss(
            site_loss: torch.Tensor, center_mask: torch.Tensor
        ) -> torch.Tensor:
            values = site_loss[center_mask]
            if values.numel() == 0:
                return site_loss.sum() * 0.0
            worst_count = max(1, math.ceil(0.25 * values.numel()))
            return values.mean() + values.topk(worst_count).values.mean()

        def polyhedron_geometry_loss(
            center_mask: torch.Tensor,
            count: int,
            bond_target: float,
            pair_targets: torch.Tensor,
        ) -> torch.Tensor:
            neighbor_indices = oxygen_distances.argsort(dim=-1)[..., :count]
            batch_index = torch.arange(mask.shape[0], device=mask.device)[
                :, None, None, None
            ]
            first = neighbor_indices[..., :, None]
            second = neighbor_indices[..., None, :]
            neighbor_pair_distances = distances[batch_index, first, second]
            pair_upper = torch.triu(
                torch.ones(count, count, dtype=torch.bool, device=mask.device),
                diagonal=1,
            )
            pair_values = neighbor_pair_distances[..., pair_upper]
            pair_values = pair_values.sort(dim=-1).values
            bond_values = sorted_oxygen[..., :count]
            site_loss = 0.25 * (bond_values - bond_target).square().sum(dim=-1)
            site_loss = site_loss + (pair_values - pair_targets).square().mean(dim=-1)
            return (site_loss * center_mask).sum() / center_mask.sum().clamp_min(1)

        p_site_loss = exact_count_site_loss(
            phosphorus,
            4,
            DEFAULT_SPEC.pair_minima["P-O"],
            DEFAULT_SPEC.p_o_bond_cutoff,
        )
        p_loss = worst_site_loss(p_site_loss, phosphorus)
        # The exact specification permits Fe coordination 4, 5, or 6. Use the
        # locally easiest valid shell instead of forcing every Fe site to CN=6.
        fe_count_losses = torch.stack(
            [
                exact_count_site_loss(
                    iron,
                    count,
                    DEFAULT_SPEC.pair_minima["Fe-O"],
                    DEFAULT_SPEC.fe_o_bond_cutoff,
                )
                for count in (4, 5, 6)
            ],
            dim=0,
        )
        if fe_coordination_prior is None:
            fe_site_loss = fe_count_losses.amin(dim=0)
        else:
            prior = torch.tensor(
                fe_coordination_prior, dtype=dtype, device=mask.device
            )
            if bool((prior <= 0).any()) or not torch.isclose(
                prior.sum(),
                torch.ones((), dtype=dtype, device=mask.device),
                atol=1.0e-4,
            ):
                raise ValueError("Fe coordination prior must be positive and sum to one")
            temperature = 0.10
            fe_site_loss = -temperature * torch.logsumexp(
                prior.log()[:, None, None] - fe_count_losses / temperature,
                dim=0,
            )
        tm_loss = worst_site_loss(fe_site_loss, iron)
        p_pair_target = torch.full(
            (6,), 1.55 * (8.0 / 3.0) ** 0.5, dtype=dtype, device=mask.device
        )
        tm_pair_target = torch.tensor(
            [2.00 * 2.0**0.5] * 12 + [4.00] * 3,
            dtype=dtype,
            device=mask.device,
        )
        p_geometry = polyhedron_geometry_loss(
            phosphorus, 4, 1.55, p_pair_target
        )
        tm_geometry = polyhedron_geometry_loss(
            iron, 6, 2.00, tm_pair_target
        )
    polyhedron = polyhedron_constraint_terms(model, terminal, mask)
    return {
        "distance": distance_loss,
        "p_coordination": p_loss,
        "fe_coordination": tm_loss,
        "p_geometry": p_geometry,
        "fe_geometry": tm_geometry,
        "minimum_distance_A": distances.masked_fill(~pair_mask, 1.0e3).amin(),
        **polyhedron,
    }


def optimize_terminal(
    model: CrystalVectorField,
    terminal: CrystalState,
    mask: torch.Tensor,
    optimization_steps: int = 250,
    learning_rate: float = 0.025,
    weights: TerminalWeights = TerminalWeights(),
    fixed_coordination_assignment: bool = False,
) -> tuple[CrystalState, list[dict[str, float]]]:
    """ShootingFlow endpoint correction with fixed generated composition.

    This is a constrained inference step, not a second generator: N and A are
    unchanged and the DFlow endpoint remains the reference through weak
    displacement regularization.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference_frac = terminal.frac.detach().clone()
    reference_lattice = terminal.lattice.detach().clone()
    state = CrystalState(
        terminal.atom.detach(),
        terminal.frac.detach().clone().requires_grad_(True),
        terminal.lattice.detach().clone().requires_grad_(True),
    )
    coordination_assignment = None
    if fixed_coordination_assignment:
        with torch.no_grad():
            physical_lattice = unstandardize_lattice(
                state.lattice, model.lattice_mean, model.lattice_std
            )
            lattice = decode_lattice(physical_lattice, mask.sum(dim=1))
            initial_distances = _periodic_distances_27(state.frac, lattice)
            type_index = state.atom.argmax(dim=-1)
            coordination_assignment = _coordination_assignment(
                type_index, mask, initial_distances
            )
    optimized = [state.frac, state.lattice]
    optimizer = torch.optim.Adam(optimized, lr=learning_rate)
    history: list[dict[str, float]] = []
    for step in range(optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        terms = terminal_constraint_terms(
            model,
            state,
            mask,
            coordination_assignment=coordination_assignment,
        )
        circular_delta = state.frac - reference_frac
        circular_delta = circular_delta - torch.floor(circular_delta + 0.5)
        displacement = (
            circular_delta.square().sum(dim=-1) * mask
        ).sum() / mask.sum().clamp_min(1)
        lattice_displacement = (state.lattice - reference_lattice).square().mean()
        # Repulsion dominates early; coordination receives full weight after
        # the worst overlaps have separated.
        coordination_ramp = min(1.0, (step + 1) / max(1.0, 0.2 * optimization_steps))
        loss = (
            weights.distance * terms["distance"]
            + coordination_ramp
            * (
                weights.p_coordination * terms["p_coordination"]
                + weights.fe_coordination * terms["fe_coordination"]
                + weights.p_geometry * terms["p_geometry"]
                + weights.fe_geometry * terms["fe_geometry"]
            )
            + weights.displacement * displacement
            + weights.lattice_displacement * lattice_displacement
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimized, 20.0)
        optimizer.step()
        with torch.no_grad():
            state.frac.remainder_(1.0)
            state.lattice.clamp_(-6.0, 6.0)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == optimization_steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "coordination_ramp": coordination_ramp,
                    "displacement": float(displacement.detach()),
                    "lattice_displacement": float(lattice_displacement.detach()),
                    **{key: float(value.detach()) for key, value in terms.items()},
                }
            )
    return CrystalState(state.atom, state.frac.detach(), state.lattice.detach()), history


def optimize_source(
    model: CrystalVectorField,
    source: CrystalState,
    mask: torch.Tensor,
    guidance_steps: int = 40,
    learning_rate: float = 0.03,
    ode_steps: int = 12,
    integrator: str = "midpoint",
    weights: GuidanceWeights = GuidanceWeights(),
    freeze_atom: bool = False,
) -> tuple[CrystalState, list[dict[str, float]]]:
    """Single-shooting D-Flow optimization of x0 with frozen flow weights."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source = CrystalState(
        source.atom.detach().clone().requires_grad_(True),
        source.frac.detach().clone().requires_grad_(True),
        source.lattice.detach().clone().requires_grad_(True),
    )
    optimized_tensors = [source.frac, source.lattice]
    if not freeze_atom:
        optimized_tensors.insert(0, source.atom)
    optimizer = torch.optim.Adam(optimized_tensors, lr=learning_rate)
    history: list[dict[str, float]] = []
    for step in range(guidance_steps):
        optimizer.zero_grad(set_to_none=True)
        terminal = integrate_flow(
            model,
            source,
            mask,
            steps=ode_steps,
            method=integrator,
            freeze_atom=freeze_atom,
        )
        terms = soft_constraint_terms(model, terminal, mask)
        prior = _source_prior(source, mask)
        loss = (
            weights.distance * terms["distance"]
            + weights.p_coordination * terms["p_coordination"]
            + weights.fe_coordination * terms["fe_coordination"]
            + terms["volume"]
            + weights.source_prior * prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimized_tensors, 10.0)
        optimizer.step()
        with torch.no_grad():
            source.frac.remainder_(1.0)
            if not freeze_atom:
                source.atom.clamp_(-8.0, 8.0)
            source.lattice.clamp_(-5.0, 5.0)
        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "source_prior": float(prior.detach()),
                **{key: float(value.detach()) for key, value in terms.items()},
            }
        )
    with torch.no_grad():
        terminal = integrate_flow(
            model,
            source,
            mask,
            steps=ode_steps,
            method=integrator,
            freeze_atom=freeze_atom,
        )
    return terminal, history

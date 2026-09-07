"""Single-level all-UMA D-Flow optimization with L-BFGS.

The discrete variables ``N`` and ``A`` are fixed for one solve.  L-BFGS acts
on ``z_X`` and ``z_L``; every closure integrates the frozen conditional Flow
and evaluates one merit function at the generated endpoint.  There is no
endpoint relaxation or post-selection stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.nn import functional as F

from .guidance import soft_constraint_terms, terminal_constraint_terms
from .lattice import decode_lattice, unstandardize_lattice
from .model import CrystalState, CrystalVectorField
from .novelty import NoveltyIndex, crystal_descriptor
from .sample import decode_terminal, integrate_flow
from .spec import DEFAULT_SPEC
from .differentiable_uma_hull import DifferentiableUMAHull
from .relaxation_surrogate import soft_crystal_descriptor


@dataclass(frozen=True)
class HybridLBFGSConfig:
    max_iterations: int = 80
    history_size: int = 20
    learning_rate: float = 0.5
    ode_steps: int = 24
    hull_threshold_eV_atom: float = 0.150
    hull_weight: float = 1.0
    hull_violation_weight: float = 100.0
    distance_weight: float = 80.0
    p_coordination_weight: float = 24.0
    fe_coordination_weight: float = 12.0
    volume_weight: float = 10.0
    novelty_weight: float = 0.01
    source_prior_weight: float = 0.002
    composition_weight: float = 40.0
    atom_temperature: float = 0.35
    distance_margin_A: float = 0.02
    hull_feasibility_only: bool = False


@dataclass
class HybridLBFGSResult:
    source: CrystalState
    terminal: CrystalState
    history: list[dict[str, float]]
    selected_values: dict[str, float]
    status: str


def relaxation_metrics_from_energy_gradient(
    energy_eV_atom: torch.Tensor,
    frac: torch.Tensor,
    lattice: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover max force and Frobenius stress from d(E/N)/d(frac,cell)."""
    grad_frac, grad_lattice = torch.autograd.grad(
        energy_eV_atom, (frac, lattice), retain_graph=True, create_graph=False
    )
    atom_count = frac.shape[1]
    cell = lattice[0]
    inverse_transpose = torch.linalg.inv(cell).mT
    forces = -atom_count * grad_frac[0] @ inverse_transpose
    volume = torch.linalg.det(cell).abs()
    stress = atom_count / volume * cell.mT @ grad_lattice[0]
    return torch.linalg.vector_norm(forces, dim=-1).amax(), torch.linalg.matrix_norm(
        stress, ord="fro"
    )


class HybridLBFGSSolver:
    def __init__(
        self,
        model: CrystalVectorField,
        mask: torch.Tensor,
        energy_per_atom: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        uma_hull_reference_eV_atom: float,
        novelty_index: NoveltyIndex | None = None,
        feasibility_callback: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, float], bool
        ] | None = None,
    ) -> None:
        self.model = model
        self.mask = mask
        self.energy_per_atom = energy_per_atom
        self.reference = float(uma_hull_reference_eV_atom)
        self.novelty_index = novelty_index
        self.feasibility_callback = feasibility_callback

    def solve(
        self, source: CrystalState,
        config: HybridLBFGSConfig = HybridLBFGSConfig(),
    ) -> HybridLBFGSResult:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        fixed_atom = source.atom.detach().clone()
        initial_frac = source.frac.detach().clone()
        initial_lattice = source.lattice.detach().clone()
        z_frac = initial_frac.clone().requires_grad_(True)
        z_lattice = initial_lattice.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            (z_frac, z_lattice), lr=config.learning_rate,
            max_iter=config.max_iterations, history_size=config.history_size,
            line_search_fn="strong_wolfe", tolerance_grad=1.0e-7,
            tolerance_change=1.0e-9,
        )
        history: list[dict[str, float]] = []
        best_key = (1, float("inf"), float("inf"))
        best_source = (z_frac.detach().clone(), z_lattice.detach().clone())
        best_row: dict[str, float] = {}

        def closure() -> torch.Tensor:
            nonlocal best_key, best_source, best_row
            optimizer.zero_grad(set_to_none=True)
            generated = integrate_flow(
                self.model, CrystalState(fixed_atom, z_frac, z_lattice), self.mask,
                steps=config.ode_steps, method="midpoint", freeze_atom=True,
            )
            types, frac, lattice = decode_terminal(self.model, generated, self.mask)
            energy = self.energy_per_atom(frac, lattice).mean()
            raw_hull = energy - self.reference
            e_hull = raw_hull.clamp_min(0.0)
            constraints = terminal_constraint_terms(
                self.model, generated, self.mask,
                margin_A=config.distance_margin_A,
            )
            physical_lattice = unstandardize_lattice(
                generated.lattice, self.model.lattice_mean, self.model.lattice_std
            )
            generated_lattice = decode_lattice(
                physical_lattice, self.mask.sum(dim=1)
            )
            volume_per_atom = torch.linalg.det(generated_lattice) / self.mask.sum(
                dim=1
            ).to(generated_lattice)
            constraints["volume"] = (
                F.relu(DEFAULT_SPEC.volume_per_atom_min_A3 - volume_per_atom).square()
                + F.relu(volume_per_atom - DEFAULT_SPEC.volume_per_atom_max_A3).square()
            ).mean()
            if self.novelty_index is None:
                novelty = energy.new_zeros(())
            else:
                descriptor = crystal_descriptor(types, frac, lattice, self.mask)
                novelty = self.novelty_index.smooth_score(descriptor).mean()
            volume_penalty = constraints["volume"]
            source_drift = (
                (z_frac - initial_frac).square().mean()
                + (z_lattice - initial_lattice).square().mean()
            )
            hull_excess = F.relu(raw_hull - config.hull_threshold_eV_atom)
            merit = (
                config.hull_weight * raw_hull
                + config.hull_violation_weight * hull_excess.square()
                + config.distance_weight * constraints["distance"]
                + config.p_coordination_weight * constraints["p_coordination"]
                + config.fe_coordination_weight * constraints["fe_coordination"]
                + config.volume_weight * volume_penalty
                - config.novelty_weight * novelty
                + config.source_prior_weight * source_drift
            )
            fmax, stress_fro = relaxation_metrics_from_energy_gradient(
                energy, frac, lattice
            )
            merit.backward()
            if not bool(torch.isfinite(merit) and torch.isfinite(z_frac.grad).all()
                        and torch.isfinite(z_lattice.grad).all()):
                raise FloatingPointError("non-finite Hybrid L-BFGS value or gradient")
            row = {
                "closure": float(len(history)), "merit": float(merit.detach()),
                "E_UMA_eV_atom": float(energy.detach()),
                "raw_E_hull_eV_atom": float(raw_hull.detach()),
                "E_hull_eV_atom": float(e_hull.detach()),
                "Fmax_eV_A": float(fmax.detach()),
                "stress_fro_eV_A3": float(stress_fro.detach()),
                "novelty": float(novelty.detach()),
                **{f"penalty_{key}": float(value.detach()) for key, value in constraints.items()
                   if key in {"distance", "p_coordination", "fe_coordination", "volume"}},
                "gradient_norm_z_X": float(z_frac.grad.detach().norm()),
                "gradient_norm_z_L": float(z_lattice.grad.detach().norm()),
            }
            history.append(row)
            feasible_relaxation = row["Fmax_eV_A"] <= 0.01 and row[
                "stress_fro_eV_A3"
            ] <= 0.10 / 160.21766208
            feasible_exact = True
            if self.feasibility_callback is not None:
                feasible_exact = self.feasibility_callback(
                    types.detach(), frac.detach(), lattice.detach(),
                    row["E_hull_eV_atom"],
                )
            row["exact_constraints_feasible"] = float(feasible_exact)
            key = (
                0 if row["raw_E_hull_eV_atom"] <= config.hull_threshold_eV_atom
                and feasible_relaxation and feasible_exact else 1,
                max(row["raw_E_hull_eV_atom"] - config.hull_threshold_eV_atom, 0.0)
                + max(row["Fmax_eV_A"] - 0.01, 0.0),
                row["merit"],
            )
            if key < best_key:
                best_key = key
                best_source = (z_frac.detach().clone(), z_lattice.detach().clone())
                best_row = dict(row)
            return merit

        optimizer.step(closure)
        final_source = CrystalState(fixed_atom, best_source[0], best_source[1])
        with torch.no_grad():
            terminal = integrate_flow(
                self.model, final_source, self.mask, steps=config.ode_steps,
                method="midpoint", freeze_atom=True,
            )
        status = "constraint_candidate" if best_key[0] == 0 else "infeasible"
        return HybridLBFGSResult(final_source, terminal, history, best_row, status)


def _minimal_composition_terms(
    probabilities: torch.Tensor, mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Only the independent composition constraints in HybridOptimization."""
    counts = (probabilities * mask.unsqueeze(-1)).sum(dim=1)
    na, fe, phosphorus, oxygen = counts.unbind(dim=-1)
    p = phosphorus.clamp_min(0.25)
    n = fe / p
    m = na / p
    y = 4.0 - oxygen / p
    valence = (3.0 - 2.0 * y - m) / n.clamp_min(0.05)
    kappa = n * (DEFAULT_SPEC.fe_valence_max - valence)
    extracted = torch.minimum(m, kappa)
    mass = (
        m * DEFAULT_SPEC.atomic_masses["Na"]
        + n * DEFAULT_SPEC.atomic_masses["Fe"]
        + DEFAULT_SPEC.atomic_masses["P"]
        + (4.0 - y) * DEFAULT_SPEC.atomic_masses["O"]
    ).clamp_min(1.0)
    capacity = extracted * DEFAULT_SPEC.faraday_C_mol / (3.6 * mass)

    def violation(value: torch.Tensor) -> torch.Tensor:
        return F.softplus(value / 0.02).mul(0.02).square().mean()

    return {
        "support": violation(0.5 - counts),
        "y_domain": violation(-y) + violation(y - 0.5),
        "m_domain": violation(1.0 - m),
        "fe_valence": violation(2.0 - valence),
        "capacity": violation(DEFAULT_SPEC.capacity_min_mAh_g - capacity)
        / DEFAULT_SPEC.capacity_min_mAh_g**2,
    }


class UnconditionalHybridLBFGSSolver:
    """Optimize the complete initial noise of the frozen unconditional Flow."""

    def __init__(
        self, model: CrystalVectorField, mask: torch.Tensor,
        energy_factory: Callable[[torch.Tensor], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        hull: DifferentiableUMAHull, atomic_number_lookup: torch.Tensor,
        novelty_index: NoveltyIndex,
        feasibility_callback: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, float], bool
        ],
    ) -> None:
        self.model = model
        self.mask = mask
        self.energy_factory = energy_factory
        self.hull = hull
        self.atomic_number_lookup = atomic_number_lookup
        self.novelty_index = novelty_index
        self.feasibility_callback = feasibility_callback

    def solve(
        self, source: CrystalState,
        config: HybridLBFGSConfig = HybridLBFGSConfig(),
    ) -> HybridLBFGSResult:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        initial = tuple(value.detach().clone() for value in source)
        with torch.no_grad():
            reference_terminal = integrate_flow(
                self.model, CrystalState(*initial), self.mask,
                steps=config.ode_steps, method="midpoint", freeze_atom=False,
            )
            fixed_indices = reference_terminal.atom.argmax(-1)
            fixed_types = (fixed_indices + 1).masked_fill(~self.mask, 0)
            fixed_species = F.one_hot(
                fixed_indices, reference_terminal.atom.shape[-1]
            ).to(reference_terminal.atom) * self.mask.unsqueeze(-1)
            fixed_numbers = self.atomic_number_lookup.to(fixed_types.device)[
                fixed_indices[0, self.mask[0]]
            ]
        fixed_energy_per_atom = self.energy_factory(fixed_numbers)
        z_atom, z_frac, z_lattice = (
            value.clone().requires_grad_(True) for value in initial
        )
        optimizer = torch.optim.LBFGS(
            (z_atom, z_frac, z_lattice), lr=config.learning_rate,
            max_iter=config.max_iterations, history_size=config.history_size,
            line_search_fn="strong_wolfe", tolerance_grad=1.0e-7,
            tolerance_change=1.0e-9,
        )
        history: list[dict[str, float]] = []
        best_key = (1, float("inf"), float("inf"))
        best_source = tuple(value.detach().clone() for value in (z_atom, z_frac, z_lattice))
        best_row: dict[str, float] = {}

        def closure() -> torch.Tensor:
            nonlocal best_key, best_source, best_row
            optimizer.zero_grad(set_to_none=True)
            terminal = integrate_flow(
                self.model, CrystalState(z_atom, z_frac, z_lattice), self.mask,
                steps=config.ode_steps, method="midpoint", freeze_atom=False,
            )
            decoded_types, frac, lattice = decode_terminal(
                self.model, terminal, self.mask
            )
            probabilities = torch.softmax(
                terminal.atom / config.atom_temperature, dim=-1
            ) * self.mask.unsqueeze(-1)
            hard = F.one_hot(
                terminal.atom.argmax(dim=-1), num_classes=terminal.atom.shape[-1]
            ).to(probabilities) * self.mask.unsqueeze(-1)
            straight_through = hard.detach() + probabilities - probabilities.detach()
            if config.hull_feasibility_only:
                types = fixed_types
                energy = fixed_energy_per_atom(frac, lattice).mean()
                raw_hull = self.hull.raw_e_hull(
                    energy.reshape(1), fixed_species, self.mask
                ).mean()
            else:
                types = decoded_types
                numbers = self.atomic_number_lookup.to(types.device)[
                    (types[0, self.mask[0]] - 1).long()
                ]
                energy = self.energy_factory(numbers)(frac, lattice).mean()
                raw_hull = self.hull.raw_e_hull(
                    energy.reshape(1), straight_through, self.mask
                ).mean()
            e_hull = raw_hull.clamp_min(0.0)
            geometry = soft_constraint_terms(
                self.model, terminal, self.mask,
                temperature=config.atom_temperature,
                probabilities_override=straight_through,
            )
            composition = _minimal_composition_terms(straight_through, self.mask)
            novelty = self.novelty_index.smooth_score(
                soft_crystal_descriptor(straight_through, frac, lattice, self.mask)
            ).mean()
            drift = sum(
                (value - reference).square().mean()
                for value, reference in zip(
                    (z_atom, z_frac, z_lattice), initial, strict=True
                )
            )
            hull_excess = F.relu(raw_hull - config.hull_threshold_eV_atom)
            if config.hull_feasibility_only:
                merit = 0.5 * hull_excess.square()
            else:
                merit = (
                    config.hull_weight * raw_hull
                    + config.hull_violation_weight * hull_excess.square()
                    + config.distance_weight * geometry["distance"]
                    + config.p_coordination_weight * geometry["p_coordination"]
                    + config.fe_coordination_weight * geometry["fe_coordination"]
                    + config.volume_weight * geometry["volume"]
                    + config.composition_weight * sum(composition.values())
                    - config.novelty_weight * novelty
                    + config.source_prior_weight * drift
                )
            fmax, stress_fro = relaxation_metrics_from_energy_gradient(
                energy, frac, lattice
            )
            merit.backward()
            gradients = (z_atom.grad, z_frac.grad, z_lattice.grad)
            if not bool(torch.isfinite(merit) and all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            )):
                raise FloatingPointError("non-finite unconditional Hybrid gradient")
            exact = self.feasibility_callback(
                types.detach(), frac.detach(), lattice.detach(), float(e_hull.detach())
            )
            row = {
                "closure": float(len(history)), "merit": float(merit.detach()),
                "E_UMA_eV_atom": float(energy.detach()),
                "raw_E_hull_eV_atom": float(raw_hull.detach()),
                "E_hull_eV_atom": float(e_hull.detach()),
                "Fmax_eV_A": float(fmax.detach()),
                "stress_fro_eV_A3": float(stress_fro.detach()),
                "novelty": float(novelty.detach()),
                "gradient_norm_z_A": float(z_atom.grad.detach().norm()),
                "gradient_norm_z_X": float(z_frac.grad.detach().norm()),
                "gradient_norm_z_L": float(z_lattice.grad.detach().norm()),
                "exact_constraints_feasible": float(exact),
                "fixed_type_region_stable": float(torch.equal(
                    decoded_types[self.mask], fixed_types[self.mask]
                )),
                **{f"geometry_{key}": float(value.detach()) for key, value in geometry.items()},
                **{f"composition_{key}": float(value.detach()) for key, value in composition.items()},
            }
            history.append(row)
            relaxation_ok = row["Fmax_eV_A"] <= 0.01 and row[
                "stress_fro_eV_A3"
            ] <= 0.10 / 160.21766208
            full = (
                row["raw_E_hull_eV_atom"] <= config.hull_threshold_eV_atom
                if config.hull_feasibility_only else exact and relaxation_ok
            )
            key = (
                0 if full else 1,
                max(row["raw_E_hull_eV_atom"] - 0.150, 0.0)
                + (0.0 if config.hull_feasibility_only else
                   max(row["Fmax_eV_A"] - 0.01, 0.0)
                   + (0.0 if exact else 1.0)),
                row["merit"],
            )
            if key < best_key:
                best_key, best_row = key, dict(row)
                best_source = tuple(
                    value.detach().clone() for value in (z_atom, z_frac, z_lattice)
                )
            return merit

        optimizer.step(closure)
        optimized_source = CrystalState(*best_source)
        with torch.no_grad():
            terminal = integrate_flow(
                self.model, optimized_source, self.mask, steps=config.ode_steps,
                method="midpoint", freeze_atom=False,
            )
        status = "constraint_candidate" if best_key[0] == 0 else "infeasible"
        return HybridLBFGSResult(
            optimized_source, terminal, history, best_row, status
        )

"""Projected Adam with simultaneous augmented-Lagrangian constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.nn import functional as F

from .differentiable_uma_hull import DifferentiableUMAHull
from .hybrid_lbfgs import relaxation_barrier, relaxation_metrics_from_energy_gradient
from .model import CrystalState, CrystalVectorField
from .sample import decode_terminal, integrate_flow
from .uma_guidance import EnergyFactory


@dataclass(frozen=True)
class HybridAdamConfig:
    max_iterations: int = 1000
    ode_steps: int = 24
    learning_rate_atom: float = 1.0e-2
    learning_rate_frac: float = 5.0e-3
    learning_rate_lattice: float = 1.0e-3
    gradient_clip_norm: float = 10.0
    hull_threshold_eV_atom: float = 0.150
    type_region_temperature: float = 1.0e-3
    type_logit_margin: float = 1.0e-4
    augmented_lagrangian_rho: float = 10.0
    dual_learning_rate: float = 0.1
    multiplier_max: float = 1_000.0
    source_prior_weight: float = 2.0e-3
    consecutive_feasible_required: int = 10
    hull_feasibility_only: bool = False
    force_barrier_weight: float = 20.0
    stress_barrier_weight: float = 20.0
    force_tolerance_eV_A: float = 0.01
    stress_tolerance_eV_A3: float = 0.10 / 160.21766208


@dataclass
class HybridAdamResult:
    source: CrystalState
    terminal: CrystalState
    history: list[dict[str, Any]]
    selected_values: dict[str, Any]
    status: str
    solver_stats: dict[str, Any]


class HybridProjectedAdamSolver:
    def __init__(
        self, model: CrystalVectorField, mask: torch.Tensor,
        uma_energy_factory: EnergyFactory, hull: DifferentiableUMAHull,
        atomic_number_lookup: torch.Tensor,
        feasibility_callback: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, float], bool
        ],
    ) -> None:
        self.model = model
        self.mask = mask
        self.uma_energy_factory = uma_energy_factory
        self.hull = hull
        self.atomic_number_lookup = atomic_number_lookup
        self.feasibility_callback = feasibility_callback

    def solve(
        self, source: CrystalState,
        config: HybridAdamConfig = HybridAdamConfig(),
    ) -> HybridAdamResult:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        reference = tuple(value.detach().clone() for value in source)
        variables = tuple(value.detach().clone().requires_grad_(True) for value in source)
        z_atom, z_frac, z_lattice = variables
        with torch.no_grad():
            reference_terminal = integrate_flow(
                self.model, CrystalState(*reference), self.mask,
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
            competitor_mask = F.one_hot(
                fixed_indices, reference_terminal.atom.shape[-1]
            ).logical_not() & self.mask.unsqueeze(-1)
        energy_per_atom = self.uma_energy_factory(fixed_numbers)

        optimizer = torch.optim.Adam([
            {"params": [z_atom], "lr": config.learning_rate_atom},
            {"params": [z_frac], "lr": config.learning_rate_frac},
            {"params": [z_lattice], "lr": config.learning_rate_lattice},
        ])
        multiplier_count = 1 if config.hull_feasibility_only else 2
        multipliers = torch.zeros(
            multiplier_count, device=z_atom.device, dtype=z_atom.dtype
        )
        rho = config.augmented_lagrangian_rho
        history: list[dict[str, Any]] = []
        best_key = (1, float("inf"), float("inf"))
        best_source = tuple(value.detach().clone() for value in variables)
        best_row: dict[str, Any] = {}
        consecutive_feasible = 0

        for iteration in range(config.max_iterations):
            optimizer.zero_grad(set_to_none=True)
            terminal = integrate_flow(
                self.model, CrystalState(*variables), self.mask,
                steps=config.ode_steps, method="midpoint", freeze_atom=False,
            )
            current_types, frac, lattice = decode_terminal(
                self.model, terminal, self.mask
            )
            energy_rows = energy_per_atom(frac, lattice)
            energy = energy_rows.mean()
            raw_hull = self.hull.raw_e_hull(
                energy.reshape(1), fixed_species, self.mask
            ).mean()
            chosen = terminal.atom.gather(-1, fixed_indices.unsqueeze(-1))
            gaps = (terminal.atom - chosen)[competitor_mask]
            type_region = (
                config.type_region_temperature * torch.logsumexp(
                    gaps / config.type_region_temperature, dim=0
                ) + config.type_logit_margin
            )
            hull_constraint = raw_hull - config.hull_threshold_eV_atom
            constraints = (
                hull_constraint.reshape(1)
                if config.hull_feasibility_only
                else torch.stack((hull_constraint, type_region))
            ).to(z_atom.dtype)
            positive = F.relu(constraints)
            drift = sum(
                (value - initial).square().mean()
                for value, initial in zip(variables, reference, strict=True)
            )
            if config.hull_feasibility_only:
                # Constant objective with only c(z)=E_hull(z)-threshold <= 0.
                # The squared positive part is the differentiable feasibility
                # merit used identically by the L-BFGS comparison.
                loss = 0.5 * positive.square().sum()
            else:
                loss = (
                    raw_hull
                    + torch.dot(multipliers.detach(), constraints)
                    + 0.5 * rho * positive.square().sum()
                    + config.source_prior_weight * drift
                )
            fmax, stress_fro = relaxation_metrics_from_energy_gradient(
                energy, frac, lattice, create_graph=True
            )
            loss = loss + relaxation_barrier(fmax, stress_fro, config)
            loss.backward()
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                variables, config.gradient_clip_norm
            )
            if not bool(
                torch.isfinite(loss)
                and torch.isfinite(constraints).all()
                and torch.isfinite(pre_clip_norm)
                and all(value.grad is not None and torch.isfinite(value.grad).all()
                        for value in variables)
            ):
                raise FloatingPointError("non-finite Projected Adam value or gradient")

            type_stable = bool(torch.equal(
                current_types[self.mask], fixed_types[self.mask]
            ))
            active_feasible = bool((constraints.detach() <= 0).all())
            exact = bool(type_stable and self.feasibility_callback(
                fixed_types.detach(), frac.detach(), lattice.detach(),
                float(raw_hull.detach().clamp_min(0)),
            ))
            row = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "E_UMA_eV_atom": float(energy.detach()),
                "raw_E_hull_eV_atom": float(raw_hull.detach()),
                "E_hull_eV_atom": float(raw_hull.detach().clamp_min(0)),
                "Fmax_eV_A": float(
                    fmax.detach()
                ),
                "stress_fro_eV_A3": float(
                    stress_fro.detach()
                ),
                "constraints": constraints.detach().cpu().tolist(),
                "multipliers": multipliers.detach().cpu().tolist(),
                "rho": float(rho),
                "gradient_norm_pre_clip": float(pre_clip_norm.detach()),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "fixed_type_region": {
                    "stable": type_stable,
                    "smooth_max_gap": float(type_region.detach()),
                },
                "active_constraints_feasible": active_feasible,
                "exact_constraints_feasible": exact,
            }
            history.append(row)
            violation = float(positive.detach().sum())
            key = (
                0 if active_feasible else 1, violation,
                row["raw_E_hull_eV_atom"],
            )
            if key < best_key:
                best_key, best_row = key, dict(row)
                best_source = tuple(value.detach().clone() for value in variables)

            consecutive_feasible = (
                consecutive_feasible + 1 if active_feasible else 0
            )
            if consecutive_feasible >= config.consecutive_feasible_required:
                break

            optimizer.step()
            with torch.no_grad():
                z_atom.clamp_(-8.0, 8.0)
                z_frac.remainder_(1.0)
                z_lattice.clamp_(-6.0, 6.0)
                if not config.hull_feasibility_only:
                    multipliers.copy_(torch.relu(
                        multipliers
                        + config.dual_learning_rate * constraints.detach()
                    ).clamp_max(config.multiplier_max))

        optimized = CrystalState(*best_source)
        with torch.no_grad():
            terminal = integrate_flow(
                self.model, optimized, self.mask, steps=config.ode_steps,
                method="midpoint", freeze_atom=False,
            )
        status = "complete" if best_key[0] == 0 else "constraints_not_satisfied"
        return HybridAdamResult(
            source=optimized, terminal=terminal, history=history,
            selected_values=best_row, status=status,
            solver_stats={
                "iterations": len(history),
                "stopped_after_consecutive_feasible": (
                    consecutive_feasible >= config.consecutive_feasible_required
                ),
                "final_rho": float(rho),
                "merit": (
                    "0.5*relu(E_hull-threshold)^2"
                    if config.hull_feasibility_only else "augmented_lagrangian"
                ),
            },
        )

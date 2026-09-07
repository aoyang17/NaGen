"""Multi-objective single-shooting optimization in geometry source space."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional runtime display
    def tqdm(iterable, **kwargs):
        return iterable

from .guidance import terminal_constraint_terms
from .model import CrystalState, CrystalVectorField
from .novelty import NoveltyIndex, crystal_descriptor
from .sample import decode_terminal, integrate_flow


def closed_form_mgda(g_first: torch.Tensor, g_second: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Minimum-norm point on the segment joining two objective gradients."""
    if g_first.shape != g_second.shape:
        raise ValueError("MGDA gradients must have identical shapes")
    difference = g_first - g_second
    denominator = torch.dot(difference, difference)
    if float(denominator) <= torch.finfo(denominator.dtype).eps:
        alpha = 0.5
    else:
        alpha = float(torch.clamp(
            torch.dot(g_second, g_second - g_first) / denominator, 0.0, 1.0
        ))
    return alpha, alpha * g_first + (1.0 - alpha) * g_second


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _flatten(gradients: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def _unflatten(vector: torch.Tensor, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    output: list[torch.Tensor] = []
    start = 0
    for tensor in tensors:
        stop = start + tensor.numel()
        output.append(vector[start:stop].reshape_as(tensor))
        start = stop
    return output


@dataclass(frozen=True)
class ShootingConfig:
    # Joint optimization starts immediately after the first unconditional
    # trajectory; no objective-free restoration phase by default.
    restoration_steps: int = 0
    guidance_steps: int = 40
    learning_rate: float = 0.01
    ode_steps: int = 24
    integrator: str = "midpoint"
    gradient_clip: float = 10.0
    source_prior: float = 0.02
    augmented_rho: float = 1.0
    objective_scales: tuple[float, float] = (1.0, 1.0)
    reintegration_atol: float = 1e-6
    reintegration_rtol: float = 1e-5
    use_hull: bool = True
    report_hull: bool = True
    use_novelty: bool = True
    use_constraints: bool = True
    distance_weight: float = 20.0
    p_coordination_weight: float = 8.0
    fe_coordination_weight: float = 5.0
    # The hull threshold is an optimization constraint, not a post-filter.
    hull_threshold_eV_atom: float = 0.150
    hull_penalty_weight: float = 100.0


@dataclass
class ShootingResult:
    z0_star: CrystalState
    terminal: CrystalState
    history: list[dict[str, Any]]
    status: str
    model_hash: str


class CrystalDesignProblem:
    """Endpoint evaluator on the all-UMA caliber with a differentiable UMA energy.

    Both terms of `E_hull` are UMA energies: the candidate from `energy_per_atom`
    and the reference from `UMAHullCache.get`.  Mixing in an MP hull or an MP2020
    compatibility correction puts the two terms on different energy scales and is
    forbidden by the 8/27 protocol, so those arguments are rejected.
    """

    def __init__(
        self, model: CrystalVectorField, mask: torch.Tensor,
        energy_per_atom: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        uma_hull_reference_eV_atom: float, novelty_index: NoveltyIndex | None = None,
        mp2020_hull_reference_eV_atom: float | None = None,
        mp2020_correction_eV_atom: float = 0.0,
        fe_coordination_prior: tuple[float, float, float] | None = None,
        constraint_margin_A: float = 0.035,
    ) -> None:
        self.model = model
        self.mask = mask
        self.energy_per_atom = energy_per_atom
        self.uma_hull_reference_eV_atom = float(uma_hull_reference_eV_atom)
        # Kept in the signature so stale callers fail loudly instead of silently
        # changing which energy scale their numbers are on.
        if mp2020_hull_reference_eV_atom is not None or mp2020_correction_eV_atom:
            raise ValueError(
                "E_hull must be scored on the all-UMA caliber; the MP2020 "
                "cross-scale reference and correction are forbidden by the 8/27 "
                "protocol.  Pass only uma_hull_reference_eV_atom."
            )
        self.novelty_index = novelty_index
        self.fe_coordination_prior = fe_coordination_prior
        if constraint_margin_A < 0:
            raise ValueError("constraint_margin_A must be non-negative")
        self.constraint_margin_A = float(constraint_margin_A)

    def evaluate(self, terminal: CrystalState) -> dict[str, torch.Tensor]:
        types, frac, lattice = decode_terminal(self.model, terminal, self.mask)
        uma = self.energy_per_atom(frac, lattice).mean()
        hull = uma - self.uma_hull_reference_eV_atom
        if self.novelty_index is None:
            novelty = (frac[self.mask] - 0.5).square().mean().sqrt()
        else:
            descriptor = crystal_descriptor(types, frac, lattice, self.mask)
            novelty = self.novelty_index.smooth_score(descriptor).mean()
        return {"E_UMA": uma, "E_hull": hull, "novelty": novelty}

    def objective_gradients(
        self, source_tensors: list[torch.Tensor], terminal: CrystalState,
        use_hull: bool = True, use_novelty: bool = True,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        types, frac, lattice = decode_terminal(self.model, terminal, self.mask)
        zero = tuple(torch.zeros_like(tensor) for tensor in source_tensors)
        values: dict[str, torch.Tensor] = {}
        if use_hull:
            uma = self.energy_per_atom(frac, lattice).mean()
            values.update({
                "E_UMA": uma,
                "E_hull": uma - self.uma_hull_reference_eV_atom,
            })
            hull_gradients = torch.autograd.grad(
                values["E_hull"], source_tensors,
                retain_graph=True, allow_unused=False,
            )
        else:
            hull_gradients = zero
        if use_novelty:
            if self.novelty_index is None:
                novelty = (frac[self.mask] - 0.5).square().mean().sqrt()
            else:
                descriptor = crystal_descriptor(types, frac, lattice, self.mask)
                novelty = self.novelty_index.smooth_score(descriptor).mean()
            values["novelty"] = novelty
            novelty_gradients = torch.autograd.grad(
                -novelty, source_tensors, retain_graph=True, allow_unused=False
            )
        else:
            novelty_gradients = zero
        return values, hull_gradients, novelty_gradients

    def constraint_terms(self, terminal: CrystalState) -> dict[str, torch.Tensor]:
        terms = terminal_constraint_terms(
            self.model,
            terminal,
            self.mask,
            margin_A=self.constraint_margin_A,
            fe_coordination_prior=self.fe_coordination_prior,
        )
        return {
            name: value for name, value in terms.items()
            if name in {"distance", "p_coordination", "fe_coordination"}
        }


class ShootingFlowRunner:
    def __init__(self, model: CrystalVectorField, problem: CrystalDesignProblem) -> None:
        self.model = model
        self.problem = problem

    def run(
        self, n_atoms: int, atom_state: torch.Tensor, z0: CrystalState,
        config: ShootingConfig = ShootingConfig(),
    ) -> ShootingResult:
        if n_atoms != int(self.problem.mask.sum()):
            raise ValueError("N is inconsistent with the fixed mask")
        if not torch.equal(atom_state, z0.atom):
            raise ValueError("A must be the fixed atom component of z0")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        before_hash = model_sha256(self.model)
        reference_frac = z0.frac.detach().clone()
        reference_lattice = z0.lattice.detach().clone()
        frac = reference_frac.clone().requires_grad_(True)
        lattice = reference_lattice.clone().requires_grad_(True)
        optimized = [frac, lattice]
        optimizer = torch.optim.Adam(optimized, lr=config.learning_rate)
        multipliers: dict[str, float] = {}
        history: list[dict[str, Any]] = []
        scale_hull = max(config.objective_scales[0], 1e-12)
        scale_novelty = max(config.objective_scales[1], 1e-12)
        status = "complete"
        total_steps = config.restoration_steps + config.guidance_steps
        progress = tqdm(
            range(total_steps), desc="M0 joint optimization", unit="step",
            leave=True,
        )
        for step in progress:
            restoring = step < config.restoration_steps
            optimizer.zero_grad(set_to_none=True)
            source = CrystalState(atom_state, frac, lattice)
            terminal = integrate_flow(
                self.model, source, self.problem.mask, steps=config.ode_steps,
                method=config.integrator, freeze_atom=True,
            )
            values, hull_parts, novelty_parts = self.problem.objective_gradients(
                optimized,
                terminal,
                config.use_hull and not restoring,
                config.use_novelty and not restoring,
            )
            g_hull = _flatten(hull_parts) / scale_hull
            g_novelty = _flatten(novelty_parts) / scale_novelty
            if restoring:
                alpha, combined = 0.0, torch.zeros_like(g_hull)
            elif config.use_hull and config.use_novelty:
                alpha, combined = closed_form_mgda(g_hull, g_novelty)
            elif config.use_hull:
                alpha, combined = 1.0, g_hull
            elif config.use_novelty:
                alpha, combined = 0.0, g_novelty
            else:
                alpha, combined = 0.0, torch.zeros_like(g_hull)
            constraints = (
                self.problem.constraint_terms(terminal)
                if config.use_constraints else {}
            )
            if config.use_hull and "E_hull" in values:
                # Exact hard-gate target represented during optimization by an
                # augmented-Lagrangian hinge. The final evaluator still applies
                # the boolean <= threshold gate.
                constraints["hull_threshold"] = torch.relu(
                    values["E_hull"] - config.hull_threshold_eV_atom
                )
            constraint_weights = {
                "distance": config.distance_weight,
                "p_coordination": config.p_coordination_weight,
                "fe_coordination": config.fe_coordination_weight,
                "hull_threshold": config.hull_penalty_weight,
            }
            if restoring:
                augmented = sum(
                    constraint_weights.get(name, 1.0) * value
                    for name, value in constraints.items()
                )
            else:
                augmented = sum(
                    multipliers.get(name, 0.0) * value
                    + 0.5
                    * config.augmented_rho
                    * constraint_weights.get(name, 1.0)
                    * value.square()
                    for name, value in constraints.items()
                )
            prior = (
                (frac - reference_frac).square().mean()
                + (lattice - reference_lattice).square().mean()
            )
            auxiliary = augmented + config.source_prior * prior
            auxiliary_parts = torch.autograd.grad(auxiliary, optimized, allow_unused=False)
            total_gradient = combined + _flatten(auxiliary_parts)
            if not bool(torch.isfinite(total_gradient).all()):
                status = "non_finite_gradient"
                break
            norm = total_gradient.norm()
            if float(norm) > config.gradient_clip:
                total_gradient = total_gradient * (config.gradient_clip / norm)
            for tensor, gradient in zip(optimized, _unflatten(total_gradient, optimized), strict=True):
                tensor.grad = gradient
            optimizer.step()
            with torch.no_grad():
                frac.remainder_(1.0)
                lattice.clamp_(-6.0, 6.0)
                if not restoring:
                    for name, value in constraints.items():
                        multipliers[name] = max(
                            0.0, multipliers.get(name, 0.0)
                            + config.augmented_rho * float(value.detach())
                        )
            denominator = g_hull.norm() * g_novelty.norm()
            cosine = float(torch.dot(g_hull, g_novelty) / denominator) if denominator > 0 else 0.0
            history.append({
                "step": step,
                "phase": "restoration" if restoring else "multiobjective",
                "phase_step": (
                    step if restoring else step - config.restoration_steps
                ),
                "E_UMA": float(values["E_UMA"].detach()) if "E_UMA" in values else None,
                "E_ref_UMA": (
                    self.problem.uma_hull_reference_eV_atom
                    if config.report_hull else None
                ),
                "E_hull": (
                    float(values["E_hull"].detach())
                    if config.report_hull and "E_hull" in values else None
                ),
                "novelty": float(values["novelty"].detach()) if "novelty" in values else None,
                "hull_gradient_norm": float(g_hull.norm()),
                "novelty_gradient_norm": float(g_novelty.norm()),
                "gradient_cosine": cosine, "mgda_hull_weight": alpha,
                "source_drift": float(prior.detach().sqrt()),
                "multipliers": dict(multipliers),
                "constraints": {name: float(value.detach()) for name, value in constraints.items()},
            })
            if not restoring and "E_hull" in values:
                progress.set_postfix(
                    E_hull=f"{float(values['E_hull'].detach()):.3f}",
                    violation=f"{float(constraints.get('hull_threshold', torch.zeros_like(values['E_hull'])).detach()):.3f}",
                )
        z0_star = CrystalState(atom_state.detach(), frac.detach(), lattice.detach())
        with torch.no_grad():
            terminal = integrate_flow(
                self.model, z0_star, self.problem.mask, steps=config.ode_steps,
                method=config.integrator, freeze_atom=True,
            )
            reintegrated = integrate_flow(
                self.model, z0_star, self.problem.mask, steps=config.ode_steps,
                method=config.integrator, freeze_atom=True,
            )
        for first, second in zip(terminal, reintegrated, strict=True):
            if not torch.allclose(
                first, second, atol=config.reintegration_atol,
                rtol=config.reintegration_rtol,
            ):
                status = "reintegration_mismatch"
        after_hash = model_sha256(self.model)
        if before_hash != after_hash:
            raise RuntimeError("Flow parameters changed during inference")
        return ShootingResult(z0_star, terminal, history, status, before_hash)

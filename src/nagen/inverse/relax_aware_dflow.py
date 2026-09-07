"""Relaxation-aware one-shot IPOPT optimization of z_A, z_X and z_L."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .differentiable_mp_hull import DifferentiableMP2020Hull
from .guidance import soft_constraint_terms
from .lattice import decode_lattice, unstandardize_lattice
from .model import CrystalState, CrystalVectorField
from .novelty import NoveltyIndex
from .relaxation_surrogate import RelaxationSurrogateEnsemble, soft_crystal_descriptor
from .sample import integrate_flow
from .spec import DEFAULT_SPEC


@dataclass(frozen=True)
class RelaxAwareConfig:
    steps: int = 160
    ode_steps: int = 24
    atom_temperature: float = 0.35
    hull_temperature: float = 0.03
    hull_uncertainty_kappa: float = 1.0
    hull_scale: float = 1.0
    hull_weight: float = 4.0
    novelty_weight: float = 0.10
    distance_weight: float = 40.0
    p_coordination_weight: float = 12.0
    fe_coordination_weight: float = 4.0
    volume_weight: float = 2.0
    composition_weight: float = 20.0
    geometry_survival_weight: float = 2.0
    source_prior_weight: float = 0.02
    ipopt_tolerance: float = 1.0e-5
    ipopt_acceptable_tolerance: float = 1.0e-4
    ipopt_print_level: int = 0


@dataclass
class RelaxAwareResult:
    z0_star: CrystalState
    terminal: CrystalState
    status: str
    history: list[dict[str, Any]]
    solver_stats: dict[str, Any]
    adaptive_weights: dict[str, float]


def _smooth_positive(value: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softplus(value / temperature) * temperature


def soft_composition_terms(
    probabilities: torch.Tensor, mask: torch.Tensor, temperature: float = 0.02
) -> dict[str, torch.Tensor]:
    counts = (probabilities * mask.unsqueeze(-1)).sum(dim=1)
    na, fe, phosphorus, oxygen = counts.unbind(dim=-1)
    phosphorus_safe = phosphorus.clamp_min(0.25)
    fe_safe = fe.clamp_min(0.25)
    m = na / phosphorus_safe
    n = fe / phosphorus_safe
    y = 4.0 - oxygen / phosphorus_safe
    fe_valence = (3.0 - 2.0 * y - m) / n.clamp_min(0.05)
    kappa_fe = m + DEFAULT_SPEC.fe_valence_max * n + 2.0 * y - 3.0
    extracted = torch.minimum(m, kappa_fe)
    mass = (
        m * DEFAULT_SPEC.atomic_masses["Na"]
        + n * DEFAULT_SPEC.atomic_masses["Fe"]
        + DEFAULT_SPEC.atomic_masses["P"]
        + (4.0 - y) * DEFAULT_SPEC.atomic_masses["O"]
    ).clamp_min(1.0)
    capacity = extracted * DEFAULT_SPEC.faraday_C_mol / (3.6 * mass)

    def violation(value: torch.Tensor) -> torch.Tensor:
        return _smooth_positive(value, temperature).square().mean()

    return {
        "support": violation(0.5 - counts).mean(),
        "y_domain": violation(DEFAULT_SPEC.y_min - y)
        + violation(y - DEFAULT_SPEC.y_max),
        "m_domain": violation(1.0 - m) + violation(n - m),
        "n_domain": violation(n - 1.0),
        "fe_valence": violation(DEFAULT_SPEC.fe_valence_min - fe_valence)
        + violation(fe_valence - DEFAULT_SPEC.fe_valence_max),
        "extractable_na": violation(-kappa_fe),
        "capacity": violation(DEFAULT_SPEC.capacity_min_mAh_g - capacity)
        / (DEFAULT_SPEC.capacity_min_mAh_g ** 2),
    }


class RelaxAwareProblem:
    def __init__(
        self, model: CrystalVectorField, mask: torch.Tensor,
        surrogate: RelaxationSurrogateEnsemble,
        hull: DifferentiableMP2020Hull,
        novelty: NoveltyIndex | None,
    ) -> None:
        self.model = model
        self.mask = mask
        self.surrogate = surrogate
        self.hull = hull
        self.novelty = novelty

    def components(
        self, terminal: CrystalState, config: RelaxAwareConfig
    ) -> dict[str, torch.Tensor]:
        probabilities = torch.softmax(
            terminal.atom / config.atom_temperature, dim=-1
        ) * self.mask.unsqueeze(-1)
        physical = unstandardize_lattice(
            terminal.lattice, self.model.lattice_mean, self.model.lattice_std
        )
        lattice = decode_lattice(physical, self.mask.sum(dim=1))
        frac = torch.remainder(terminal.frac, 1.0)
        descriptor = soft_crystal_descriptor(
            probabilities, frac, lattice, self.mask
        )
        prediction = self.surrogate.forward_descriptor(descriptor)
        conservative_energy = (
            prediction["final_energy_mean"]
            + config.hull_uncertainty_kappa * prediction["final_energy_std"]
        )
        raw_hull = self.hull.raw_e_hull(
            conservative_energy, probabilities, self.mask
        )
        e_hull = _smooth_positive(raw_hull, config.hull_temperature).mean()
        hull_violation = _smooth_positive(
            raw_hull - DEFAULT_SPEC.hull_max_eV_atom, config.hull_temperature
        ).square().mean() / max(config.hull_scale ** 2, 1.0e-12)
        if self.novelty is None:
            novelty = descriptor.square().mean(dim=-1).sqrt().mean()
        else:
            novelty = self.novelty.smooth_score(descriptor).mean()
        geometry = soft_constraint_terms(
            self.model, terminal, self.mask,
            temperature=config.atom_temperature,
        )
        composition_rows = soft_composition_terms(probabilities, self.mask)
        composition = sum(composition_rows.values())
        survival = -torch.log(
            prediction["geometry_probability_mean"].clamp_min(1.0e-6)
        ).mean()
        return {
            "E_hull_proxy": e_hull,
            "raw_E_hull_proxy": raw_hull.mean(),
            "final_E_UMA_prediction": prediction["final_energy_mean"].mean(),
            "final_E_UMA_uncertainty": prediction["final_energy_std"].mean(),
            "hull_violation": hull_violation,
            "novelty": novelty,
            "distance": geometry["distance"],
            "p_coordination": geometry["p_coordination"],
            "fe_coordination": geometry["fe_coordination"],
            "volume": geometry["volume"],
            "composition": composition,
            "geometry_survival": survival,
            **{f"composition_{key}": value for key, value in composition_rows.items()},
        }


class RelaxAwareIPOPTSolver:
    def __init__(self, model: CrystalVectorField, problem: RelaxAwareProblem) -> None:
        self.model = model
        self.problem = problem

    def solve(self, z0: CrystalState, config: RelaxAwareConfig = RelaxAwareConfig()):
        import casadi as ca

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.problem.surrogate.parameters():
            parameter.requires_grad_(False)
        reference = tuple(item.detach().clone() for item in z0)
        sizes = [item.numel() for item in reference]
        offsets = np.cumsum([0, *sizes])
        variable_count = sum(sizes)
        atom_shape, frac_shape, lattice_shape = [item.shape for item in reference]

        def unpack(vector: Any, requires_grad: bool) -> CrystalState:
            flat = torch.as_tensor(
                np.asarray(vector, dtype=np.float64).reshape(-1),
                device=reference[0].device, dtype=reference[0].dtype,
            )
            tensors = []
            for index, (start, stop, shape) in enumerate(zip(
                offsets[:-1], offsets[1:], (atom_shape, frac_shape, lattice_shape),
                strict=True,
            )):
                tensor = flat[start:stop].reshape(shape).clone()
                tensor.requires_grad_(requires_grad)
                tensors.append(tensor)
            return CrystalState(*tensors)

        def raw_components(source: CrystalState):
            terminal = integrate_flow(
                self.model, source, self.problem.mask,
                steps=config.ode_steps, method="midpoint", freeze_atom=False,
            )
            return terminal, self.problem.components(terminal, config)

        # Per-source gradient balancing is computed once and then frozen, so
        # IPOPT sees one deterministic scalar NLP throughout the solve.
        initial_source = CrystalState(*[
            item.detach().clone().requires_grad_(True) for item in reference
        ])
        _, initial_components = raw_components(initial_source)
        base_weights = {
            "hull_violation": config.hull_weight,
            "distance": config.distance_weight,
            "p_coordination": config.p_coordination_weight,
            "fe_coordination": config.fe_coordination_weight,
            "volume": config.volume_weight,
            "composition": config.composition_weight,
            "geometry_survival": config.geometry_survival_weight,
        }
        gradient_norms = {}
        for name in base_weights:
            gradients = torch.autograd.grad(
                initial_components[name], tuple(initial_source), retain_graph=True,
                allow_unused=True,
            )
            gradient_norms[name] = float(torch.sqrt(sum(
                item.square().sum() for item in gradients if item is not None
            )).detach())
        active = [value for value in gradient_norms.values() if value > 1.0e-10]
        target_norm = float(np.median(active)) if active else 1.0
        adaptive_weights = {
            name: weight * float(np.clip(
                target_norm / max(gradient_norms[name], 1.0e-10), 0.1, 10.0
            ))
            for name, weight in base_weights.items()
        }
        history: list[dict[str, Any]] = []
        best_value = float("inf")
        best_vector = np.concatenate([
            item.detach().cpu().double().numpy().reshape(-1) for item in reference
        ])

        def evaluate(vector: Any, need_gradient: bool):
            nonlocal best_value, best_vector
            source = unpack(vector, need_gradient)
            if need_gradient:
                terminal, components = raw_components(source)
            else:
                with torch.no_grad():
                    terminal, components = raw_components(source)
            drift = sum(
                (current - original).square().mean()
                for current, original in zip(source, reference, strict=True)
            )
            merit = (
                components["E_hull_proxy"] / max(config.hull_scale, 1.0e-12)
                - config.novelty_weight * components["novelty"]
                + sum(adaptive_weights[name] * components[name] for name in adaptive_weights)
                + config.source_prior_weight * drift
            )
            value = float(merit.detach())
            if value < best_value:
                best_value = value
                best_vector = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
            if need_gradient:
                gradients = torch.autograd.grad(merit, tuple(source))
                gradient = torch.cat([item.reshape(-1) for item in gradients])
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("non-finite relax-aware D-Flow gradient")
                result_gradient = gradient.detach().cpu().double().numpy()
            else:
                result_gradient = None
                history.append({
                    "evaluation": len(history), "merit": value,
                    **{name: float(item.detach()) for name, item in components.items()},
                })
            return value, result_gradient

        class Objective(ca.Callback):
            def __init__(self):
                ca.Callback.__init__(self)
                self.jac = None
                self.value_x = self.value = self.gradient_x = self.gradient = None
                self.construct("relax_aware_torch_objective")
            def get_n_in(self): return 1
            def get_n_out(self): return 1
            def get_sparsity_in(self, index): return ca.Sparsity.dense(variable_count, 1)
            def get_sparsity_out(self, index): return ca.Sparsity.scalar()
            def eval(self, arguments):
                vector = np.asarray(arguments[0], dtype=np.float64).reshape(-1)
                if self.value_x is None or not np.array_equal(vector, self.value_x):
                    self.value, _ = evaluate(vector, False)
                    self.value_x = vector.copy()
                return [self.value]
            def has_jacobian(self): return True
            def get_jacobian(self, name, inames, onames, opts):
                self.jac = Jacobian(name, self, opts)
                return self.jac

        class Jacobian(ca.Callback):
            def __init__(self, name, parent, opts):
                ca.Callback.__init__(self); self.parent = parent; self.construct(name, opts)
            def get_n_in(self): return 2
            def get_n_out(self): return 1
            def get_sparsity_in(self, index):
                return ca.Sparsity.dense(variable_count, 1) if index == 0 else ca.Sparsity.scalar()
            def get_sparsity_out(self, index): return ca.Sparsity.dense(1, variable_count)
            def eval(self, arguments):
                vector = np.asarray(arguments[0], dtype=np.float64).reshape(-1)
                if self.parent.gradient_x is None or not np.array_equal(vector, self.parent.gradient_x):
                    _, self.parent.gradient = evaluate(vector, True)
                    self.parent.gradient_x = vector.copy()
                return [ca.DM(self.parent.gradient).T]

        objective = Objective()
        symbolic = ca.MX.sym("z_A_X_L", variable_count)
        solver = ca.nlpsol("relax_aware_ipopt", "ipopt", {
            "x": symbolic, "f": objective(symbolic)
        }, {
            "print_time": False, "ipopt.print_level": config.ipopt_print_level,
            "ipopt.max_iter": config.steps, "ipopt.tol": config.ipopt_tolerance,
            "ipopt.acceptable_tol": config.ipopt_acceptable_tolerance,
            "ipopt.hessian_approximation": "limited-memory",
        })
        x0 = np.concatenate([
            item.detach().cpu().double().numpy().reshape(-1) for item in reference
        ])
        lower = np.concatenate((
            np.full(sizes[0], -8.0), np.zeros(sizes[1]), np.full(sizes[2], -6.0)
        ))
        upper = np.concatenate((
            np.full(sizes[0], 8.0), np.ones(sizes[1]), np.full(sizes[2], 6.0)
        ))
        solution = solver(x0=x0, lbx=lower, ubx=upper)
        stats = dict(solver.stats())
        evaluate(np.asarray(solution["x"]).reshape(-1), False)
        z0_star = unpack(best_vector, False)
        with torch.no_grad():
            terminal = integrate_flow(
                self.model, z0_star, self.problem.mask,
                steps=config.ode_steps, method="midpoint", freeze_atom=False,
            )
        serializable = {
            key: value for key, value in stats.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        serializable["objective_evaluations"] = len(history)
        return_status = str(stats.get("return_status", "unknown"))
        status = "complete" if np.isfinite(best_value) else "non_finite"
        serializable["return_status"] = return_status
        return RelaxAwareResult(
            z0_star, terminal, status, history, serializable, adaptive_weights
        )

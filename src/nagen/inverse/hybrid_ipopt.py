"""IPOPT solver for the unified unconditional all-UMA D-Flow problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F

from .differentiable_uma_hull import DifferentiableUMAHull
from .guidance import soft_constraint_terms
from .hybrid_lbfgs import _minimal_composition_terms
from .model import CrystalState, CrystalVectorField
from .novelty import NoveltyIndex
from .relaxation_surrogate import soft_crystal_descriptor
from .sample import decode_terminal, integrate_flow
from .uma_guidance import SecondOrderEvaluator


@dataclass(frozen=True)
class HybridIPOPTConfig:
    max_iterations: int = 300
    ode_steps: int = 24
    constraint_mode: str = "full"
    atom_temperature: float = 0.35
    type_region_temperature: float = 1.0e-3
    type_logit_margin: float = 1.0e-4
    hull_threshold_eV_atom: float = 0.150
    force_tolerance_eV_A: float = 0.01
    stress_tolerance_eV_A3: float = 0.10 / 160.21766208
    hull_objective_weight: float = 50.0
    hull_violation_weight: float = 0.0
    # Force and stress are explicit IPOPT constraints. Keeping their duplicate
    # objective penalties at zero prevents their very small physical tolerances
    # from overwhelming the deliberately high hull objective.
    force_violation_weight: float = 0.0
    stress_violation_weight: float = 0.0
    distance_weight: float = 80.0
    p_coordination_weight: float = 24.0
    fe_coordination_weight: float = 12.0
    volume_weight: float = 10.0
    composition_weight: float = 200.0
    novelty_weight: float = 0.01
    source_prior_weight: float = 0.002
    tolerance: float = 1.0e-5
    acceptable_tolerance: float = 1.0e-4
    print_level: int = 0


@dataclass
class HybridIPOPTResult:
    source: CrystalState
    terminal: CrystalState
    history: list[dict[str, Any]]
    selected_values: dict[str, Any]
    status: str
    solver_stats: dict[str, Any]


class HybridIPOPTSolver:
    def __init__(
        self, model: CrystalVectorField, mask: torch.Tensor,
        uma_evaluator: SecondOrderEvaluator, hull: DifferentiableUMAHull,
        atomic_number_lookup: torch.Tensor, novelty: NoveltyIndex,
        feasibility_callback: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, float], bool
        ],
    ) -> None:
        self.model = model
        self.mask = mask
        self.uma_evaluator = uma_evaluator
        self.hull = hull
        self.atomic_number_lookup = atomic_number_lookup
        self.novelty = novelty
        self.feasibility_callback = feasibility_callback

    def solve(
        self, source: CrystalState,
        config: HybridIPOPTConfig = HybridIPOPTConfig(),
    ) -> HybridIPOPTResult:
        try:
            import casadi as ca
        except ImportError as error:
            raise RuntimeError("CasADi/IPOPT is required") from error
        modes = {"hull", "hull_force", "hull_force_stress", "full"}
        if config.constraint_mode not in modes:
            raise ValueError(
                f"constraint_mode must be one of {sorted(modes)}, got "
                f"{config.constraint_mode!r}"
            )
        use_force = config.constraint_mode in {
            "hull_force", "hull_force_stress", "full"
        }
        use_stress = config.constraint_mode in {"hull_force_stress", "full"}
        use_full_problem = config.constraint_mode == "full"
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        reference = tuple(value.detach().clone() for value in source)
        sizes = tuple(value.numel() for value in reference)
        variable_count = sum(sizes)
        history: list[dict[str, Any]] = []
        best_key = (1, 1, float("inf"), float("inf"))
        best_vector = torch.cat([value.reshape(-1) for value in reference]).detach()
        best_row: dict[str, Any] = {}
        with torch.no_grad():
            reference_terminal = integrate_flow(
                self.model, CrystalState(*reference), self.mask,
                steps=config.ode_steps, method="midpoint", freeze_atom=False,
            )
            fixed_type_indices = reference_terminal.atom.argmax(dim=-1)
            fixed_types = (fixed_type_indices + 1).masked_fill(~self.mask, 0)
            fixed_species = F.one_hot(
                fixed_type_indices, reference_terminal.atom.shape[-1]
            ).to(reference_terminal.atom) * self.mask.unsqueeze(-1)
            fixed_numbers = self.atomic_number_lookup.to(fixed_types.device)[
                fixed_type_indices[0, self.mask[0]]
            ]
            competitor_mask = F.one_hot(
                fixed_type_indices, reference_terminal.atom.shape[-1]
            ).logical_not() & self.mask.unsqueeze(-1)

        def evaluate(vector: Any):
            nonlocal best_key, best_vector, best_row
            flat = torch.as_tensor(
                np.asarray(vector, dtype=np.float64).reshape(-1),
                dtype=reference[0].dtype, device=reference[0].device,
            )
            offset = 0
            variables = []
            for template, size in zip(reference, sizes, strict=True):
                variables.append(
                    flat[offset:offset + size].reshape_as(template).clone().requires_grad_(True)
                )
                offset += size
            z_atom, z_frac, z_lattice = variables
            terminal = integrate_flow(
                self.model, CrystalState(*variables), self.mask,
                steps=config.ode_steps, method="midpoint", freeze_atom=False,
            )
            current_types, frac, lattice = decode_terminal(
                self.model, terminal, self.mask
            )
            species = fixed_species
            energy_rows, forces, stress = self.uma_evaluator(
                fixed_numbers, frac, lattice
            )
            energy = energy_rows.mean()
            raw_hull = self.hull.raw_e_hull(
                energy.reshape(1), species, self.mask
            ).mean()
            e_hull = raw_hull.clamp_min(0.0)
            fmax = torch.linalg.vector_norm(forces, dim=-1).amax()
            stress_fro = torch.linalg.matrix_norm(stress[0], ord="fro")
            geometry = soft_constraint_terms(
                self.model, terminal, self.mask,
                temperature=config.atom_temperature,
                probabilities_override=species,
            )
            composition = _minimal_composition_terms(species, self.mask)
            novelty = self.novelty.smooth_score(
                soft_crystal_descriptor(species, frac, lattice, self.mask)
            ).mean()
            drift = sum(
                (value - initial).square().mean()
                for value, initial in zip(variables, reference, strict=True)
            )
            hull_violation = F.relu(raw_hull - config.hull_threshold_eV_atom)
            force_violation = F.relu(
                fmax - config.force_tolerance_eV_A
            ) / config.force_tolerance_eV_A
            stress_violation = F.relu(
                stress_fro - config.stress_tolerance_eV_A3
            ) / config.stress_tolerance_eV_A3
            chosen_logits = terminal.atom.gather(
                -1, fixed_type_indices.unsqueeze(-1)
            )
            competitor_gaps = (
                terminal.atom - chosen_logits
            )[competitor_mask]
            # A smooth conservative approximation of max(competitor-chosen).
            # If this is <= 0, every fixed type remains the strict argmax.
            type_region = (
                config.type_region_temperature * torch.logsumexp(
                    competitor_gaps / config.type_region_temperature, dim=0
                )
                + config.type_logit_margin
            )
            full_problem_terms = (
                config.distance_weight * geometry["distance"]
                + config.p_coordination_weight * geometry["p_coordination"]
                + config.fe_coordination_weight * geometry["fe_coordination"]
                + config.volume_weight * geometry["volume"]
                + config.composition_weight * sum(composition.values())
                - config.novelty_weight * novelty
            ) if use_full_problem else raw_hull.new_zeros(())
            merit = (
                config.hull_objective_weight * raw_hull
                + config.hull_violation_weight * hull_violation.square()
                + config.force_violation_weight * force_violation.square()
                + config.stress_violation_weight * stress_violation.square()
                + full_problem_terms
                + config.source_prior_weight * drift
            )
            # Dimensionless, smooth constraints.  Per-atom squared force
            # constraints are exactly equivalent to Fmax <= F_tol, without
            # the nonsmooth maximum.  The squared Frobenius constraint is
            # likewise equivalent to the stress norm threshold.
            constraint_parts = [
                (raw_hull - config.hull_threshold_eV_atom).reshape(1)
            ]
            if use_force:
                constraint_parts.append(
                    forces.square().sum(dim=-1)
                    - config.force_tolerance_eV_A ** 2
                )
            if use_stress:
                stress_scale = 1.0 / 160.21766208  # 1 GPa in eV/A^3
                constraint_parts.append((
                    stress_fro.square() - config.stress_tolerance_eV_A3 ** 2
                ).reshape(1) / stress_scale ** 2)
            constraint_parts.append(type_region.reshape(1))
            constraint_values = torch.cat(constraint_parts)
            objective_gradients = torch.autograd.grad(
                merit, variables, retain_graph=True, allow_unused=True
            )
            objective_gradients = tuple(
                torch.zeros_like(variable) if gradient is None else gradient
                for variable, gradient in zip(variables, objective_gradients, strict=True)
            )
            jacobian_rows = []
            for index in range(len(constraint_values)):
                gradients = torch.autograd.grad(
                    constraint_values[index], variables,
                    retain_graph=index < len(constraint_values) - 1,
                    allow_unused=True,
                )
                gradients = tuple(
                    torch.zeros_like(variable) if gradient is None else gradient
                    for variable, gradient in zip(variables, gradients, strict=True)
                )
                jacobian_rows.append(torch.cat([item.reshape(-1) for item in gradients]))
            gradient = torch.cat([item.reshape(-1) for item in objective_gradients])
            jacobian = torch.stack(jacobian_rows)
            if not bool(
                torch.isfinite(merit) and torch.isfinite(constraint_values).all()
                and torch.isfinite(gradient).all() and torch.isfinite(jacobian).all()
            ):
                raise FloatingPointError("non-finite IPOPT value or derivative")
            type_stable = bool(torch.equal(
                current_types[self.mask], fixed_types[self.mask]
            ))
            exact = bool(type_stable and self.feasibility_callback(
                fixed_types.detach(), frac.detach(), lattice.detach(),
                float(e_hull.detach()),
            ))
            row: dict[str, Any] = {
                "evaluation": len(history), "merit": float(merit.detach()),
                "E_UMA_eV_atom": float(energy.detach()),
                "raw_E_hull_eV_atom": float(raw_hull.detach()),
                "E_hull_eV_atom": float(e_hull.detach()),
                "Fmax_eV_A": float(fmax.detach()),
                "stress_fro_eV_A3": float(stress_fro.detach()),
                "novelty": float(novelty.detach()),
                "exact_constraints_feasible": bool(exact),
                "active_constraint_mode": config.constraint_mode,
                "fixed_type_region": {
                    "stable": type_stable,
                    "smooth_max_gap": float(type_region.detach()),
                },
                "ipopt_constraints": constraint_values.detach().cpu().tolist(),
                "objective_gradient_norm": float(gradient.norm().detach()),
                "constraint_jacobian_row_norms": jacobian.norm(dim=1).detach().cpu().tolist(),
                "geometry": {key: float(value.detach()) for key, value in geometry.items()},
                "composition": {key: float(value.detach()) for key, value in composition.items()},
            }
            history.append(row)
            relaxation_ok = (
                row["Fmax_eV_A"] <= config.force_tolerance_eV_A
                and row["stress_fro_eV_A3"] <= config.stress_tolerance_eV_A3
            )
            active_ok = bool((constraint_values.detach() <= 0.0).all())
            full = bool(active_ok and (exact if use_full_problem else True))
            row["active_constraints_feasible"] = active_ok
            violation = sum(max(float(value), 0.0) for value in row["ipopt_constraints"])
            key = (
                0 if full else 1, 0 if type_stable else 1,
                violation + (
                    0.0 if (exact or not use_full_problem) else 1.0
                ), row["merit"],
            )
            if key < best_key:
                best_key, best_row = key, dict(row)
                best_vector = flat.detach().clone()
            return (
                float(merit.detach()), gradient.detach().cpu().double().numpy(),
                constraint_values.detach().cpu().double().numpy(),
                jacobian.detach().cpu().double().numpy(),
            )

        cached_x = None
        cached = None

        def cached_evaluate(argument: Any):
            nonlocal cached_x, cached
            vector = np.asarray(argument, dtype=np.float64).reshape(-1)
            if cached_x is None or not np.array_equal(vector, cached_x):
                cached = evaluate(vector)
                cached_x = vector.copy()
            return cached

        constraint_count = (
            2 + (int(self.mask.sum()) if use_force else 0)
            + (1 if use_stress else 0)
        )

        class Objective(ca.Callback):
            def __init__(self, name):
                ca.Callback.__init__(self); self.jac = None; self.construct(name)
            def get_n_in(self): return 1
            def get_n_out(self): return 1
            def get_sparsity_in(self, i): return ca.Sparsity.dense(variable_count, 1)
            def get_sparsity_out(self, i): return ca.Sparsity.scalar()
            def eval(self, args): return [cached_evaluate(args[0])[0]]
            def has_jacobian(self): return True
            def get_jacobian(self, name, inames, onames, opts):
                self.jac = ObjectiveJac(name, self, opts); return self.jac

        class ObjectiveJac(ca.Callback):
            def __init__(self, name, parent, opts):
                ca.Callback.__init__(self); self.parent = parent; self.construct(name, opts)
            def get_n_in(self): return 2
            def get_n_out(self): return 1
            def get_sparsity_in(self, i):
                return ca.Sparsity.dense(variable_count, 1) if i == 0 else ca.Sparsity.scalar()
            def get_sparsity_out(self, i): return ca.Sparsity.dense(1, variable_count)
            def eval(self, args): return [ca.DM(cached_evaluate(args[0])[1]).T]

        class Constraints(ca.Callback):
            def __init__(self, name):
                ca.Callback.__init__(self); self.jac = None; self.construct(name)
            def get_n_in(self): return 1
            def get_n_out(self): return 1
            def get_sparsity_in(self, i): return ca.Sparsity.dense(variable_count, 1)
            def get_sparsity_out(self, i): return ca.Sparsity.dense(constraint_count, 1)
            def eval(self, args): return [ca.DM(cached_evaluate(args[0])[2])]
            def has_jacobian(self): return True
            def get_jacobian(self, name, inames, onames, opts):
                self.jac = ConstraintsJac(name, self, opts); return self.jac

        class ConstraintsJac(ca.Callback):
            def __init__(self, name, parent, opts):
                ca.Callback.__init__(self); self.parent = parent; self.construct(name, opts)
            def get_n_in(self): return 2
            def get_n_out(self): return 1
            def get_sparsity_in(self, i):
                return ca.Sparsity.dense(variable_count, 1) if i == 0 else ca.Sparsity.dense(constraint_count, 1)
            def get_sparsity_out(self, i): return ca.Sparsity.dense(constraint_count, variable_count)
            def eval(self, args): return [ca.DM(cached_evaluate(args[0])[3])]

        objective = Objective("hybrid_ipopt_objective")
        constraints = Constraints("hybrid_ipopt_constraints")
        symbolic = ca.MX.sym("initial_noise", variable_count)
        nlp = {"x": symbolic, "f": objective(symbolic), "g": constraints(symbolic)}
        options = {
            "print_time": False, "ipopt.print_level": config.print_level,
            "ipopt.max_iter": config.max_iterations, "ipopt.tol": config.tolerance,
            "ipopt.acceptable_tol": config.acceptable_tolerance,
            "ipopt.hessian_approximation": "limited-memory",
        }
        solver = ca.nlpsol("hybrid_ipopt", "ipopt", nlp, options)
        x0 = torch.cat([value.reshape(-1) for value in reference]).cpu().double().numpy()
        atom_size, frac_size, lattice_size = sizes
        lower = np.concatenate((
            np.full(atom_size, -8.0), np.zeros(frac_size), np.full(lattice_size, -6.0)
        ))
        upper = np.concatenate((
            np.full(atom_size, 8.0), np.ones(frac_size), np.full(lattice_size, 6.0)
        ))
        solution = solver(
            x0=x0, lbx=lower, ubx=upper,
            lbg=np.full(constraint_count, -np.inf), ubg=np.zeros(constraint_count),
        )
        stats = dict(solver.stats())
        solution_vector = np.asarray(solution["x"]).reshape(-1)
        evaluate(solution_vector)
        offset = 0
        selected = []
        for template, size in zip(reference, sizes, strict=True):
            selected.append(best_vector[offset:offset + size].reshape_as(template))
            offset += size
        optimized = CrystalState(*selected)
        with torch.no_grad():
            terminal = integrate_flow(
                self.model, optimized, self.mask, steps=config.ode_steps,
                method="midpoint", freeze_atom=False,
            )
        status = "complete" if best_key[0] == 0 else "constraints_not_satisfied"
        serializable = {
            key: value for key, value in stats.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        serializable["evaluations"] = len(history)
        return HybridIPOPTResult(
            optimized, terminal, history, best_row, status, serializable
        )

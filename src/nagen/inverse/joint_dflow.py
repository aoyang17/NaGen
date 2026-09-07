"""Single CasADi/IPOPT nonlinear D-Flow optimization over source noise.

Every optimization step integrates the frozen Flow model from the current
source, evaluates energy/novelty/all differentiable geometry constraints at the
endpoint, and backpropagates one scalar merit function to the source.  There is
no restoration phase and no endpoint post-optimization.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.nn import functional as F

from .differentiable_mp_hull import DifferentiableMP2020Hull
from .guidance import soft_constraint_terms, terminal_constraint_terms
from .lattice import decode_lattice, unstandardize_lattice
from .model import CrystalState, CrystalVectorField
from .novelty import NoveltyIndex, crystal_descriptor
from .relax_aware_dflow import soft_composition_terms
from .relaxation_surrogate import soft_crystal_descriptor
from .sample import decode_terminal, integrate_flow
from .spec import DEFAULT_SPEC


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class JointDFlowConfig:
    steps: int = 80
    # ``learning_rate`` and ``gradient_clip`` are retained for manifest/CLI
    # compatibility with v2 campaigns; IPOPT does not use either quantity.
    learning_rate: float = 0.003
    ode_steps: int = 24
    integrator: str = "midpoint"
    gradient_clip: float = 10.0
    # Generated endpoints can initially lie tens of eV/atom above the MP hull.
    # This is a numerical scale only; the physical 0.150 threshold is unchanged.
    e_hull_scale_eV_atom: float = 10.0
    novelty_scale: float = 100.0
    novelty_weight: float = 0.10
    hull_threshold_eV_atom: float = 0.150
    hull_threshold_weight: float = 1.0
    hull_constraint_tolerance_eV_atom: float = 1.0e-6
    distance_weight: float = 40.0
    p_coordination_weight: float = 12.0
    fe_coordination_weight: float = 2.0
    volume_weight: float = 2.0
    source_prior_weight: float = 0.02
    composition_weight: float = 20.0
    atom_temperature: float = 0.35
    constraint_margin_A: float = 0.10
    reintegration_atol: float = 1.0e-6
    reintegration_rtol: float = 1.0e-5
    ipopt_tolerance: float = 1.0e-5
    ipopt_acceptable_tolerance: float = 1.0e-4
    ipopt_print_level: int = 0


@dataclass
class JointDFlowResult:
    z0_star: CrystalState
    terminal: CrystalState
    history: list[dict[str, Any]]
    status: str
    model_hash: str
    solver_stats: dict[str, Any]


class JointCrystalProblem:
    def __init__(
        self,
        model: CrystalVectorField,
        mask: torch.Tensor,
        energy_per_atom: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        mp_hull_reference_eV_atom: float,
        mp2020_candidate_correction_eV_atom: float,
        novelty_index: NoveltyIndex | None,
        fe_coordination_prior: tuple[float, float, float] | None = None,
        energy_factory: Callable[[torch.Tensor], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = None,
        differentiable_hull: DifferentiableMP2020Hull | None = None,
        atomic_number_lookup: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.mask = mask
        self.energy_per_atom = energy_per_atom
        self.mp_hull_reference_eV_atom = float(mp_hull_reference_eV_atom)
        self.mp2020_candidate_correction_eV_atom = float(
            mp2020_candidate_correction_eV_atom
        )
        self.novelty_index = novelty_index
        self.fe_coordination_prior = fe_coordination_prior
        self.energy_factory = energy_factory
        self.differentiable_hull = differentiable_hull
        self.atomic_number_lookup = atomic_number_lookup

    @property
    def variable_composition(self) -> bool:
        return (
            self.energy_factory is not None
            and self.differentiable_hull is not None
            and self.atomic_number_lookup is not None
        )

    def values(self, terminal: CrystalState) -> dict[str, torch.Tensor]:
        types, frac, lattice = decode_terminal(self.model, terminal, self.mask)
        probabilities = torch.softmax(terminal.atom / 0.35, dim=-1)
        probabilities = probabilities * self.mask.unsqueeze(-1)
        if self.variable_composition:
            hard_offsets = terminal.atom.argmax(dim=-1)
            numbers = self.atomic_number_lookup.to(hard_offsets)[hard_offsets[self.mask]]
            energy = self.energy_factory(numbers)(frac, lattice).mean()
            hard_probabilities = F.one_hot(
                hard_offsets, num_classes=terminal.atom.shape[-1]
            ).to(terminal.atom) * self.mask.unsqueeze(-1)
            # Straight-through composition: the forward value remains the
            # exact hard-composition MP2020 hull, while backward uses the
            # analytic active-facet gradient through soft composition and z_A.
            hull_probabilities = (
                hard_probabilities.detach()
                + probabilities
                - probabilities.detach()
            )
            raw_hull = self.differentiable_hull.raw_e_hull(
                energy.reshape(1), hull_probabilities, self.mask
            ).mean()
        else:
            energy = self.energy_per_atom(frac, lattice).mean()
            raw_hull = (
                energy + self.mp2020_candidate_correction_eV_atom
                - self.mp_hull_reference_eV_atom
            )
        e_hull = raw_hull.clamp_min(0.0)
        if self.novelty_index is None:
            novelty = frac[self.mask].square().mean().sqrt()
        elif self.variable_composition:
            descriptor = soft_crystal_descriptor(
                probabilities, frac, lattice, self.mask
            )
            novelty = self.novelty_index.smooth_score(descriptor).mean()
        else:
            descriptor = crystal_descriptor(types, frac, lattice, self.mask)
            novelty = self.novelty_index.smooth_score(descriptor).mean()
        return {
            "E_UMA": energy,
            "raw_E_hull": raw_hull,
            "E_hull": e_hull,
            "novelty": novelty,
        }

    def constraints(
        self, terminal: CrystalState, margin_A: float
    ) -> dict[str, torch.Tensor]:
        if self.variable_composition:
            terms = soft_constraint_terms(
                self.model, terminal, self.mask,
                temperature=0.35,
            )
        else:
            terms = terminal_constraint_terms(
                self.model,
                terminal,
                self.mask,
                margin_A=margin_A,
                fe_coordination_prior=self.fe_coordination_prior,
            )
        physical = unstandardize_lattice(
            terminal.lattice, self.model.lattice_mean, self.model.lattice_std
        )
        lattice = decode_lattice(physical, self.mask.sum(dim=1))
        volume_per_atom = torch.linalg.det(lattice) / self.mask.sum(dim=1).to(
            lattice
        )
        volume = (
            F.relu(DEFAULT_SPEC.volume_per_atom_min_A3 - volume_per_atom).square()
            + F.relu(
                volume_per_atom - DEFAULT_SPEC.volume_per_atom_max_A3
            ).square()
        ).mean()
        result = {
            "distance": terms["distance"],
            "p_coordination": terms["p_coordination"],
            "fe_coordination": terms["fe_coordination"],
            "volume": volume,
        }
        if self.variable_composition:
            probabilities = torch.softmax(terminal.atom / 0.35, dim=-1)
            rows = soft_composition_terms(probabilities, self.mask)
            result["composition"] = sum(rows.values())
        return result


class JointDFlowSolver:
    def __init__(self, model: CrystalVectorField, problem: JointCrystalProblem) -> None:
        self.model = model
        self.problem = problem

    def solve(
        self,
        z0: CrystalState,
        config: JointDFlowConfig = JointDFlowConfig(),
    ) -> JointDFlowResult:
        if config.steps <= 0:
            raise ValueError("joint D-Flow requires at least one optimization step")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        before_hash = model_sha256(self.model)
        try:
            import casadi as ca
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "CasADi is required for JointDFlowSolver; install casadi in "
                "the active NaGen environment"
            ) from error

        reference_atom = z0.atom.detach().clone()
        reference_frac = z0.frac.detach().clone()
        reference_lattice = z0.lattice.detach().clone()
        optimize_atom = self.problem.variable_composition
        atom = reference_atom
        history: list[dict[str, Any]] = []
        best_merit = float("inf")
        best_selection_key = (1, float("inf"), float("inf"))
        best_hull_constraint = float("inf")
        best_atom = reference_atom.clone()
        best_frac = reference_frac.clone()
        best_lattice = reference_lattice.clone()
        atom_numel = reference_atom.numel() if optimize_atom else 0
        frac_numel = reference_frac.numel()
        lattice_numel = reference_lattice.numel()
        variable_count = atom_numel + frac_numel + lattice_numel

        def evaluate(vector: Any) -> tuple[float, Any, float, Any]:
            nonlocal best_merit, best_selection_key, best_hull_constraint
            nonlocal best_atom, best_frac, best_lattice
            flat = torch.as_tensor(
                np.asarray(vector, dtype=np.float64).reshape(-1),
                device=reference_frac.device,
                dtype=reference_frac.dtype,
            )
            offset = 0
            if optimize_atom:
                atom = flat[:atom_numel].reshape_as(reference_atom).clone().requires_grad_(True)
                offset = atom_numel
            else:
                atom = reference_atom
            frac = flat[offset:offset + frac_numel].reshape_as(reference_frac).clone().requires_grad_(True)
            lattice = flat[offset + frac_numel:].reshape_as(reference_lattice).clone().requires_grad_(True)
            source = CrystalState(atom, frac, lattice)
            terminal = integrate_flow(
                self.model,
                source,
                self.problem.mask,
                steps=config.ode_steps,
                method=config.integrator,
                freeze_atom=not optimize_atom,
            )
            values = self.problem.values(terminal)
            constraints = self.problem.constraints(
                terminal, config.constraint_margin_A
            )
            hull_scaled = values["E_hull"] / max(
                config.e_hull_scale_eV_atom, 1.0e-12
            )
            hull_violation = F.relu(
                values["E_hull"] - config.hull_threshold_eV_atom
            ) / max(config.e_hull_scale_eV_atom, 1.0e-12)
            novelty_scaled = values["novelty"] / max(
                config.novelty_scale, 1.0e-12
            )
            circular_delta = frac - reference_frac
            circular_delta = circular_delta - torch.floor(circular_delta + 0.5)
            source_drift = (
                circular_delta.square().mean()
                + (lattice - reference_lattice).square().mean()
            )
            if optimize_atom:
                source_drift = source_drift + (
                    atom - reference_atom
                ).square().mean()
            merit = (
                hull_scaled
                + config.hull_threshold_weight * hull_violation.square()
                - config.novelty_weight * novelty_scaled
                + config.distance_weight * constraints["distance"]
                + config.p_coordination_weight * constraints["p_coordination"]
                + config.fe_coordination_weight * constraints["fe_coordination"]
                + config.volume_weight * constraints["volume"]
                + config.composition_weight * constraints.get(
                    "composition", hull_scaled.new_zeros(())
                )
                + config.source_prior_weight * source_drift
            )
            # The formal IPOPT inequality uses raw_E_hull.  Because the upper
            # bound is positive, raw_E_hull <= 0.150 is exactly equivalent to
            # max(0, raw_E_hull) <= 0.150, while avoiding the zero derivative
            # introduced by clamp_min below the physical hull.
            hull_constraint = values["raw_E_hull"]
            hull_scale = max(config.e_hull_scale_eV_atom, 1.0e-12)
            # This is an algebraically equivalent O(1)-scaled form of the
            # physical hard constraint.  Reports always retain eV/atom.
            ipopt_hull_constraint = (
                hull_constraint - config.hull_threshold_eV_atom
            ) / hull_scale
            variables = (atom, frac, lattice) if optimize_atom else (frac, lattice)
            gradients = torch.autograd.grad(merit, variables, retain_graph=True)
            hull_gradients = torch.autograd.grad(
                ipopt_hull_constraint, variables, allow_unused=True
            )
            hull_gradients = tuple(
                torch.zeros_like(variable) if gradient is None else gradient
                for variable, gradient in zip(variables, hull_gradients, strict=True)
            )
            gradient = torch.cat([item.reshape(-1) for item in gradients])
            hull_gradient = torch.cat([
                item.reshape(-1) for item in hull_gradients
            ])
            gradient_norm = torch.linalg.vector_norm(gradient)
            ipopt_hull_gradient_norm = torch.linalg.vector_norm(hull_gradient)
            hull_gradient_norm = ipopt_hull_gradient_norm * hull_scale
            if not bool(
                torch.isfinite(merit)
                and torch.isfinite(hull_constraint)
                and torch.isfinite(ipopt_hull_constraint)
                and torch.isfinite(gradient).all()
                and torch.isfinite(hull_gradient).all()
            ):
                raise FloatingPointError("non-finite joint D-Flow objective/gradient")
            with torch.no_grad():
                merit_value = float(merit)
                hull_constraint_value = float(hull_constraint)
                hull_excess = max(
                    hull_constraint_value - config.hull_threshold_eV_atom, 0.0
                )
                hull_feasible = (
                    hull_excess <= config.hull_constraint_tolerance_eV_atom
                )
                selection_key = (
                    0 if hull_feasible else 1,
                    0.0 if hull_feasible else hull_excess,
                    merit_value,
                )
                if selection_key < best_selection_key:
                    best_selection_key = selection_key
                    best_merit = merit_value
                    best_hull_constraint = hull_constraint_value
                    if optimize_atom:
                        best_atom = atom.detach().clone()
                    best_frac = frac.detach().clone()
                    best_lattice = lattice.detach().clone()
            history.append({
                "evaluation": len(history),
                "merit": float(merit.detach()),
                "E_UMA": float(values["E_UMA"].detach()),
                "raw_E_hull": float(values["raw_E_hull"].detach()),
                "E_hull": float(values["E_hull"].detach()),
                "hull_constraint_value_eV_atom": hull_constraint_value,
                "hull_constraint_residual_eV_atom": (
                    hull_constraint_value - config.hull_threshold_eV_atom
                ),
                "hull_constraint_satisfied": hull_feasible,
                "ipopt_hull_constraint_scaled": float(
                    ipopt_hull_constraint.detach()
                ),
                "novelty": float(values["novelty"].detach()),
                "hull_threshold_violation": float(hull_violation.detach()),
                "source_drift": float(source_drift.detach().sqrt()),
                "gradient_norm": float(gradient_norm.detach()),
                "hull_constraint_gradient_norm": float(
                    hull_gradient_norm.detach()
                ),
                "ipopt_hull_constraint_scaled_gradient_norm": float(
                    ipopt_hull_gradient_norm.detach()
                ),
                "constraints": {
                    name: float(value.detach())
                    for name, value in constraints.items()
                },
            })
            return (
                merit_value,
                gradient.detach().cpu().double().numpy(),
                hull_constraint_value,
                float(ipopt_hull_constraint.detach()),
                hull_gradient.detach().cpu().double().numpy(),
            )

        cached_x = None
        cached_evaluation = None

        def cached_evaluate(argument: Any) -> tuple[float, Any, float, float, Any]:
            nonlocal cached_x, cached_evaluation
            vector = np.asarray(argument, dtype=np.float64).reshape(-1)
            if cached_x is None or not np.array_equal(vector, cached_x):
                cached_evaluation = evaluate(vector)
                cached_x = vector.copy()
            return cached_evaluation

        class TorchObjective(ca.Callback):
            def __init__(self, name: str) -> None:
                ca.Callback.__init__(self)
                self.jacobian_callback = None
                self.construct(name)

            def get_n_in(self) -> int:
                return 1

            def get_n_out(self) -> int:
                return 1

            def get_sparsity_in(self, index: int):
                return ca.Sparsity.dense(variable_count, 1)

            def get_sparsity_out(self, index: int):
                return ca.Sparsity.scalar()

            def eval(self, arguments: list[Any]):
                value, _, _, _, _ = cached_evaluate(arguments[0])
                return [value]

            def has_jacobian(self) -> bool:
                return True

            def get_jacobian(self, name, inames, onames, opts):
                self.jacobian_callback = TorchObjectiveJacobian(name, self, opts)
                return self.jacobian_callback

        class TorchObjectiveJacobian(ca.Callback):
            def __init__(self, name: str, parent: TorchObjective, opts) -> None:
                ca.Callback.__init__(self)
                self.parent = parent
                self.construct(name, opts)

            def get_n_in(self) -> int:
                # Nominal input and nominal scalar output, per CasADi's
                # user-supplied full-Jacobian callback contract.
                return 2

            def get_n_out(self) -> int:
                return 1

            def get_sparsity_in(self, index: int):
                if index == 0:
                    return ca.Sparsity.dense(variable_count, 1)
                return ca.Sparsity.scalar()

            def get_sparsity_out(self, index: int):
                return ca.Sparsity.dense(1, variable_count)

            def eval(self, arguments: list[Any]):
                _, gradient, _, _, _ = cached_evaluate(arguments[0])
                return [ca.DM(gradient).T]

        class TorchHullConstraint(ca.Callback):
            def __init__(self, name: str) -> None:
                ca.Callback.__init__(self)
                self.jacobian_callback = None
                self.construct(name)

            def get_n_in(self) -> int:
                return 1

            def get_n_out(self) -> int:
                return 1

            def get_sparsity_in(self, index: int):
                return ca.Sparsity.dense(variable_count, 1)

            def get_sparsity_out(self, index: int):
                return ca.Sparsity.scalar()

            def eval(self, arguments: list[Any]):
                _, _, _, value, _ = cached_evaluate(arguments[0])
                return [value]

            def has_jacobian(self) -> bool:
                return True

            def get_jacobian(self, name, inames, onames, opts):
                self.jacobian_callback = TorchHullConstraintJacobian(
                    name, self, opts
                )
                return self.jacobian_callback

        class TorchHullConstraintJacobian(ca.Callback):
            def __init__(
                self, name: str, parent: TorchHullConstraint, opts
            ) -> None:
                ca.Callback.__init__(self)
                self.parent = parent
                self.construct(name, opts)

            def get_n_in(self) -> int:
                return 2

            def get_n_out(self) -> int:
                return 1

            def get_sparsity_in(self, index: int):
                if index == 0:
                    return ca.Sparsity.dense(variable_count, 1)
                return ca.Sparsity.scalar()

            def get_sparsity_out(self, index: int):
                return ca.Sparsity.dense(1, variable_count)

            def eval(self, arguments: list[Any]):
                _, _, _, _, gradient = cached_evaluate(arguments[0])
                return [ca.DM(gradient).T]

        objective = TorchObjective("joint_dflow_torch_objective")
        hull_constraint = TorchHullConstraint("joint_dflow_hull_constraint")
        symbolic_x = ca.MX.sym("source_noise", variable_count)
        nlp = {
            "x": symbolic_x,
            "f": objective(symbolic_x),
            "g": hull_constraint(symbolic_x),
        }
        options = {
            "print_time": False,
            "ipopt.print_level": config.ipopt_print_level,
            "ipopt.max_iter": config.steps,
            "ipopt.tol": config.ipopt_tolerance,
            "ipopt.acceptable_tol": config.ipopt_acceptable_tolerance,
            # PyTorch supplies exact first derivatives. Limited-memory avoids
            # differentiating through the Python callback a second time.
            "ipopt.hessian_approximation": "limited-memory",
        }
        solver = ca.nlpsol("joint_dflow_ipopt", "ipopt", nlp, options)
        x0 = torch.cat([
            *([reference_atom.reshape(-1)] if optimize_atom else []),
            reference_frac.reshape(-1), reference_lattice.reshape(-1)
        ]).detach().cpu().double().numpy()
        lower = np.concatenate([
            *([np.full(atom_numel, -8.0)] if optimize_atom else []),
            np.zeros(frac_numel), np.full(lattice_numel, -6.0)
        ])
        upper = np.concatenate([
            *([np.full(atom_numel, 8.0)] if optimize_atom else []),
            np.ones(frac_numel), np.full(lattice_numel, 6.0)
        ])
        solution = solver(
            x0=x0,
            lbx=lower,
            ubx=upper,
            lbg=-np.inf,
            ubg=0.0,
        )
        solver_stats = dict(solver.stats())
        return_status = str(solver_stats.get("return_status", "unknown"))
        finite_solution = np.isfinite(np.asarray(solution["x"])).all()
        accepted = {
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        }
        status = "complete" if return_status in accepted and finite_solution else return_status

        # Include the returned point in best-point selection even if IPOPT did
        # not request a final objective evaluation at exactly this vector.
        solution_evaluation = evaluate(np.asarray(solution["x"]).reshape(-1))
        solution_hull = float(solution_evaluation[2])
        z0_star = CrystalState(best_atom, best_frac, best_lattice)
        selected_hull = best_hull_constraint
        if (
            selected_hull
            > config.hull_threshold_eV_atom
            + config.hull_constraint_tolerance_eV_atom
        ):
            status = "hull_constraint_violated"
        with torch.no_grad():
            terminal = integrate_flow(
                self.model, z0_star, self.problem.mask,
                steps=config.ode_steps, method=config.integrator,
                freeze_atom=not optimize_atom,
            )
            repeated = integrate_flow(
                self.model, z0_star, self.problem.mask,
                steps=config.ode_steps, method=config.integrator,
                freeze_atom=not optimize_atom,
            )
        for first, second in zip(terminal, repeated, strict=True):
            if not torch.allclose(
                first, second,
                atol=config.reintegration_atol,
                rtol=config.reintegration_rtol,
            ):
                status = "reintegration_mismatch"
        if before_hash != model_sha256(self.model):
            raise RuntimeError("Flow parameters changed during D-Flow optimization")
        serializable_stats = {
            key: value for key, value in solver_stats.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        serializable_stats["return_status"] = return_status
        serializable_stats["objective_evaluations"] = len(history)
        serializable_stats["hull_constraint_upper_eV_atom"] = (
            config.hull_threshold_eV_atom
        )
        serializable_stats["hull_constraint_scale_eV_atom"] = (
            config.e_hull_scale_eV_atom
        )
        serializable_stats["ipopt_scaled_hull_constraint_upper"] = 0.0
        serializable_stats["solution_hull_constraint_eV_atom"] = solution_hull
        serializable_stats["selected_hull_constraint_eV_atom"] = selected_hull
        serializable_stats["hull_constraint_satisfied"] = (
            selected_hull
            <= config.hull_threshold_eV_atom
            + config.hull_constraint_tolerance_eV_atom
        )
        serializable_stats["optimized_source_channels"] = (
            ["z_A", "z_X", "z_L"] if optimize_atom else ["z_X", "z_L"]
        )
        return JointDFlowResult(
            z0_star, terminal, history, status, before_hash, serializable_stats
        )

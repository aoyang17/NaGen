"""CasADi-IPOPT source-space optimization with PyTorch derivatives.

N and A are fixed after exact composition rejection.  IPOPT optimizes z_X and
z_L, while every value/Jacobian evaluation integrates the complete frozen Flow
and evaluates the surrogate, smooth geometry constraints, and UMA force/stress.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import casadi as ca
import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from nagen.inverse.constraints import evaluate_feasibility, minimum_distance
from nagen.inverse.generate import load_model, sample_feasible_compositions
from nagen.inverse.guidance import polyhedron_constraint_terms
from nagen.inverse.lattice import decode_lattice, unstandardize_lattice
from nagen.inverse.model import CrystalState
from nagen.inverse.novelty import crystal_descriptor
from nagen.inverse.sample import integrate_flow
from nagen.inverse.uma_guidance import load_uma_native_second_order_evaluator
from nagen.surrogate.ehull_mace import load_surrogate
from run_surrogate_shootingflow import (
    build_surrogate_graph_batch,
    constraint_penalty,
    reference_descriptors,
)


def smooth_logmeanexp(values: torch.Tensor, tau: float) -> torch.Tensor:
    return tau * (
        torch.logsumexp(values / tau, dim=0) - math.log(values.numel())
    )


class TorchNLP:
    def __init__(
        self, *, flow, surrogate, surrogate_ckpt, uma_evaluator,
        atom, mask, elements, initial_frac, initial_lattice,
        novelty_types, ref_desc, ref_mean, ref_std, args,
    ) -> None:
        self.flow = flow
        self.surrogate = surrogate
        self.surrogate_ckpt = surrogate_ckpt
        self.uma_evaluator = uma_evaluator
        self.atom = atom
        self.mask = mask
        self.elements = elements
        self.initial_frac = initial_frac
        self.initial_lattice = initial_lattice
        self.novelty_types = novelty_types
        self.ref_desc = ref_desc
        self.ref_mean = ref_mean
        self.ref_std = ref_std
        self.args = args
        self.n_atoms = len(elements)
        self.n_variables = 3 * self.n_atoms + 6
        self.n_outputs = 14  # objective followed by 13 explicit constraints
        self.cache_x: np.ndarray | None = None
        self.cache_values: np.ndarray | None = None
        self.cache_jacobian: np.ndarray | None = None
        self.last_metrics: dict[str, float] = {}
        self.evaluations = 0
        self.jacobian_evaluations = 0

        thresholds = []
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                thresholds.append(minimum_distance(elements[i], elements[j]))
        self.pair_thresholds = torch.tensor(
            thresholds, device=atom.device, dtype=atom.dtype
        )

    def initial_vector(self) -> np.ndarray:
        return np.concatenate((
            self.initial_frac.detach().cpu().numpy().reshape(-1),
            self.initial_lattice.detach().cpu().numpy().reshape(-1),
        )).astype(np.float64)

    def _tensors(self, vector: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        device, dtype = self.atom.device, self.atom.dtype
        tensor = torch.as_tensor(vector, device=device, dtype=dtype)
        frac = tensor[: 3 * self.n_atoms].reshape(1, self.n_atoms, 3)
        lattice = tensor[3 * self.n_atoms :].reshape(1, 6)
        return frac.requires_grad_(True), lattice.requires_grad_(True)

    def _values(
        self,
        vector: np.ndarray,
        z_frac: torch.Tensor | None = None,
        z_lattice: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]
    ]:
        if z_frac is None or z_lattice is None:
            z_frac, z_lattice = self._tensors(vector)
        terminal = integrate_flow(
            self.flow, CrystalState(self.atom, z_frac, z_lattice), self.mask,
            steps=self.args.ode_steps, method="midpoint", freeze_atom=True,
        )
        lattice = decode_lattice(
            unstandardize_lattice(
                terminal.lattice, self.flow.lattice_mean, self.flow.lattice_std,
            ),
            self.mask.sum(dim=1),
        )
        graph = build_surrogate_graph_batch(
            [self.elements], terminal.frac, lattice,
            self.surrogate_ckpt["element_to_index"],
            self.surrogate.config.cutoff_A, self.surrogate.config.max_neighbors,
        )
        graph = {key: value.to(z_frac.device) for key, value in graph.items()}
        graph["frac"] = terminal.frac[self.mask]
        graph["lattice"] = lattice
        prediction = self.surrogate(graph).reshape(())

        descriptor = crystal_descriptor(
            self.novelty_types, terminal.frac, lattice, self.mask,
        )
        novelty = (
            torch.cdist(
                (descriptor - self.ref_mean) / self.ref_std,
                (self.ref_desc - self.ref_mean) / self.ref_std,
            ).amin() / descriptor.shape[-1] ** 0.5
        )
        objective = (
            self.args.property_energy_weight
            * (prediction / self.args.energy_scale).square()
            + self.args.property_novelty_weight
            * ((self.args.novelty_target - novelty) / self.args.novelty_scale).square()
            + self.args.source_prior_weight * (
                (z_frac - self.initial_frac).square().mean()
                + (z_lattice - self.initial_lattice).square().mean()
            )
        )

        shifts = torch.tensor(
            [(i, j, k) for i in (-1.0, 0.0, 1.0)
             for j in (-1.0, 0.0, 1.0) for k in (-1.0, 0.0, 1.0)],
            device=z_frac.device, dtype=z_frac.dtype,
        )
        delta = terminal.frac[0, :, None, :] - terminal.frac[0, None, :, :]
        cart = torch.einsum("sijd,dk->sijk", delta[None] + shifts[:, None, None], lattice[0])
        distances = -self.args.tau_distance * torch.logsumexp(
            -torch.linalg.vector_norm(cart, dim=-1) / self.args.tau_distance,
            dim=0,
        )
        upper = torch.triu(
            torch.ones(self.n_atoms, self.n_atoms, dtype=torch.bool, device=z_frac.device),
            diagonal=1,
        )
        pair_violations = self.pair_thresholds - distances[upper]
        distance_constraint = smooth_logmeanexp(pair_violations, 0.05) / 0.1

        volume_per_atom = torch.linalg.det(lattice[0]) / self.n_atoms
        p_indices = [i for i, value in enumerate(self.elements) if value == "P"]
        fe_indices = [i for i, value in enumerate(self.elements) if value == "Fe"]
        oxygen = torch.tensor(
            [value == "O" for value in self.elements], device=z_frac.device
        )
        p_counts = torch.stack([
            torch.sigmoid((2.0 - distances[i, oxygen]) / self.args.tau_coord).sum()
            for i in p_indices
        ])
        fe_counts = torch.stack([
            torch.sigmoid((2.5 - distances[i, oxygen]) / self.args.tau_coord).sum()
            for i in fe_indices
        ])
        p_constraint = smooth_logmeanexp((p_counts - 4.0).abs(), 0.1) - 0.25
        fe_lower = smooth_logmeanexp(4.0 - fe_counts, 0.1) - 0.25
        fe_upper = smooth_logmeanexp(fe_counts - 6.0, 0.1) - 0.25

        poly = polyhedron_constraint_terms(
            self.flow, terminal, self.mask,
        )
        atomic_numbers = torch.tensor(
            [{"Na": 11, "Fe": 26, "P": 15, "O": 8}[e] for e in self.elements],
            dtype=torch.long, device=z_frac.device,
        )
        _, forces, stress = self.uma_evaluator(
            atomic_numbers, terminal.frac, lattice,
        )
        fmax = torch.linalg.vector_norm(forces, dim=-1).amax()
        stress_fro = torch.linalg.vector_norm(stress.reshape(-1), ord=2)
        stress_tol = self.args.stress_tolerance_gpa / 160.21766208

        constraints = torch.stack((
            (prediction - self.args.threshold) / self.args.energy_scale,
            distance_constraint,
            (10.5 - volume_per_atom) / 5.0,
            (volume_per_atom - 20.5) / 5.0,
            p_constraint,
            fe_lower,
            fe_upper,
            poly["poly_center"] - self.args.poly_tolerance,
            poly["poly_face"] - self.args.poly_tolerance,
            poly["poly_coplanar"] - self.args.poly_tolerance,
            (fmax - self.args.force_tolerance) / 0.1,
            (stress_fro - stress_tol) / 0.001,
            constraint_penalty(
                self.elements, terminal.frac[0], lattice,
                self.args.tau_distance, self.args.tau_coord,
            ) - self.args.aggregate_geometry_tolerance,
        ))
        metrics = {
            "surrogate_e_hull": float(prediction.detach()),
            "novelty": float(novelty.detach()),
            "volume_per_atom": float(volume_per_atom.detach()),
            "soft_min_distance_constraint": float(distance_constraint.detach()),
            "p_coordination_constraint": float(p_constraint.detach()),
            "fe_lower_constraint": float(fe_lower.detach()),
            "fe_upper_constraint": float(fe_upper.detach()),
            "fmax_eV_A": float(fmax.detach()),
            "stress_fro_eV_A3": float(stress_fro.detach()),
        }
        return objective, constraints, terminal.frac, lattice, metrics

    def evaluate(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if self.cache_x is not None and np.array_equal(vector, self.cache_x):
            assert self.cache_values is not None and self.cache_jacobian is not None
            return self.cache_values, self.cache_jacobian
        z_frac, z_lattice = self._tensors(vector)
        objective, constraints, _, _, metrics = self._values(
            vector, z_frac=z_frac, z_lattice=z_lattice,
        )
        outputs = torch.cat((objective.reshape(1), constraints))
        rows = []
        for index, value in enumerate(outputs):
            gradients = torch.autograd.grad(
                value, (z_frac, z_lattice),
                retain_graph=index + 1 < outputs.numel(), allow_unused=False,
            )
            rows.append(torch.cat((gradients[0].reshape(-1), gradients[1].reshape(-1))))
        values = outputs.detach().double().cpu().numpy()
        jacobian = torch.stack(rows).detach().double().cpu().numpy()
        self.cache_x = vector.copy()
        self.cache_values = values
        self.cache_jacobian = jacobian
        self.last_metrics = metrics
        self.evaluations += 1
        self.jacobian_evaluations += 1
        return values, jacobian


class TorchJacobianCallback(ca.Callback):
    def __init__(self, name: str, problem: TorchNLP) -> None:
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
        _, jacobian = self.problem.evaluate(np.asarray(arguments[0]).reshape(-1))
        return [ca.DM(jacobian)]


class TorchVectorCallback(ca.Callback):
    def __init__(self, name: str, problem: TorchNLP) -> None:
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
        values, _ = self.problem.evaluate(np.asarray(arguments[0]).reshape(-1))
        return [ca.DM(values)]
    def has_jacobian(self): return True
    def get_jacobian(self, name, inames, onames, opts):
        self.jacobian_callback = TorchJacobianCallback(name, self.problem)
        return self.jacobian_callback


def select_source(flow, atom, mask, elements, generator, args):
    device = atom.device
    count, n_atoms = args.source_candidates, len(elements)
    frac = torch.rand(count, n_atoms, 3, device=device, generator=generator)
    lattice = torch.randn(count, 6, device=device, generator=generator)
    repeated_atom = atom.expand(count, -1, -1)
    repeated_mask = mask.expand(count, -1)
    with torch.no_grad():
        terminal = integrate_flow(
            flow, CrystalState(repeated_atom, frac, lattice), repeated_mask,
            steps=args.ode_steps, method="midpoint", freeze_atom=True,
        )
        physical = decode_lattice(
            unstandardize_lattice(
                terminal.lattice, flow.lattice_mean, flow.lattice_std,
            ), repeated_mask.sum(dim=1),
        )
        scores = torch.stack([
            constraint_penalty(
                elements, terminal.frac[index], physical[index:index + 1],
                args.tau_distance, args.tau_coord,
            )
            for index in range(count)
        ])
    selected = int(scores.argmin())
    return frac[selected:selected + 1], lattice[selected:selected + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", required=True)
    parser.add_argument("--surrogate", required=True)
    parser.add_argument("--packed-data", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--start-sample", type=int, default=0,
        help="Advance deterministic composition/source streams but optimize only this index onward.",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-candidates", type=int, default=8)
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--ipopt-max-iterations", type=int, default=16)
    parser.add_argument("--max-wall-time", type=float, default=1140.0)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--energy-scale", type=float, default=0.15)
    parser.add_argument("--novelty-target", type=float, default=1.0)
    parser.add_argument("--novelty-scale", type=float, default=1.0)
    parser.add_argument("--property-energy-weight", type=float, default=0.8)
    parser.add_argument("--property-novelty-weight", type=float, default=0.2)
    parser.add_argument("--source-prior-weight", type=float, default=0.01)
    parser.add_argument("--tau-distance", type=float, default=0.08)
    parser.add_argument("--tau-coord", type=float, default=0.08)
    parser.add_argument("--force-tolerance", type=float, default=0.01)
    parser.add_argument("--stress-tolerance-gpa", type=float, default=0.10)
    parser.add_argument("--poly-tolerance", type=float, default=0.01)
    parser.add_argument("--aggregate-geometry-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    flow, flow_ckpt = load_model(args.flow, device, use_ema=True)
    surrogate, surrogate_ckpt = load_surrogate(args.surrogate, device)
    ref_desc, ref_mean, ref_std = reference_descriptors(args.packed_data, 0, device)
    uma_evaluator, uma_metadata = load_uma_native_second_order_evaluator(
        args.uma_checkpoint, device=device, task_name="omat",
    )
    for model in (flow, surrogate):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    types, masks = sample_feasible_compositions(
        flow, flow_ckpt, args.count, generator, args.ode_steps, "midpoint",
        pool_factor=512, max_rounds=60, max_atoms=184,
        proposal_batch_limit=512,
    )
    inverse = {int(value): key for key, value in flow_ckpt["element_to_index"].items()}
    records = []
    for sample_id in range(args.count):
        row_types = types[sample_id, masks[sample_id]]
        elements = [inverse[int(value)] for value in row_types.tolist()]
        n_atoms = len(elements)
        row_mask = torch.ones(1, n_atoms, dtype=torch.bool, device=device)
        atom = torch.nn.functional.one_hot(
            row_types - 1, num_classes=flow.config.vocab_size,
        ).float().unsqueeze(0) * 2.0
        # Mirror the L-BFGS runner's initially allocated source exactly.  With
        # one sample per process this leaves both solvers at the same RNG state
        # before screening the shared source-candidate pool.
        torch.rand(1, n_atoms, 3, device=device, generator=generator)
        torch.randn(1, 6, device=device, generator=generator)
        initial_frac, initial_lattice = select_source(
            flow, atom, row_mask, elements, generator, args,
        )
        if sample_id < args.start_sample:
            continue
        novelty_map = {"Na": 1, "Fe": 2, "P": 3, "O": 4}
        novelty_types = torch.tensor(
            [[novelty_map[element] for element in elements]],
            dtype=torch.long, device=device,
        )
        problem = TorchNLP(
            flow=flow, surrogate=surrogate, surrogate_ckpt=surrogate_ckpt,
            uma_evaluator=uma_evaluator, atom=atom, mask=row_mask,
            elements=elements, initial_frac=initial_frac,
            initial_lattice=initial_lattice, novelty_types=novelty_types,
            ref_desc=ref_desc, ref_mean=ref_mean, ref_std=ref_std, args=args,
        )
        callback = TorchVectorCallback(f"torch_nlp_{sample_id}", problem)
        x_symbol = ca.MX.sym("x", problem.n_variables)
        output = callback(x_symbol)
        nlp = {"x": x_symbol, "f": output[0], "g": output[1:]}
        remaining = max(1.0, args.max_wall_time - (time.monotonic() - started))
        options = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.max_iter": args.ipopt_max_iterations,
            "ipopt.max_wall_time": remaining,
            "ipopt.tol": 1.0e-4,
            "ipopt.acceptable_tol": 1.0e-3,
            "ipopt.acceptable_iter": 3,
        }
        solver = ca.nlpsol(f"ipopt_{sample_id}", "ipopt", nlp, options)
        x0 = problem.initial_vector()
        lower = np.concatenate((np.zeros(3 * n_atoms), np.full(6, -5.0)))
        upper_bound = np.concatenate((np.ones(3 * n_atoms), np.full(6, 5.0)))
        sample_started = time.monotonic()
        try:
            solution = solver(
                x0=x0, lbx=lower, ubx=upper_bound,
                lbg=np.full(problem.n_outputs - 1, -np.inf),
                ubg=np.zeros(problem.n_outputs - 1),
            )
            optimized = np.asarray(solution["x"]).reshape(-1)
            status = solver.stats().get("return_status", "unknown")
        except Exception as error:
            optimized = x0
            status = f"exception:{type(error).__name__}:{error}"
        z_frac, z_lattice = problem._tensors(optimized)
        with torch.no_grad():
            terminal = integrate_flow(
                flow, CrystalState(atom, z_frac, z_lattice), row_mask,
                steps=args.ode_steps, method="midpoint", freeze_atom=True,
            )
            lattice = decode_lattice(
                unstandardize_lattice(
                    terminal.lattice, flow.lattice_mean, flow.lattice_std,
                ), row_mask.sum(dim=1),
            )
        values, _ = problem.evaluate(optimized)
        prediction = float(problem.last_metrics["surrogate_e_hull"])
        structure = Structure(
            Lattice(lattice[0].detach().cpu().numpy()), elements,
            terminal.frac[0].detach().cpu().numpy(),
        )
        analytic_checks = evaluate_feasibility(
            elements, terminal.frac[0].detach().cpu().numpy(),
            lattice[0].detach().cpu().numpy(), prediction,
        )
        explicit_constraints_feasible = bool(np.max(values[1:]) <= 1.0e-3)
        record = {
            "sample_id": sample_id, "solver": "casadi-ipopt",
            "status": status, "N": n_atoms, "formula": structure.formula,
            "wall_time_seconds": time.monotonic() - sample_started,
            "function_evaluations": problem.evaluations,
            "jacobian_evaluations": problem.jacobian_evaluations,
            "optimized_e_hull_surrogate": prediction,
            "passes_threshold": prediction <= args.threshold,
            "final_constraint_vector": values[1:].tolist(),
            "maximum_constraint_violation": float(
                np.maximum(values[1:], 0.0).max(initial=0.0)
            ),
            "explicit_constraints_feasible": explicit_constraints_feasible,
            "final_uma_fmax_eV_A": problem.last_metrics["fmax_eV_A"],
            "final_uma_stress_fro_eV_A3": problem.last_metrics["stress_fro_eV_A3"],
            "optimized_novelty": problem.last_metrics["novelty"],
            "analytic_constraint_checks": analytic_checks,
            "all_constraints_feasible": bool(
                analytic_checks["feasible"] and explicit_constraints_feasible
            ),
            "elements": elements,
            "frac_coords": terminal.frac[0].detach().cpu().tolist(),
            "lattice": lattice[0].detach().cpu().tolist(),
        }
        records.append(record)
        print(json.dumps({
            key: record[key] for key in (
                "sample_id", "status", "N", "formula", "wall_time_seconds",
                "maximum_constraint_violation", "all_constraints_feasible",
            )
        }, ensure_ascii=False), flush=True)
        if time.monotonic() - started >= args.max_wall_time:
            break
    summary = {
        "version": "nagen-casadi-ipopt-comparison-v1",
        "solver": "casadi-ipopt",
        "hessian_approximation": "limited-memory",
        "uma_metadata": uma_metadata,
        "count": len(records),
        "exact_feasible": sum(r["analytic_constraint_checks"]["feasible"] for r in records),
        "all_constraints_feasible": sum(r["all_constraints_feasible"] for r in records),
        "wall_time_seconds": time.monotonic() - started,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({key: summary[key] for key in ("count", "exact_feasible", "wall_time_seconds")}), flush=True)


if __name__ == "__main__":
    main()

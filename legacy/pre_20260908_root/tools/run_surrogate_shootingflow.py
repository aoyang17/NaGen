"""ShootingFlow experiment with a frozen unconditional Flow and E_hull surrogate.

The experiment fixes N and A for each start and optimizes only continuous source
variables (z_X, z_L). No UMA relaxation or true E_hull evaluation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from nagen.inverse.generate import load_model, sample_feasible_compositions
from nagen.inverse.lattice import decode_lattice, unstandardize_lattice
from nagen.inverse.model import CrystalState
from nagen.inverse.sample import integrate_flow
from nagen.inverse.constraints import evaluate_feasibility
from nagen.inverse.guidance import polyhedron_constraint_terms
from nagen.inverse.novelty import crystal_descriptor
from nagen.inverse.dataset import PackedCrystalDataset, collate_crystals
from nagen.surrogate.ehull_mace import (
    collate_graphs,
    load_surrogate,
    structure_to_model_batch,
)
from nagen.inverse.uma_guidance import load_uma_native_second_order_evaluator


def reference_descriptors(path: str, count: int, device: torch.device):
    packed = torch.load(path, map_location="cpu", weights_only=False)
    allowed = {int(packed["element_to_index"][e]) for e in ("Na", "Fe", "P", "O")}
    rows = [i for i in range(len(packed["ids"])) if set(int(x) for x in packed["types"][packed["offsets"][i]:packed["offsets"][i+1]]) <= allowed]
    if count > 0:
        rows = rows[:count]
    ds = PackedCrystalDataset(path, "train")
    ds.indices = rows
    remap = {int(packed["element_to_index"][e]) : j + 1 for j, e in enumerate(("Na", "Fe", "P", "O"))}
    descriptors = []
    for start in range(0, len(rows), 64):
        b = collate_crystals([
            ds[i] for i in range(start, min(start + 64, len(rows)))
        ])
        b["types"] = torch.tensor([
            [remap.get(int(x), 0) for x in row] for row in b["types"].tolist()
        ], dtype=torch.long)
        with torch.no_grad():
            descriptors.append(crystal_descriptor(
                b["types"].to(device), b["frac_coords"].to(device),
                b["lattice"].to(device), b["mask"].to(device),
            ))
    d = torch.cat(descriptors)
    return d, d.mean(0), d.std(0).clamp_min(1e-4)


def constraint_penalty_components(
    elements, frac, lattice, tau_distance=0.08, tau_coord=0.08,
):
    """Differentiable soft form of the frozen A5/A6 constraints.

    This is deliberately kept local to the surrogate shooting experiment: the
    exact acceptance gate remains ``evaluate_feasibility`` and therefore still
    comes from ``docs/HybridOptimization.md``/``inverse.constraints``.
    """
    n = frac.shape[0]
    shifts = torch.tensor([(i,j,k) for i in (-1.,0.,1.) for j in (-1.,0.,1.) for k in (-1.,0.,1.)], device=frac.device, dtype=frac.dtype)
    delta = frac[:,None,:] - frac[None,:,:] + shifts[:,None,None,:]
    cart = torch.einsum("sijd,dk->sijk", delta, lattice[0])
    distances = torch.linalg.vector_norm(cart, dim=-1)
    dist = -tau_distance * torch.logsumexp(-distances / tau_distance, dim=0)
    names = elements
    minima = {"P-O":1.40,"Fe-O":1.55,"Na-O":2.00,"O-O":2.00,"P-P":2.60,"Na-Na":2.20}
    pair_terms = []
    for i in range(n):
        for j in range(i+1,n):
            key = "-".join(sorted((names[i], names[j]), key=("Na","Fe","P","O").index))
            threshold = minima.get(key, 0.75 * ((1.66 if names[i]=="Na" else 1.32 if names[i]=="Fe" else 1.07 if names[i]=="P" else .66) + (1.66 if names[j]=="Na" else 1.32 if names[j]=="Fe" else 1.07 if names[j]=="P" else .66)))
            pair_terms.append(
                torch.nn.functional.softplus(
                    (frac.new_tensor(threshold) - dist[i, j]) / tau_distance
                ).square()
            )
    stacked_pairs = torch.stack(pair_terms)
    # Mean normalization removes the spurious O(N^2) dependence on cell size;
    # the max term keeps the worst collision from being diluted.
    pair_pen = stacked_pairs.mean() + stacked_pairs.max()
    volume = torch.linalg.det(lattice[0]).abs() / n
    volume_pen = (
        torch.relu(frac.new_tensor(10.5) - volume).div(5.0).square()
        + torch.relu(volume - frac.new_tensor(20.5)).div(5.0).square()
    )
    coord_terms = []
    for i, e in enumerate(names):
        if e not in ("P", "Fe"):
            continue
        cutoff = 2.0 if e == "P" else 2.5
        mask = torch.tensor([x == "O" for x in names], device=frac.device)
        ordered = torch.sort(dist[i][mask]).values
        # Exact coordination is represented by a shell boundary: the 4th
        # oxygen must be inside the bond cutoff and the next oxygen outside.
        if ordered.numel() >= 4:
            coord_terms.append(
                torch.nn.functional.softplus(
                    (ordered[3] - cutoff) / tau_coord
                ).square()
            )
        else:
            coord_terms.append(
                torch.nn.functional.softplus(
                    frac.new_tensor(4.0 - ordered.numel())
                ).square()
            )
        upper_index = 4 if e == "P" else 6
        if ordered.numel() > upper_index:
            coord_terms.append(
                torch.nn.functional.softplus(
                    (cutoff - ordered[upper_index]) / tau_coord
                ).square()
            )
    if coord_terms:
        stacked_coord = torch.stack(coord_terms)
        coord_pen = stacked_coord.mean() + stacked_coord.max()
    else:
        coord_pen = frac.new_zeros(())
    return {
        "distance": 10.0 * pair_pen,
        "volume": 0.5 * volume_pen,
        "coordination": 2.0 * coord_pen,
    }


def constraint_penalty(elements, frac, lattice, tau_distance=0.08, tau_coord=0.08):
    """Backward-compatible scalar sum of the differentiable components."""
    return torch.stack(tuple(constraint_penalty_components(
        elements, frac, lattice, tau_distance, tau_coord,
    ).values())).sum()


def po4_tetrahedron_penalty(
    elements, frac, lattice, margin=0.01, temperature=0.02, assignments=None,
):
    """Differentiable fixed-assignment proxy for P inside a PO4 tetrahedron.

    The four nearest O vertices are selected without gradients; the barycentric
    coordinates of P (the origin) are then evaluated differentiably.  A later
    exact PBC check is the acceptance criterion.
    """
    oxygen = torch.tensor([e == "O" for e in elements], device=frac.device)
    oxygen_indices = torch.where(oxygen)[0]
    if oxygen_indices.numel() < 4:
        return frac.new_tensor(1.0)
    shifts = torch.tensor(
        [(i, j, k) for i in (-1., 0., 1.) for j in (-1., 0., 1.)
         for k in (-1., 0., 1.)], device=frac.device, dtype=frac.dtype,
    )
    delta = frac[:, None, :] - frac[None, :, :] + shifts[:, None, None, :]
    vectors_all = torch.einsum("sijd,dk->sijk", delta, lattice[0])
    distances_all = torch.linalg.vector_norm(vectors_all, dim=-1)
    nearest = distances_all.detach().argmin(dim=0)
    vectors = vectors_all.gather(
        0, nearest[None, :, :, None].expand(1, *nearest.shape, 3)
    ).squeeze(0)
    distances = distances_all.gather(0, nearest[None, :, :]).squeeze(0)
    penalties = []
    for center, element in enumerate(elements):
        if element != "P":
            continue
        if assignments is not None:
            vertices = vectors[center, torch.as_tensor(
                assignments[center], device=frac.device, dtype=torch.long,
            )]
        else:
            selected = torch.topk(
                distances[center, oxygen_indices].detach(), k=4, largest=False,
            ).indices
            vertices = vectors[center, oxygen_indices[selected]]
        matrix = torch.cat((vertices.transpose(0, 1), vertices.new_ones(1, 4)), dim=0)
        rhs = vertices.new_tensor((0.0, 0.0, 0.0, 1.0))
        weights = torch.linalg.pinv(matrix) @ rhs
        barycentric = torch.nn.functional.softplus(
            (margin - weights) / temperature
        ).square().mean()
        volume = torch.linalg.det(
            torch.stack((vertices[1] - vertices[0], vertices[2] - vertices[0],
                         vertices[3] - vertices[0]), dim=1)
        ).abs() / 6.0
        nondegenerate = torch.nn.functional.softplus(
            (margin - volume) / temperature
        ).square()
        penalties.append(barycentric + nondegenerate)
    return torch.stack(penalties).mean() if penalties else frac.new_zeros(())


def po4_exact_checks(elements, frac, lattice, cutoff=2.0, tolerance=1.0e-7):
    """Exact PBC PO4 and tetrahedral-inclusion check, matching analysis_poly semantics."""
    frac = np.asarray(frac, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    oxygen = [index for index, element in enumerate(elements) if element == "O"]
    rows = []
    for center, element in enumerate(elements):
        if element != "P":
            continue
        vectors = []
        for vertex in oxygen:
            delta = frac[vertex] - frac[center]
            candidates = np.asarray([
                (delta + np.asarray((i, j, k))) @ lattice
                for i in (-1., 0., 1.) for j in (-1., 0., 1.) for k in (-1., 0., 1.)
            ])
            vector = candidates[np.argmin(np.linalg.norm(candidates, axis=1))]
            vectors.append((vertex, vector, float(np.linalg.norm(vector))))
        neighbours = [(index, vector) for index, vector, distance in vectors if distance <= cutoff]
        row = {"site": center, "vertices": [index for index, _ in neighbours],
               "coordination": len(neighbours), "inside": False,
               "minimum_barycentric": None, "volume_A3": None}
        if len(neighbours) == 4:
            vertices = np.asarray([vector for _, vector in neighbours])
            matrix = np.vstack((vertices.T, np.ones(4)))
            try:
                weights = np.linalg.solve(matrix, np.asarray((0., 0., 0., 1.)))
                volume = abs(np.linalg.det(np.column_stack((
                    vertices[1] - vertices[0], vertices[2] - vertices[0],
                    vertices[3] - vertices[0],
                )))) / 6.0
                row.update({"minimum_barycentric": float(weights.min()),
                            "volume_A3": float(volume),
                            "inside": bool(weights.min() > tolerance and volume > tolerance)})
            except np.linalg.LinAlgError:
                pass
        rows.append(row)
    complete = bool(rows) and all(row["coordination"] == 4 for row in rows)
    return {"complete": complete, "passes": bool(rows) and all(
        row["coordination"] == 4 and row["inside"] for row in rows
    ), "sites": rows}


def augmented_lagrangian_merit(
    violations: torch.Tensor,
    multipliers: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Mean Powell-Hestenes-Rockafellar merit for nonnegative violations."""
    return (
        multipliers.detach() * violations
        + 0.5 * rho.detach() * violations.square()
    ).sum(dim=1).mean()


def build_surrogate_graph_batch(
    elements_batch,
    frac: torch.Tensor,
    lattice: torch.Tensor,
    element_to_index,
    cutoff_A: float,
    max_neighbors: int,
):
    """Build disconnected periodic graphs while keeping geometry differentiable."""
    rows = []
    for row, elements in enumerate(elements_batch):
        structure = Structure(
            Lattice(lattice[row].detach().cpu().numpy()),
            elements,
            frac[row, : len(elements)].detach().cpu().numpy(),
        )
        graph = structure_to_model_batch(
            structure, element_to_index, cutoff_A, max_neighbors,
        )
        rows.append({
            "index": row,
            "types": graph["types"],
            "frac": graph["frac"],
            "lattice": graph["lattice"][0],
            "edge_src": graph["edge_src"],
            "edge_dst": graph["edge_dst"],
            "edge_shift": graph["edge_shift"],
            "target": graph["target"][0],
        })
    return collate_graphs(rows)


def optimize_batch(
    *, flow, surrogate, surrogate_ckpt, uma_evaluator, type_rows, masks,
    inverse, ref_desc, ref_mean, ref_std, generator, args, sample_offset,
):
    """Optimize a variable-size batch of fixed-composition Flow sources."""
    device = masks.device
    batch_size, n_max = masks.shape
    atom = torch.nn.functional.one_hot(
        (type_rows - 1).clamp_min(0), num_classes=flow.config.vocab_size,
    ).float() * 2.0 * masks.unsqueeze(-1)
    elements_batch = [
        [inverse[int(value)] for value in type_rows[row, masks[row]].tolist()]
        for row in range(batch_size)
    ]
    # Draw each structure using its true atom count.  Besides avoiding random
    # padded coordinates, this makes source streams reproducible across the
    # batched L-BFGS and one-structure-at-a-time IPOPT implementations.
    initial_frac = torch.zeros(batch_size, n_max, 3, device=device)
    initial_lattice = torch.empty(batch_size, 6, device=device)
    for row, elements in enumerate(elements_batch):
        initial_frac[row, : len(elements)] = torch.rand(
            len(elements), 3, device=device, generator=generator,
        )
        initial_lattice[row] = torch.randn(
            6, device=device, generator=generator,
        )
    if args.source_candidates > 1:
        for row, elements in enumerate(elements_batch):
            candidate_count = args.source_candidates
            n = len(elements)
            candidate_atom = atom[row : row + 1, :n].expand(
                candidate_count, -1, -1
            )
            candidate_mask = torch.ones(
                candidate_count, n, dtype=torch.bool, device=device,
            )
            candidate_frac = torch.rand(
                candidate_count, n, 3, device=device, generator=generator,
            )
            candidate_lattice = torch.randn(
                candidate_count, 6, device=device, generator=generator,
            )
            with torch.no_grad():
                candidate_terminal = integrate_flow(
                    flow,
                    CrystalState(
                        candidate_atom, candidate_frac, candidate_lattice
                    ),
                    candidate_mask, steps=args.ode_steps,
                    method="midpoint", freeze_atom=True,
                )
                candidate_physical_lattice = decode_lattice(
                    unstandardize_lattice(
                        candidate_terminal.lattice,
                        flow.lattice_mean, flow.lattice_std,
                    ),
                    candidate_mask.sum(dim=1),
                )
                candidate_scores = torch.stack([
                    constraint_penalty(
                        elements,
                        candidate_terminal.frac[index, : len(elements)],
                        candidate_physical_lattice[index : index + 1],
                        args.tau_distance, args.tau_coord,
                    )
                    for index in range(candidate_count)
                ])
                if args.require_initial_po4:
                    passing = []
                    for index in range(candidate_count):
                        check = po4_exact_checks(
                            elements,
                            candidate_terminal.frac[index, :n].cpu().numpy(),
                            candidate_physical_lattice[index].cpu().numpy(),
                        )
                        if check["complete"]:
                            passing.append(index)
                    if not passing:
                        raise RuntimeError(
                            f"no PO4-valid endpoint among {candidate_count} source candidates "
                            f"for fixed composition {elements.count('Na')}:"
                            f"{elements.count('Fe')}:{elements.count('P')}:{elements.count('O')}"
                        )
                    selected = min(passing, key=lambda index: float(candidate_scores[index]))
                else:
                    selected = int(candidate_scores.argmin())
                initial_frac[row, :n].copy_(candidate_frac[selected])
                initial_lattice[row].copy_(candidate_lattice[selected])
    po4_assignments = None
    if args.require_initial_po4:
        po4_assignments = []
        with torch.no_grad():
            selected_terminal = integrate_flow(
                flow, CrystalState(atom, initial_frac, initial_lattice), masks,
                steps=args.ode_steps, method="midpoint", freeze_atom=True,
            )
            selected_lattice = decode_lattice(
                unstandardize_lattice(
                    selected_terminal.lattice, flow.lattice_mean, flow.lattice_std,
                ), masks.sum(dim=1),
            )
            for row, elements in enumerate(elements_batch):
                check = po4_exact_checks(
                    elements, selected_terminal.frac[row, :len(elements)].cpu().numpy(),
                    selected_lattice[row].cpu().numpy(),
                )
                if not check["complete"]:
                    raise RuntimeError("selected PO4 endpoint failed reproducibility check")
                po4_assignments.append({
                    site["site"]: site["vertices"] for site in check["sites"]
                })
    z_frac = initial_frac.clone().requires_grad_(True)
    z_lattice = initial_lattice.clone().requires_grad_(True)
    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [z_frac, z_lattice], lr=args.lr, max_iter=1,
            history_size=args.lbfgs_history_size, line_search_fn=None,
            tolerance_grad=1.0e-7, tolerance_change=1.0e-9,
        )
    else:
        optimizer = torch.optim.Adam([z_frac, z_lattice], lr=args.lr)
    novelty_types = torch.zeros_like(type_rows)
    novelty_map = {"Na": 1, "Fe": 2, "P": 3, "O": 4}
    for row, elements in enumerate(elements_batch):
        novelty_types[row, : len(elements)] = torch.tensor(
            [novelty_map[element] for element in elements], device=device,
        )

    best_key = torch.full((batch_size,), float("inf"), device=device)
    best_frac = z_frac.detach().clone()
    best_lattice = z_lattice.detach().clone()
    initial_predictions = None
    graph_batch = None
    constraint_names = [
        "distance", "volume", "coordination", "poly_center",
        "poly_face", "poly_coplanar", "po4_inside", "e_hull",
    ]
    if uma_evaluator is not None:
        constraint_names.extend(("force", "stress"))
    constraint_multipliers = torch.full(
        (batch_size, len(constraint_names)),
        args.alm_initial_multiplier, device=device,
    )
    constraint_rho = torch.full(
        (len(constraint_names),), args.augmented_lagrangian_rho, device=device,
    )
    previous_outer_violation = None
    best_violations = torch.full_like(constraint_multipliers, float("inf"))
    best_fmax = torch.full((batch_size,), float("nan"), device=device)
    best_stress_fro = torch.full((batch_size,), float("nan"), device=device)
    adam_steps = (
        args.adam_steps if args.adam_steps >= 0
        else max(0, args.steps - args.lbfgs_steps)
    )

    iterations_completed = 0
    time_limit_reached = False
    for iteration in range(args.steps):
        if iteration > 0 and time.monotonic() >= args.deadline_monotonic:
            time_limit_reached = True
            break
        if args.optimizer == "hybrid-alm" and iteration == adam_steps:
            optimizer = torch.optim.LBFGS(
                [z_frac, z_lattice], lr=args.lbfgs_lr, max_iter=1,
                history_size=args.lbfgs_history_size, line_search_fn=None,
                tolerance_grad=1.0e-7, tolerance_change=1.0e-9,
            )
        if args.restoration_steps > 0 and iteration == args.restoration_steps:
            # Mechanical constraints were intentionally inactive during the
            # feasibility warm start. Do not allow such a point to outrank a
            # later candidate merely because its inactive violations were zero.
            best_key.fill_(float("inf"))
            best_violations.fill_(float("inf"))
        optimizer.zero_grad(set_to_none=True)
        terminal = integrate_flow(
            flow, CrystalState(atom, z_frac, z_lattice), masks,
            steps=args.ode_steps, method="midpoint", freeze_atom=True,
        )
        physical_lattice = decode_lattice(
            unstandardize_lattice(
                terminal.lattice, flow.lattice_mean, flow.lattice_std,
            ),
            masks.sum(dim=1),
        )
        if iteration == 0 or iteration % max(1, args.graph_refresh_steps) == 0:
            graph_batch = build_surrogate_graph_batch(
                elements_batch, terminal.frac, physical_lattice,
                surrogate_ckpt["element_to_index"], surrogate.config.cutoff_A,
                surrogate.config.max_neighbors,
            )
            graph_batch = {key: value.to(device) for key, value in graph_batch.items()}
        assert graph_batch is not None
        graph_batch["frac"] = terminal.frac[masks]
        graph_batch["lattice"] = physical_lattice
        predictions = surrogate(graph_batch)
        if initial_predictions is None:
            initial_predictions = predictions.detach().clone()

        descriptors = crystal_descriptor(
            novelty_types, terminal.frac, physical_lattice, masks,
        )
        distances = torch.cdist(
            (descriptors - ref_mean) / ref_std,
            (ref_desc - ref_mean) / ref_std,
        ) / descriptors.shape[-1] ** 0.5
        novelty = distances.min(dim=1).values
        analytic_rows = []
        poly_rows = []
        for row, elements in enumerate(elements_batch):
            n = len(elements)
            analytic_rows.append(constraint_penalty_components(
                elements, terminal.frac[row, :n],
                physical_lattice[row : row + 1],
                args.tau_distance, args.tau_coord,
            ))
            row_terminal = CrystalState(
                terminal.atom[row : row + 1, :n],
                terminal.frac[row : row + 1, :n],
                terminal.lattice[row : row + 1],
            )
            row_mask = torch.ones(1, n, dtype=torch.bool, device=device)
            terms = polyhedron_constraint_terms(flow, row_terminal, row_mask)
            terms["po4_inside"] = po4_tetrahedron_penalty(
                elements, terminal.frac[row, :n], physical_lattice[row : row + 1],
                margin=args.po4_inside_margin, temperature=args.po4_temperature,
                assignments=None if po4_assignments is None else po4_assignments[row],
            )
            poly_rows.append(terms)

        property_rows = (
            args.property_energy_weight
            * (predictions / args.energy_scale).square()
            + args.property_novelty_weight
            * ((args.novelty_target - novelty) / args.novelty_scale).square()
        )
        property_active = iteration >= args.restoration_steps
        property_loss = property_rows.mean() * float(property_active)
        fmax = predictions.new_zeros(batch_size)
        stress_fro = predictions.new_zeros(batch_size)
        mechanical_active = (
            uma_evaluator is not None and iteration >= args.restoration_steps
        )
        if mechanical_active:
            fmax_rows, stress_rows = [], []
            for row, elements in enumerate(elements_batch):
                atomic_numbers = torch.tensor(
                    [{"Na": 11, "Fe": 26, "P": 15, "O": 8}[e] for e in elements],
                    dtype=torch.long, device=device,
                )
                _, forces, stress = uma_evaluator(
                    atomic_numbers,
                    terminal.frac[row : row + 1, : len(elements)],
                    physical_lattice[row : row + 1],
                )
                fmax_rows.append(torch.linalg.vector_norm(forces, dim=-1).amax())
                # The evaluator returns one stress matrix with shape [1, 3, 3].
                # Reduce all of its matrix/batch entries to one scalar per
                # structure so the batch metric remains shape [batch_size].
                stress_rows.append(
                    torch.linalg.vector_norm(stress.reshape(-1), ord=2)
                )
            fmax = torch.stack(fmax_rows)
            stress_fro = torch.stack(stress_rows)
            stress_tol = args.stress_tolerance_gpa / 160.21766208
        component_rows = []
        for row in range(batch_size):
            components = [
                analytic_rows[row]["distance"],
                analytic_rows[row]["volume"],
                analytic_rows[row]["coordination"],
                args.poly_center_weight * poly_rows[row]["poly_center"],
                args.poly_face_weight * poly_rows[row]["poly_face"],
                args.poly_coplanar_weight * poly_rows[row]["poly_coplanar"],
                args.po4_constraint_weight * poly_rows[row]["po4_inside"],
                torch.relu((predictions[row] - args.threshold) / args.energy_scale),
            ]
            if uma_evaluator is not None:
                components.extend((
                    torch.relu((fmax[row] - args.force_tolerance)
                               / args.force_tolerance)
                    if mechanical_active else predictions.new_zeros(()),
                    torch.relu((stress_fro[row] - stress_tol) / stress_tol)
                    if mechanical_active else predictions.new_zeros(()),
                ))
            component_rows.append(torch.stack(components))
        # UMA emits float64 while the Flow and surrogate use float32. Keep one
        # optimization dtype so ALM state updates and best-candidate recording
        # are type-stable; the cast remains differentiable.
        raw_constraint_measures = torch.stack(component_rows).to(predictions.dtype)
        # Smooth geometric proxies have small positive tails. Treat values below
        # the declared tolerance as feasible; exact acceptance remains separate.
        constraints = torch.log1p(torch.relu(
            raw_constraint_measures - args.soft_constraint_tolerance
        ))
        constraint_merit_rows = (
            constraint_multipliers.detach() * constraints
            + 0.5 * constraint_rho.detach() * constraints.square()
        ).sum(dim=1)
        constraint_merit = constraint_merit_rows.mean()
        source_drift = (
            ((z_frac - initial_frac).square() * masks.unsqueeze(-1)).sum(dim=(1, 2))
            / (3 * masks.sum(dim=1).clamp_min(1))
            + (z_lattice - initial_lattice).square().mean(dim=1)
        )
        source_prior = source_drift.mean()
        # HybridOptimization.md defines one unified objective.  Every term is
        # active at every source-space optimization step; there is no geometry
        # warm-up phase that temporarily disables the property objective.
        loss = (
            property_loss + constraint_merit
            + args.source_prior_weight * source_prior
        )

        # Save the source whose metrics were actually evaluated, before Adam moves it.
        stress_tol = args.stress_tolerance_gpa / 160.21766208
        candidate_key = (
            args.best_violation_weight
            * constraints.detach().amax(dim=1)
            + constraints.detach().sum(dim=1)
            + property_rows.detach()
            + args.source_prior_weight * source_drift.detach()
        )
        candidate_eligible = (
            mechanical_active or uma_evaluator is None
            or args.restoration_steps == 0
        )
        improved = (candidate_key < best_key) & candidate_eligible
        best_key = torch.where(improved, candidate_key, best_key)
        best_frac[improved] = z_frac.detach()[improved]
        best_lattice[improved] = z_lattice.detach()[improved]
        best_violations[improved] = constraints.detach()[improved]
        best_fmax[improved] = fmax.detach().to(best_fmax.dtype)[improved]
        best_stress_fro[improved] = stress_fro.detach().to(
            best_stress_fro.dtype
        )[improved]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z_frac, z_lattice], args.gradient_clip_norm)
        if isinstance(optimizer, torch.optim.LBFGS):
            # max_iter=1 makes this one expensive objective/gradient evaluation
            # per outer iteration while preserving L-BFGS curvature history.
            optimizer.step(lambda: loss)
        else:
            optimizer.step()
        iterations_completed = iteration + 1
        with torch.no_grad():
            if (iteration + 1) % args.alm_outer_steps == 0:
                current_outer = constraints.detach().mean(dim=0)
                constraint_multipliers.add_(
                    args.dual_learning_rate
                    * constraint_rho.unsqueeze(0)
                    * constraints.detach()
                ).clamp_(min=0.0, max=args.multiplier_max)
                if previous_outer_violation is not None:
                    stalled = current_outer > (
                        args.alm_progress_ratio * previous_outer_violation
                    )
                    constraint_rho[stalled].mul_(args.alm_rho_growth)
                    constraint_rho.clamp_(max=args.alm_rho_max)
                previous_outer_violation = current_outer
            z_frac.remainder_(1.0)
            z_frac.mul_(masks.unsqueeze(-1))
            z_lattice.clamp_(-5.0, 5.0)

    with torch.no_grad():
        final = integrate_flow(
            flow, CrystalState(atom, best_frac, best_lattice), masks,
            steps=args.ode_steps, method="midpoint", freeze_atom=True,
        )
        final_lattice = decode_lattice(
            unstandardize_lattice(final.lattice, flow.lattice_mean, flow.lattice_std),
            masks.sum(dim=1),
        )
    final_graph = build_surrogate_graph_batch(
        elements_batch, final.frac, final_lattice,
        surrogate_ckpt["element_to_index"], surrogate.config.cutoff_A,
        surrogate.config.max_neighbors,
    )
    final_graph = {key: value.to(device) for key, value in final_graph.items()}
    final_graph["frac"] = final.frac[masks]
    final_graph["lattice"] = final_lattice
    with torch.no_grad():
        final_predictions = surrogate(final_graph)
        final_descriptors = crystal_descriptor(
            novelty_types, final.frac, final_lattice, masks,
        )
        final_novelty = torch.cdist(
            (final_descriptors - ref_mean) / ref_std,
            (ref_desc - ref_mean) / ref_std,
        ).min(dim=1).values / final_descriptors.shape[-1] ** 0.5

    records = []
    for row, elements in enumerate(elements_batch):
        n = len(elements)
        prediction = float(final_predictions[row])
        structure = Structure(
            Lattice(final_lattice[row].detach().cpu().numpy()), elements,
            final.frac[row, :n].detach().cpu().numpy(),
        )
        analytic_checks = evaluate_feasibility(
            elements, final.frac[row, :n].detach().cpu().numpy(),
            final_lattice[row].detach().cpu().numpy(), prediction,
        )
        po4_checks = po4_exact_checks(
            elements, final.frac[row, :n].detach().cpu().numpy(),
            final_lattice[row].detach().cpu().numpy(),
        )
        soft_constraints_feasible = bool(
            (best_violations[row] <= args.alm_feasibility_tolerance).all()
        )
        records.append({
            "sample_id": sample_offset + row,
            "N": n,
            "elements": elements,
            "formula": structure.formula,
            "initial_e_hull_surrogate": float(initial_predictions[row]),
            "optimized_e_hull_surrogate": prediction,
            "optimized_novelty": float(final_novelty[row]),
            "optimizer": args.optimizer,
            "optimizer_schedule": {
                "restoration_steps": min(
                    iterations_completed, args.restoration_steps
                ),
                "adam_steps": min(iterations_completed, adam_steps)
                if args.optimizer == "hybrid-alm" else 0,
                "lbfgs_steps": max(0, iterations_completed - adam_steps)
                if args.optimizer == "hybrid-alm" else 0,
            },
            "optimization_iterations": iterations_completed,
            "time_limit_reached": time_limit_reached,
            "source_candidates": args.source_candidates,
            "passes_threshold": prediction <= args.threshold,
            "optimized_uma_fmax_eV_A": (
                float(best_fmax[row]) if uma_evaluator is not None else None
            ),
            "optimized_uma_stress_fro_GPa": (
                float(best_stress_fro[row] * 160.21766208)
                if uma_evaluator is not None else None
            ),
            "soft_constraint_violations": {
                name: float(best_violations[row, index])
                for index, name in enumerate(constraint_names)
            },
            "soft_constraints_feasible": soft_constraints_feasible,
            "analytic_constraint_checks": analytic_checks,
            "po4_tetrahedron_checks": po4_checks,
            "all_constraints_feasible": bool(
                analytic_checks["feasible"] and po4_checks["passes"]
                and soft_constraints_feasible
            ),
            "frac_coords": final.frac[row, :n].detach().cpu().tolist(),
            "lattice": final_lattice[row].detach().cpu().tolist(),
        })
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow", required=True)
    p.add_argument("--surrogate", required=True)
    p.add_argument(
        "--packed-data",
        default="/mnt/data2/aobo/NaGen/inverse_v1/data/naxl_packed_v1.pt",
        help="Packed reference data used only to construct the novelty reference set.",
    )
    p.add_argument("--novelty-reference-count", type=int, default=0,
                   help="Reference structures for novelty; 0 uses the full set.")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Number of samples optimized concurrently on one device.")
    p.add_argument("--steps", type=int, default=140)
    p.add_argument(
        "--optimizer", choices=("adam", "lbfgs", "hybrid-alm"),
        default="hybrid-alm",
    )
    p.add_argument(
        "--adam-steps", type=int, default=-1,
        help="Adam steps for hybrid-alm; -1 means steps - lbfgs-steps.",
    )
    p.add_argument("--lbfgs-steps", type=int, default=20)
    p.add_argument("--lbfgs-lr", type=float, default=0.2)
    p.add_argument(
        "--restoration-steps", type=int, default=0,
        help="Initial Adam steps with property and UMA force/stress terms inactive; "
             "geometry and E_hull feasibility constraints remain active.",
    )
    p.add_argument("--lbfgs-history-size", type=int, default=10)
    p.add_argument("--source-candidates", type=int, default=1)
    p.add_argument("--require-initial-po4", action="store_true",
                   help="Only optimize a source whose generated endpoint already passes exact PO4 checks.")
    p.add_argument(
        "--time-limit-seconds", type=float, default=0.0,
        help="Per-process wall-clock limit including model/data initialization; 0 disables.",
    )
    p.add_argument(
        "--geometry-steps", type=int, default=0,
        help="Deprecated compatibility option; must be 0 because all objective "
             "terms are active throughout optimization.",
    )
    p.add_argument("--ode-steps", type=int, default=24)
    p.add_argument("--lr", type=float, default=0.04)
    p.add_argument("--gradient-clip-norm", type=float, default=5.0)
    p.add_argument("--source-prior-weight", type=float, default=0.01)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-atoms", type=int, default=184)
    p.add_argument("--composition-pool-factor", type=int, default=6,
                   help="Proposal multiplier for exact composition rejection sampling.")
    p.add_argument("--composition-max-rounds", type=int, default=20,
                   help="Maximum rounds for exact composition rejection sampling.")
    p.add_argument("--composition-proposal-batch-limit", type=int, default=512,
                   help="Maximum Flow composition proposals evaluated per round.")
    p.add_argument("--max-p-sites", type=int, default=0,
                   help="Optional post-generation composition screen; 0 disables it.")
    p.add_argument("--composition-oversample-factor", type=int, default=1,
                   help="Generate this many feasible compositions before optional screening.")
    p.add_argument("--constraint-weight", type=float, default=20.0)
    p.add_argument("--augmented-lagrangian-rho", type=float, default=1.0)
    p.add_argument("--alm-initial-multiplier", type=float, default=1.0)
    p.add_argument("--alm-outer-steps", type=int, default=10)
    p.add_argument("--alm-rho-growth", type=float, default=2.0)
    p.add_argument("--alm-rho-max", type=float, default=1000.0)
    p.add_argument("--alm-progress-ratio", type=float, default=0.75)
    p.add_argument("--alm-feasibility-tolerance", type=float, default=1.0e-4)
    p.add_argument("--best-violation-weight", type=float, default=1000.0)
    p.add_argument(
        "--soft-constraint-tolerance", type=float, default=1.0e-4,
        help="Positive tail tolerated for smooth geometric violation proxies.",
    )
    p.add_argument("--dual-learning-rate", type=float, default=1.0)
    p.add_argument("--multiplier-max", type=float, default=1000.0)
    p.add_argument("--property-energy-weight", type=float, default=0.8)
    p.add_argument("--property-novelty-weight", type=float, default=0.2)
    p.add_argument("--novelty-target", type=float, default=1.0)
    p.add_argument("--energy-scale", type=float, default=0.15)
    p.add_argument("--novelty-scale", type=float, default=1.0)
    p.add_argument("--poly-center-weight", type=float, default=8.0)
    p.add_argument("--poly-face-weight", type=float, default=8.0)
    p.add_argument("--poly-coplanar-weight", type=float, default=4.0)
    p.add_argument("--po4-constraint-weight", type=float, default=0.0,
                   help="Weight for the P-in-PO4-tetrahedron inner loss; 0 is baseline.")
    p.add_argument("--po4-inside-margin", type=float, default=0.01)
    p.add_argument("--po4-temperature", type=float, default=0.02)
    p.add_argument("--tau-distance", type=float, default=0.08)
    p.add_argument("--tau-coord", type=float, default=0.08)
    p.add_argument("--graph-refresh-steps", type=int, default=10,
                   help="Rebuild surrogate periodic neighbor graph every K steps.")
    p.add_argument("--uma-checkpoint", default=None,
                   help="Optional UMA checkpoint for differentiable force/stress barriers.")
    p.add_argument("--force-barrier-weight", type=float, default=20.0)
    p.add_argument("--stress-barrier-weight", type=float, default=20.0)
    p.add_argument("--force-tolerance", type=float, default=0.01)
    p.add_argument("--stress-tolerance-gpa", type=float, default=0.10)
    p.add_argument("--cif-dir", default=None,
                   help="Optional directory for one final CIF per generated candidate.")
    args = p.parse_args()
    args.deadline_monotonic = (
        time.monotonic() + args.time_limit_seconds
        if args.time_limit_seconds > 0 else float("inf")
    )
    if args.geometry_steps != 0:
        raise ValueError(
            "--geometry-steps must be 0: HybridOptimization uses the unified "
            "property and constraint objective at every optimization step"
        )
    if args.source_candidates <= 0:
        raise ValueError("--source-candidates must be positive")
    if args.require_initial_po4 and args.source_candidates < 2:
        raise ValueError("--require-initial-po4 requires --source-candidates >= 2")
    if args.alm_outer_steps <= 0:
        raise ValueError("--alm-outer-steps must be positive")
    if args.optimizer == "hybrid-alm":
        adam_steps = (
            args.adam_steps if args.adam_steps >= 0
            else args.steps - args.lbfgs_steps
        )
        if not 0 <= adam_steps <= args.steps:
            raise ValueError("hybrid-alm Adam steps must be within [0, steps]")
    if not 0 <= args.restoration_steps < args.steps:
        raise ValueError("--restoration-steps must be within [0, steps)")
    if args.restoration_steps > adam_steps:
        raise ValueError("restoration must finish before the L-BFGS phase")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    flow, flow_ckpt = load_model(args.flow, device, use_ema=True)
    surrogate, surrogate_ckpt = load_surrogate(args.surrogate, device)
    ref_desc, ref_mean, ref_std = reference_descriptors(
        args.packed_data, args.novelty_reference_count, device,
    )
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    uma_evaluator = None
    if args.uma_checkpoint:
        uma_evaluator, _ = load_uma_native_second_order_evaluator(
            args.uma_checkpoint, device=device, task_name="omat"
        )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    proposal_count = args.count * args.composition_oversample_factor
    types, masks = sample_feasible_compositions(
        flow, flow_ckpt, proposal_count, generator, args.ode_steps, "midpoint",
        pool_factor=args.composition_pool_factor,
        max_rounds=args.composition_max_rounds,
        max_atoms=args.max_atoms,
        proposal_batch_limit=args.composition_proposal_batch_limit,
    )
    if args.max_p_sites > 0:
        keep = (types == int(flow_ckpt["element_to_index"]["P"])).sum(dim=1) <= args.max_p_sites
        types, masks = types[keep], masks[keep]
        if len(types) < args.count:
            raise RuntimeError(
                f"only {len(types)} of {args.count} feasible compositions passed "
                f"the P-site screen P<={args.max_p_sites}"
            )
    types, masks = types[:args.count], masks[:args.count]
    types = types.to(device)
    masks = masks.to(device)
    inverse = {int(v): k for k, v in flow_ckpt["element_to_index"].items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cif_dir = Path(args.cif_dir) if args.cif_dir else None
    if cif_dir:
        cif_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    records = []
    for start in range(0, args.count, args.batch_size):
        stop = min(start + args.batch_size, args.count)
        batch_records = optimize_batch(
            flow=flow, surrogate=surrogate, surrogate_ckpt=surrogate_ckpt,
            uma_evaluator=uma_evaluator, type_rows=types[start:stop],
            masks=masks[start:stop], inverse=inverse, ref_desc=ref_desc,
            ref_mean=ref_mean, ref_std=ref_std, generator=generator, args=args,
            sample_offset=start,
        )
        records.extend(batch_records)
        for record in batch_records:
            if cif_dir:
                Structure(
                    Lattice(record["lattice"]), record["elements"], record["frac_coords"],
                ).to(filename=str(cif_dir / f"sample_{record['sample_id']:03d}.cif"))
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {
        "version": "nagen-surrogate-shootingflow-v1",
        "flow_checkpoint": str(Path(args.flow).resolve()),
        "surrogate_checkpoint": str(Path(args.surrogate).resolve()),
        "flow_frozen": True, "surrogate_frozen": True,
        "optimized_variables": ["z_X", "z_L"],
        "fixed_variables": ["N", "A"],
        "uma_relaxation_performed": False,
        "true_e_hull_evaluation_performed": False,
        "count": len(records),
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "adam_steps": args.adam_steps,
        "lbfgs_steps": args.lbfgs_steps,
        "restoration_steps": args.restoration_steps,
        "alm_outer_steps": args.alm_outer_steps,
        "alm_initial_rho": args.augmented_lagrangian_rho,
        "constraint_residual_scaling": "log1p",
        "source_candidates": args.source_candidates,
        "time_limit_seconds": args.time_limit_seconds,
        "threshold": args.threshold,
        "geometry_steps": args.geometry_steps,
        "property_energy_weight": args.property_energy_weight,
        "property_novelty_weight": args.property_novelty_weight,
        "constraint_weight": args.constraint_weight,
        "tau_distance": args.tau_distance,
        "tau_coord": args.tau_coord,
        "graph_refresh_steps": args.graph_refresh_steps,
        "po4_constraint_weight": args.po4_constraint_weight,
        "po4_inside_margin": args.po4_inside_margin,
        "novelty_reference_count": int(ref_desc.shape[0]),
        "uma_force_stress_barrier": bool(args.uma_checkpoint),
        "passing": sum(r["passes_threshold"] for r in records),
        "soft_feasible": sum(r["soft_constraints_feasible"] for r in records),
        "exact_feasible": sum(
            r["analytic_constraint_checks"]["feasible"] for r in records
        ),
        "records": records,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("count", "passing", "threshold")}), flush=True)


if __name__ == "__main__":
    main()

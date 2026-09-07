"""ShootingFlow experiment with a frozen unconditional Flow and E_hull surrogate.

The experiment fixes N and A for each start and optimizes only continuous source
variables (z_X, z_L). No UMA relaxation or true E_hull evaluation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pymatgen.core import Lattice, Structure

from nagen.inverse.generate import load_model, sample_feasible_compositions
from nagen.inverse.lattice import decode_lattice, unstandardize_lattice
from nagen.inverse.model import CrystalState
from nagen.inverse.sample import integrate_flow
from nagen.inverse.constraints import evaluate_feasibility, composition_metrics
from nagen.inverse.novelty import crystal_descriptor
from nagen.inverse.dataset import PackedCrystalDataset, collate_crystals
from nagen.surrogate.ehull_mace import load_surrogate, structure_to_model_batch


def reference_descriptors(path: str, count: int, device: torch.device):
    packed = torch.load(path, map_location="cpu", weights_only=False)
    allowed = {int(packed["element_to_index"][e]) for e in ("Na", "Fe", "P", "O")}
    rows = [i for i in range(len(packed["ids"])) if set(int(x) for x in packed["types"][packed["offsets"][i]:packed["offsets"][i+1]]) <= allowed]
    rows = rows[:count]
    ds = PackedCrystalDataset(path, "train")
    ds.indices = rows
    b = collate_crystals([ds[i] for i in range(len(rows))])
    remap = {int(packed["element_to_index"][e]) : j + 1 for j, e in enumerate(("Na", "Fe", "P", "O"))}
    b["types"] = torch.tensor([[remap.get(int(x), 0) for x in row] for row in b["types"].tolist()], dtype=torch.long)
    with torch.no_grad():
        d = crystal_descriptor(b["types"].to(device), b["frac_coords"].to(device), b["lattice"].to(device), b["mask"].to(device))
    return d, d.mean(0), d.std(0).clamp_min(1e-4)


def constraint_penalty(elements, frac, lattice, tau_distance=0.08, tau_coord=0.08):
    """Differentiable soft form of the frozen A5/A6 constraints.

    This is deliberately kept local to the surrogate shooting experiment: the
    exact acceptance gate remains ``evaluate_feasibility`` and therefore still
    comes from ``docs/OPTIMIZATION.md``/``inverse.constraints``.
    """
    n = frac.shape[0]
    shifts = torch.tensor([(i,j,k) for i in (-1.,0.,1.) for j in (-1.,0.,1.) for k in (-1.,0.,1.)], device=frac.device, dtype=frac.dtype)
    delta = frac[:,None,:] - frac[None,:,:] + shifts[:,None,None,:]
    cart = torch.einsum("sijd,dk->sijk", delta, lattice[0])
    distances = torch.linalg.vector_norm(cart, dim=-1)
    dist = -tau_distance * torch.logsumexp(-distances / tau_distance, dim=0)
    names = elements
    minima = {"P-O":1.40,"Fe-O":1.55,"Na-O":2.00,"O-O":2.00,"P-P":2.60,"Na-Na":2.20}
    pair_pen = frac.new_zeros(())
    for i in range(n):
        for j in range(i+1,n):
            key = "-".join(sorted((names[i], names[j]), key=("Na","Fe","P","O").index))
            threshold = minima.get(key, 0.75 * ((1.66 if names[i]=="Na" else 1.32 if names[i]=="Fe" else 1.07 if names[i]=="P" else .66) + (1.66 if names[j]=="Na" else 1.32 if names[j]=="Fe" else 1.07 if names[j]=="P" else .66)))
            pair_pen = pair_pen + torch.nn.functional.softplus((frac.new_tensor(threshold) - dist[i,j]) / tau_distance).square()
    volume = torch.linalg.det(lattice[0]).abs() / n
    volume_pen = torch.relu(frac.new_tensor(10.5)-volume).square() + torch.relu(volume-frac.new_tensor(20.5)).square()
    coord_pen = frac.new_zeros(())
    for i, e in enumerate(names):
        if e not in ("P", "Fe"):
            continue
        cutoff = 2.0 if e == "P" else 2.5
        mask = torch.tensor([x == "O" for x in names], device=frac.device)
        ordered = torch.sort(dist[i][mask]).values
        # Exact coordination is represented by a shell boundary: the 4th
        # oxygen must be inside the bond cutoff and the next oxygen outside.
        if ordered.numel() >= 4:
            coord_pen = coord_pen + torch.nn.functional.softplus((ordered[3] - cutoff) / tau_coord).square()
        else:
            coord_pen = coord_pen + torch.nn.functional.softplus(frac.new_tensor(4.0 - ordered.numel())) ** 2
        upper_index = 4 if e == "P" else 6
        if ordered.numel() > upper_index:
            coord_pen = coord_pen + torch.nn.functional.softplus((cutoff - ordered[upper_index]) / tau_coord).square()
    return 10.0 * pair_pen + 0.5 * volume_pen + 2.0 * coord_pen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow", required=True)
    p.add_argument("--surrogate", required=True)
    p.add_argument(
        "--packed-data",
        default="/mnt/data2/aobo/NaGen/inverse_v1/data/naxl_packed_v1.pt",
        help="Packed reference data used only for composition fallback and novelty.",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--geometry-steps", type=int, default=80)
    p.add_argument("--ode-steps", type=int, default=24)
    p.add_argument("--lr", type=float, default=0.04)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-atoms", type=int, default=64)
    p.add_argument("--novelty-weight", type=float, default=0.02)
    p.add_argument("--constraint-weight", type=float, default=1.0)
    p.add_argument("--tau-distance", type=float, default=0.08)
    p.add_argument("--tau-coord", type=float, default=0.08)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    flow, flow_ckpt = load_model(args.flow, device, use_ema=True)
    surrogate, surrogate_ckpt = load_surrogate(args.surrogate, device)
    ref_desc, ref_mean, ref_std = reference_descriptors(args.packed_data, 512, device)
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    try:
        types, masks = sample_feasible_compositions(
            flow, flow_ckpt, args.count, generator, args.ode_steps, "midpoint",
            max_atoms=args.max_atoms,
        )
    except RuntimeError:
        # The unconditional composition head may not yet satisfy exact charge
        # filters. For this diagnostic, draw valid N/A compositions from the
        # original packed corpus and test only the continuous shooting bridge.
        packed = torch.load(
            args.packed_data,
            map_location="cpu", weights_only=False,
        )
        allowed_indices = {int(flow_ckpt["element_to_index"][e]) for e in ("Na", "Fe", "P", "O")}
        candidates = [i for i in range(len(packed["ids"]))
                      if int(packed["offsets"][i + 1] - packed["offsets"][i]) <= args.max_atoms
                      and set(int(x) for x in packed["types"][packed["offsets"][i]:packed["offsets"][i + 1]]) <= allowed_indices
                      and all(composition_metrics([
                          {int(v): k for k, v in packed["element_to_index"].items()}[int(x)]
                          for x in packed["types"][packed["offsets"][i]:packed["offsets"][i + 1]]
                      ])[key] for key in ("domain_ok", "charge_ok", "capacity_ok"))]
        if len(candidates) < args.count:
            pick = torch.randint(len(candidates), (args.count,), generator=torch.Generator().manual_seed(args.seed))
        else:
            pick = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(args.seed))[:args.count]
        rows = [candidates[int(i)] for i in pick]
        n_atoms = [int(packed["offsets"][i + 1] - packed["offsets"][i]) for i in rows]
        n_max = max(n_atoms)
        masks = torch.arange(n_max)[None, :] < torch.tensor(n_atoms)[:, None]
        types = torch.zeros(args.count, n_max, dtype=torch.long)
        for r, i in enumerate(rows):
            types[r, :n_atoms[r]] = packed["types"][packed["offsets"][i]:packed["offsets"][i + 1]]
    types = types.to(device)
    masks = masks.to(device)
    inverse = {int(v): k for k, v in flow_ckpt["element_to_index"].items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in range(args.count):
        n = int(masks[row].sum())
        mask = torch.ones(1, n, dtype=torch.bool, device=device)
        type_row = types[row : row + 1, :n]
        atom = torch.nn.functional.one_hot(
            (type_row - 1).clamp_min(0), num_classes=flow.config.vocab_size
        ).float() * 2.0
        source = CrystalState(
            atom,
            torch.rand(1, n, 3, device=device, generator=generator),
            torch.randn(1, 6, device=device, generator=generator),
        )
        z_frac = source.frac.detach().clone().requires_grad_(True)
        z_lattice = source.lattice.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([z_frac, z_lattice], lr=args.lr)
        best = float("inf")
        best_state = None
        for iteration in range(args.steps):
            opt.zero_grad(set_to_none=True)
            terminal = integrate_flow(
                flow, CrystalState(source.atom, z_frac, z_lattice), mask,
                steps=args.ode_steps, method="midpoint", freeze_atom=True,
            )
            physical_lattice = decode_lattice(
                unstandardize_lattice(terminal.lattice, flow.lattice_mean, flow.lattice_std),
                mask.sum(dim=1),
            )
            elements = [inverse[int(x)] for x in type_row[0].tolist()]
            # Build a fixed neighbor topology once; coordinates and lattice remain
            # differentiable tensors throughout the shooting optimization.
            if iteration == 0:
                ref = Structure(Lattice(physical_lattice[0].detach().cpu().numpy()), elements,
                                terminal.frac[0].detach().cpu().numpy())
                batch = structure_to_model_batch(
                    ref, surrogate_ckpt["element_to_index"],
                    surrogate.config.cutoff_A, surrogate.config.max_neighbors,
                )
                batch = {k: v.to(device) for k, v in batch.items()}
            batch["frac"] = terminal.frac[0]
            batch["lattice"] = physical_lattice
            prediction = surrogate(batch).reshape(())
            # Differentiable novelty: maximize nearest standardized descriptor
            # distance to a fixed reference subset (single scalar objective).
            nov_types = torch.tensor(
                [[{"Na":1,"Fe":2,"P":3,"O":4}[e] for e in elements]],
                device=device, dtype=torch.long,
            )
            nov_mask = torch.ones(1, n, dtype=torch.bool, device=device)
            descriptor = crystal_descriptor(nov_types, terminal.frac, physical_lattice, nov_mask)
            distances = torch.cdist((descriptor - ref_mean) / ref_std, (ref_desc - ref_mean) / ref_std) / descriptor.shape[-1] ** 0.5
            novelty = distances.min()
            constraints = constraint_penalty(elements, terminal.frac[0], physical_lattice, args.tau_distance, args.tau_coord)
            threshold_loss = 0.02 * torch.nn.functional.softplus(
                (prediction - args.threshold) / 0.02
            )
            prior = 0.01 * (z_frac.square().mean() + z_lattice.square().mean())
            if iteration < args.geometry_steps:
                # Stage 1: project the generated endpoint into the analytic
                # geometry-feasible region before applying property guidance.
                loss = args.constraint_weight * constraints + prior
                stage = "geometry_projection"
            else:
                # Stage 2: one scalar objective near the feasible geometry
                # manifold; novelty is an additive objective, not a Pareto task.
                loss = threshold_loss - args.novelty_weight * novelty + args.constraint_weight * constraints + prior
                stage = "hull_novelty_guidance"
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z_frac, z_lattice], 5.0)
            opt.step()
            with torch.no_grad():
                z_frac.remainder_(1.0)
                z_lattice.clamp_(-5.0, 5.0)
            value = float(prediction.detach())
            if iteration >= args.geometry_steps and value < best:
                best = value
                best_state = (z_frac.detach().clone(), z_lattice.detach().clone())
        if best_state is None:
            best_state = (z_frac.detach().clone(), z_lattice.detach().clone())
        assert best_state is not None
        with torch.no_grad():
            final = integrate_flow(
                flow, CrystalState(source.atom, best_state[0], best_state[1]), mask,
                steps=args.ode_steps, method="midpoint", freeze_atom=True,
            )
            physical_lattice = decode_lattice(
                unstandardize_lattice(final.lattice, flow.lattice_mean, flow.lattice_std),
                mask.sum(dim=1),
            )
            batch["frac"] = final.frac[0]
            batch["lattice"] = physical_lattice
            final_prediction = float(surrogate(batch).item())
            elements = [inverse[int(x)] for x in type_row[0].tolist()]
        records.append({
            "sample_id": row, "N": n, "elements": elements,
            "formula": Structure(Lattice(physical_lattice[0].cpu().numpy()), elements,
                                   final.frac[0].cpu().numpy()).formula,
            "initial_e_hull_surrogate": None,
            "optimized_e_hull_surrogate": final_prediction,
            "novelty_objective_weight": args.novelty_weight,
            "passes_threshold": final_prediction <= args.threshold,
            "analytic_constraint_checks": evaluate_feasibility(
                elements, final.frac[0].cpu().numpy(), physical_lattice[0].cpu().numpy(), final_prediction
            ),
            "frac_coords": final.frac[0].cpu().tolist(),
            "lattice": physical_lattice[0].cpu().tolist(),
        })
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
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
        "threshold": args.threshold,
        "geometry_steps": args.geometry_steps,
        "novelty_weight": args.novelty_weight,
        "constraint_weight": args.constraint_weight,
        "tau_distance": args.tau_distance,
        "tau_coord": args.tau_coord,
        "passing": sum(r["passes_threshold"] for r in records),
        "records": records,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("count", "passing", "threshold")}), flush=True)


if __name__ == "__main__":
    main()

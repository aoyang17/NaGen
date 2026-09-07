"""Small-data overfit diagnostic for the production crystal Flow.

This intentionally reuses ``CrystalVectorField`` and ``flow_matching_loss``.
It asks whether the production objective can memorize either one structure or
32 same-composition structures before data scale or downstream optimization is
allowed to enter the diagnosis.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from nagen.inverse.constraints import evaluate_feasibility, pbc_distance_matrix
from nagen.inverse.lattice import decode_lattice, encode_lattice, standardize_lattice, unstandardize_lattice
from nagen.inverse.model import CrystalState, CrystalVectorField, FlowConfig, flow_matching_loss
from nagen.inverse.sample import integrate_flow


def choose_indices(data: dict, count: int) -> list[int]:
    train = torch.where(data["split"] == 0)[0].tolist()
    offsets = data["offsets"]
    if count == 1:
        element_by_index = {1: "Na", 2: "Fe", 3: "P", 4: "O"}
        eligible = []
        for index in train:
            start, stop = int(offsets[index]), int(offsets[index + 1])
            elements = [element_by_index[int(value)] for value in data["types"][start:stop]]
            result = evaluate_feasibility(
                elements, data["frac_coords"][start:stop].numpy(), data["lattices"][index].numpy()
            )
            non_hull_checks = {key: value for key, value in result["checks"].items() if key != "hull_threshold"}
            if all(non_hull_checks.values()):
                eligible.append(index)
        if not eligible:
            raise RuntimeError("no training structure passes every non-hull hard constraint")
        return [min(eligible, key=lambda i: int(offsets[i + 1] - offsets[i]))]
    groups: dict[tuple[int, str], list[int]] = {}
    for index in train:
        key = (int(offsets[index + 1] - offsets[index]), data["formulas"][index])
        groups.setdefault(key, []).append(index)
    eligible = [items for items in groups.values() if len(items) >= count]
    if not eligible:
        raise RuntimeError(f"no same-(N, formula) group contains {count} structures")
    # Prefer the smallest cell among the largest families to keep this diagnostic cheap.
    items = min(eligible, key=lambda rows: (int(offsets[rows[0] + 1] - offsets[rows[0]]), -len(rows)))
    return items[:count]


def make_batch(data: dict, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    offsets = data["offsets"]
    sizes = [int(offsets[i + 1] - offsets[i]) for i in indices]
    n_max = max(sizes)
    types = torch.zeros(len(indices), n_max, dtype=torch.long)
    frac = torch.zeros(len(indices), n_max, 3)
    mask = torch.zeros(len(indices), n_max, dtype=torch.bool)
    for row, index in enumerate(indices):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        n = stop - start
        types[row, :n] = data["types"][start:stop].long()
        frac[row, :n] = data["frac_coords"][start:stop].float()
        mask[row, :n] = True
    return {
        "types": types.to(device), "frac": frac.to(device), "mask": mask.to(device),
        "N": mask.sum(1).to(device), "lattice": data["lattices"][indices].float().to(device),
    }


def shuffled_training_batch(batch: dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    types, frac = batch["types"].clone(), batch["frac"].clone()
    for row, n_value in enumerate(batch["N"].tolist()):
        permutation = torch.randperm(n_value, device=types.device)
        types[row, :n_value] = types[row, permutation]
        frac[row, :n_value] = frac[row, permutation]
    lattice_state = standardize_lattice(encode_lattice(batch["lattice"], batch["N"]), mean, std)
    return {"types": types, "frac": frac, "mask": batch["mask"], "lattice_state": lattice_state}


def minimum_distance(frac: np.ndarray, lattice: np.ndarray) -> float:
    distances = pbc_distance_matrix(frac, lattice)
    return float(distances[np.triu_indices(len(frac), 1)].min())


def fingerprint(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    distances = pbc_distance_matrix(frac, lattice)
    return np.sort(distances[np.triu_indices(len(frac), 1)])


@torch.no_grad()
def evaluate(model: CrystalVectorField, batch: dict[str, torch.Tensor], rounds: int, ode_steps: int = 24) -> dict:
    model.eval()
    device = batch["types"].device
    mask, n_atoms = batch["mask"], batch["N"]
    target_state = standardize_lattice(
        encode_lattice(batch["lattice"], n_atoms), model.lattice_mean, model.lattice_std
    )
    element_by_index = {1: "Na", 2: "Fe", 3: "P", 4: "O"}
    target_fingerprints = [
        fingerprint(batch["frac"][i, : int(n_atoms[i])].cpu().numpy(), batch["lattice"][i].cpu().numpy())
        for i in range(len(n_atoms))
    ]
    records = []
    frac_losses, zero_losses = [], []
    for round_index in range(rounds):
        source_frac = torch.remainder(batch["frac"] + 0.5 * torch.randn_like(batch["frac"]), 1.0)
        delta = batch["frac"] - source_frac
        delta = delta - torch.floor(delta + 0.5)
        zero_losses.append(float((delta.square().sum(-1) * mask).sum() / mask.sum()))
        atom = torch.nn.functional.one_hot((batch["types"] - 1).clamp_min(0), 4).float() * 2.0
        atom = atom * mask.unsqueeze(-1)
        source = CrystalState(atom, source_frac, torch.randn_like(target_state))
        terminal = integrate_flow(model, source, mask, steps=ode_steps, freeze_atom=True)
        physical = unstandardize_lattice(terminal.lattice, model.lattice_mean, model.lattice_std)
        lattices = decode_lattice(physical, n_atoms)
        for row, n_value in enumerate(n_atoms.tolist()):
            out_frac = terminal.frac[row, :n_value].cpu().numpy()
            out_lattice = lattices[row].cpu().numpy()
            target_frac = batch["frac"][row, :n_value].cpu().numpy()
            target_lattice = batch["lattice"][row].cpu().numpy()
            source_coordinates = source_frac[row, :n_value].cpu().numpy()
            elements = [element_by_index[int(v)] for v in batch["types"][row, :n_value].tolist()]
            check = evaluate_feasibility(elements, out_frac, out_lattice)
            out_fp = fingerprint(out_frac, out_lattice)
            nearest_fp_mae = min(float(np.mean(np.abs(out_fp - target_fp))) for target_fp in target_fingerprints)
            paired_delta = target_frac - out_frac
            paired_delta -= np.floor(paired_delta + 0.5)
            records.append({
                "round": round_index, "row": row,
                "source_min_A": minimum_distance(source_coordinates, target_lattice),
                "output_min_A": check["values"]["minimum_distance_A"],
                "target_min_A": minimum_distance(target_frac, target_lattice),
                "paired_frac_rms": float(np.sqrt(np.mean(paired_delta ** 2))),
                "nearest_target_pair_distance_mae_A": nearest_fp_mae,
                "minimum_distance_pass": check["checks"]["minimum_pbc_distances"],
                "p_coordination_pass": check["checks"]["p_coordination_eq_4"],
                "fe_coordination_pass": check["checks"]["fe_coordination_in_4_5_6"],
                "volume_pass": check["checks"]["volume_per_atom_range"],
            })
        probe_t = torch.rand(len(n_atoms), 1, device=device).clamp_(1e-4, 1 - 1e-4)
        frac_t = torch.remainder(source_frac + probe_t[:, None, :] * delta, 1.0)
        lattice_t = (1 - probe_t) * source.lattice + probe_t * target_state
        prediction = model(atom, frac_t, lattice_t, probe_t, mask)
        frac_losses.append(float(((prediction.frac - delta).square().sum(-1) * mask).sum() / mask.sum()))

    def stats(key: str) -> dict[str, float]:
        values = np.asarray([record[key] for record in records], dtype=float)
        return {"mean": float(values.mean()), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}

    time_binned = {}
    for t_value in (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
        losses, baselines = [], []
        for _ in range(max(8, rounds // 2)):
            source_frac = torch.remainder(batch["frac"] + 0.5 * torch.randn_like(batch["frac"]), 1.0)
            delta = batch["frac"] - source_frac
            delta = delta - torch.floor(delta + 0.5)
            t = torch.full((len(n_atoms), 1), t_value, device=device)
            frac_t = torch.remainder(source_frac + t[:, None, :] * delta, 1.0)
            lattice_source = torch.randn_like(target_state)
            lattice_t = (1 - t) * lattice_source + t * target_state
            atom = torch.nn.functional.one_hot((batch["types"] - 1).clamp_min(0), 4).float() * 2.0
            atom = atom * mask.unsqueeze(-1)
            prediction = model(atom, frac_t, lattice_t, t, mask)
            losses.append(float(((prediction.frac - delta).square().sum(-1) * mask).sum() / mask.sum()))
            baselines.append(float((delta.square().sum(-1) * mask).sum() / mask.sum()))
        time_binned[f"{t_value:.2f}"] = {
            "model": float(np.mean(losses)), "zero": float(np.mean(baselines)),
            "relative_improvement": float(1 - np.mean(losses) / np.mean(baselines)),
        }

    return {
        "ode_steps": ode_steps,
        "n_generated": len(records),
        "coordinate_loss": {"model_mean": float(np.mean(frac_losses)), "zero_velocity_mean": float(np.mean(zero_losses)),
                            "relative_improvement": float(1 - np.mean(frac_losses) / np.mean(zero_losses))},
        "source_min_A": stats("source_min_A"), "output_min_A": stats("output_min_A"),
        "target_min_A": stats("target_min_A"), "paired_frac_rms": stats("paired_frac_rms"),
        "nearest_target_pair_distance_mae_A": stats("nearest_target_pair_distance_mae_A"),
        "pass_rates": {key: float(np.mean([record[key] for record in records])) for key in
                       ("minimum_distance_pass", "p_coordination_pass", "fe_coordination_pass", "volume_pass")},
        "coordinate_loss_by_t": time_binned,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--count", type=int, choices=(1, 32), required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--ode-steps", type=int, default=24)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    indices = choose_indices(data, args.count)
    batch = make_batch(data, indices, device)
    encoded = encode_lattice(batch["lattice"], batch["N"])
    mean = encoded.mean(0)
    std = encoded.std(0, unbiased=False).clamp_min(1e-4)
    model = CrystalVectorField(FlowConfig(vocab_size=4), mean, std).to(device)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        mean = saved["lattice_mean"].to(device)
        std = saved["lattice_std"].to(device)
        model = CrystalVectorField(FlowConfig(**saved["config"]), mean, std).to(device)
        model.load_state_dict(saved["model"])
        result = {
            "experiment": "small_data_flow_overfit_evaluation_v1", "count": args.count,
            "checkpoint": args.checkpoint, "indices": indices,
            "evaluation": evaluate(model, batch, rounds=32 if args.count == 1 else 4, ode_steps=args.ode_steps),
        }
        output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
        result_path = output / f"overfit_{args.count}_ode{args.ode_steps}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"result": str(result_path), "evaluation": result["evaluation"]}, indent=2))
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        prepared = shuffled_training_batch(batch, mean, std)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = flow_matching_loss(model, prepared["types"], prepared["frac"],
                                               prepared["lattice_state"], prepared["mask"], geometry_only=True)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        gradient_norm = float(clip_grad_norm_(model.parameters(), 5.0))
        scaler.step(optimizer); scaler.update()
        if step == 1 or step % 100 == 0 or step == args.steps:
            record = {"step": step, "loss": float(metrics["loss"]), "frac_loss": float(metrics["frac_loss"]),
                      "lattice_loss": float(metrics["lattice_loss"]), "pair_distance_loss": float(metrics["pair_distance_loss"]),
                      "gradient_norm": gradient_norm}
            history.append(record); print(json.dumps(record), flush=True)
    result = {
        "experiment": "small_data_flow_overfit_v1", "count": args.count, "steps": args.steps,
        "seed": args.seed, "lr": args.lr, "indices": indices,
        "ids": [data["ids"][i] for i in indices], "formulas": [data["formulas"][i] for i in indices],
        "history": history,
        "evaluation": evaluate(model, batch, rounds=32 if args.count == 1 else 4, ode_steps=args.ode_steps),
    }
    output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config.__dict__, "lattice_mean": mean.cpu(),
                "lattice_std": std.cpu(), "indices": indices}, output / f"overfit_{args.count}.pt")
    (output / f"overfit_{args.count}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(output / f'overfit_{args.count}.json'), "evaluation": result["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()

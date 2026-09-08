"""Train the unconditional family-agnostic crystal flow."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from .dataset import PackedCrystalDataset, collate_crystals
from .lattice import encode_lattice, standardize_lattice
from .model import CrystalVectorField, FlowConfig, flow_matching_loss


class AtomBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: PackedCrystalDataset,
        atom_budget: int,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.atom_budget = atom_budget
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        offsets = dataset.data["offsets"]
        self.sizes = [
            int(offsets[index + 1] - offsets[index]) for index in dataset.indices
        ]
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        buckets: dict[int, list[int]] = defaultdict(list)
        for local_index, size in enumerate(self.sizes):
            buckets[min(size // 16, 11)].append(local_index)
        batches: list[list[int]] = []
        for indices in buckets.values():
            if self.shuffle:
                rng.shuffle(indices)
            current: list[int] = []
            current_max = 0
            for index in indices:
                proposal_max = max(current_max, self.sizes[index])
                if current and proposal_max * (len(current) + 1) > self.atom_budget:
                    batches.append(current)
                    current = []
                    current_max = 0
                current.append(index)
                current_max = max(current_max, self.sizes[index])
            if current:
                batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        usable = len(batches) - (len(batches) % self.world_size)
        yield from batches[:usable][self.rank :: self.world_size]

    def __len__(self) -> int:
        global_batches = max(1, math.ceil(sum(self.sizes) / self.atom_budget))
        return max(1, global_batches // self.world_size)


def lattice_statistics(dataset: PackedCrystalDataset) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.tensor(dataset.indices, dtype=torch.long)
    lattice = dataset.data["lattices"][indices].float()
    offsets = dataset.data["offsets"]
    n_atoms = offsets[indices + 1] - offsets[indices]
    encoded = encode_lattice(lattice, n_atoms)
    return encoded.mean(dim=0), encoded.std(dim=0).clamp_min(1e-4)


def prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    # Jointly permute A and X to prevent dependence on serialized atom order.
    for row, n_atoms in enumerate(batch["N"].tolist()):
        permutation = torch.randperm(n_atoms, device=device)
        batch["types"][row, :n_atoms] = batch["types"][row, permutation]
        batch["frac_coords"][row, :n_atoms] = batch["frac_coords"][row, permutation]
    encoded = encode_lattice(batch["lattice"], batch["N"])
    batch["lattice_state"] = standardize_lattice(encoded, lattice_mean, lattice_std)
    return batch


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
    geometry_only: bool = False,
    geometry_validity_weight: float = 0.0,
    coupling: str = "noise",
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = prepare_batch(
            raw_batch,
            device,
            (model.module if hasattr(model, "module") else model).lattice_mean,
            (model.module if hasattr(model, "module") else model).lattice_std,
        )
        _, metrics = flow_matching_loss(
            model,
            batch["types"],
            batch["frac_coords"],
            batch["lattice_state"],
            batch["mask"],
            geometry_only=geometry_only,
            geometry_validity_weight=geometry_validity_weight,
            coupling=coupling,
        )
        for key, value in metrics.items():
            totals[key] += float(value)
        count += 1
    if dist.is_available() and dist.is_initialized():
        keys = (
            "loss",
            "atom_loss",
            "frac_loss",
            "lattice_loss",
            "endpoint_type_loss",
            "pair_distance_loss",
            "geometry_validity_loss",
        )
        packed = torch.tensor(
            [*(totals[key] for key in keys), float(count)], device=device, dtype=torch.float64
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        totals = {key: float(packed[i]) for i, key in enumerate(keys)}
        count = int(packed[-1])
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: CrystalVectorField,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    dataset: PackedCrystalDataset,
    ema_model: CrystalVectorField | None = None,
    training_mode: str = "joint",
    geometry_validity_weight: float = 0.0,
    coupling: str = "noise",
) -> None:
    payload = model.checkpoint_payload()
    sizes = torch.diff(dataset.data["offsets"])[dataset.indices]
    payload.update(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "n_histogram": torch.bincount(sizes, minlength=int(sizes.max()) + 1),
            "element_to_index": dataset.data["element_to_index"],
            "dataset_source_hash": dataset.data.get(
                "source_first_1MiB_sha256", dataset.data.get("source_hashes")
            ),
            "dataset_version": dataset.data.get("version"),
            "dataset_filter": dataset.data.get("filter_definition"),
            "training_mode": training_mode,
            "training_objective": {
                "geometry_validity_weight": geometry_validity_weight,
                "coupling": coupling,
            },
        }
    )
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--atom-budget", type=int, default=1400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--resume")
    parser.add_argument("--init")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--geometry-validity-weight", type=float, default=0.0)
    parser.add_argument("--coupling", choices=("noise", "assignment", "polyhedral"), default="noise")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full crystal pretraining")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_dataset = PackedCrystalDataset(args.dataset, "train")
    val_dataset = PackedCrystalDataset(args.dataset, "val")
    lattice_mean, lattice_std = lattice_statistics(train_dataset)
    config = FlowConfig(
        vocab_size=len(train_dataset.data["element_to_index"]),
        hidden_dim=args.hidden,
        layers=args.layers,
        heads=args.heads,
    )
    base_model = CrystalVectorField(config, lattice_mean, lattice_std).to(device)
    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    ema_model = copy.deepcopy(base_model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    start_epoch = 0
    history: list[dict[str, float]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        base_model.load_state_dict(checkpoint["model"])
        if "ema_model" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_model"])
        else:
            ema_model.load_state_dict(base_model.state_dict())
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
    elif args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        initial_state = checkpoint.get("ema_model", checkpoint["model"])
        base_model.load_state_dict(initial_state)
        ema_model.load_state_dict(initial_state)
    model: torch.nn.Module = base_model
    if distributed:
        model = DistributedDataParallel(base_model, device_ids=[local_rank])

    train_sampler = AtomBudgetBatchSampler(
        train_dataset,
        args.atom_budget,
        shuffle=True,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    val_sampler = AtomBudgetBatchSampler(
        val_dataset,
        args.atom_budget,
        shuffle=False,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_crystals,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_crystals,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    scaler = torch.amp.GradScaler("cuda")
    out = Path(args.out)
    log_path = out.with_suffix(".jsonl")
    best_validation = min(
        (record.get("val_loss", float("inf")) for record in history),
        default=float("inf"),
    )
    for epoch in range(start_epoch, args.epochs):
        if epoch < args.warmup_epochs:
            lr = args.lr * float(epoch + 1) / max(1, args.warmup_epochs)
        else:
            progress = (epoch - args.warmup_epochs) / max(
                1, args.epochs - args.warmup_epochs - 1
            )
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        started = time.time()
        totals: dict[str, float] = defaultdict(float)
        steps = 0
        for raw_batch in train_loader:
            batch = prepare_batch(raw_batch, device, lattice_mean.to(device), lattice_std.to(device))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, metrics = flow_matching_loss(
                    model,
                    batch["types"],
                    batch["frac_coords"],
                    batch["lattice_state"],
                    batch["mask"],
                    geometry_only=args.geometry_only,
                    geometry_validity_weight=args.geometry_validity_weight,
                    coupling=args.coupling,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), base_model.parameters(), strict=True
                ):
                    ema_parameter.mul_(args.ema_decay).add_(
                        parameter.detach(), alpha=1.0 - args.ema_decay
                    )
            for key, value in metrics.items():
                totals[key] += float(value)
            steps += 1
        if distributed:
            keys = (
                "loss",
                "atom_loss",
                "frac_loss",
                "lattice_loss",
                "endpoint_type_loss",
                "pair_distance_loss",
                "geometry_validity_loss",
            )
            packed = torch.tensor(
                [*(totals[key] for key in keys), float(steps)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            totals = {key: float(packed[i]) for i, key in enumerate(keys)}
            steps = int(packed[-1])
        train_metrics = {f"train_{key}": value / steps for key, value in totals.items()}
        val_metrics = validate(
            model,
            val_loader,
            device,
            geometry_only=args.geometry_only,
            geometry_validity_weight=args.geometry_validity_weight,
            coupling=args.coupling,
        )
        record = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if rank == 0:
            history.append(record)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            save_checkpoint(
                out,
                base_model,
                optimizer,
                epoch,
                history,
                train_dataset,
                ema_model,
                "geometry" if args.geometry_only else "joint",
                args.geometry_validity_weight,
                args.coupling,
            )
            if record["val_loss"] < best_validation:
                best_validation = record["val_loss"]
                save_checkpoint(
                    out.with_name(out.stem + ".best" + out.suffix),
                    base_model, optimizer, epoch, history, train_dataset,
                    ema_model, "geometry" if args.geometry_only else "joint",
                    args.geometry_validity_weight,
                    args.coupling,
                )
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

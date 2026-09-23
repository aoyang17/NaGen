"""Retrain the crystal Flow from a swappable dataset adapter.

Examples
--------
Packed ShootingCSP dataset::

    python -m shootingcsp.training.train_flow \
      --dataset data/packed/naxl.pt --dataset-kind packed --out runs/flow.pt

JSONL dataset with A/X_frac/L_matrix fields::

    python -m shootingcsp.training.train_flow \
      --dataset data/train.jsonl --dataset-kind jsonl \
      --train-split train --val-split val --out runs/flow.pt
"""

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
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from shootingcsp.inverse.dataset import collate_crystals
from shootingcsp.inverse.lattice import encode_lattice, standardize_lattice
from shootingcsp.inverse.model import CrystalVectorField, FlowConfig, flow_matching_loss
from shootingcsp.inverse.spec import (
    DEFAULT_FE_COORDINATION_OPTIONS,
    parse_fe_coordination_options,
)
from shootingcsp.training.data import CrystalDataset, DatasetSpec, load_crystal_dataset


class AtomBudgetBatchSampler(Sampler[list[int]]):
    """Build atom-budget batches without assuming a packed dataset layout."""

    def __init__(
        self,
        dataset: CrystalDataset,
        atom_budget: int,
        *,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if atom_budget < 1:
            raise ValueError("atom_budget must be positive")
        self.sizes = tuple(int(value) for value in dataset.atom_counts)
        self.atom_budget = int(atom_budget)
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        buckets: dict[int, list[int]] = defaultdict(list)
        for local_index, size in enumerate(self.sizes):
            buckets[min(size // 16, 11)].append(local_index)
        batches: list[list[int]] = []
        for bucket in buckets.values():
            if self.shuffle:
                rng.shuffle(bucket)
            current: list[int] = []
            current_max = 0
            for index in bucket:
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


def lattice_statistics(dataset: CrystalDataset) -> tuple[torch.Tensor, torch.Tensor]:
    lattices, atom_counts = [], []
    for index in range(len(dataset)):
        item = dataset[index]
        lattices.append(item["lattice"].float())
        atom_counts.append(int(item["N"]))
    encoded = encode_lattice(torch.stack(lattices), torch.tensor(atom_counts))
    return encoded.mean(dim=0), encoded.std(dim=0).clamp_min(1e-4)


def prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
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
    *,
    max_batches: int,
    geometry_only: bool,
    geometry_validity_weight: float,
    coupling: str,
    fe_coordination_options: tuple[int, ...],
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    base_model = model.module if hasattr(model, "module") else model
    for raw_batch in loader:
        if batches >= max_batches:
            break
        batch = prepare_batch(
            raw_batch,
            device,
            base_model.lattice_mean,
            base_model.lattice_std,
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
            fe_coordination_options=fe_coordination_options,
        )
        for key, value in metrics.items():
            totals[key] += float(value)
        batches += 1
    if dist.is_available() and dist.is_initialized():
        keys = tuple(totals)
        packed = torch.tensor(
            [*(totals[key] for key in keys), float(batches)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        totals = {key: float(packed[index]) for index, key in enumerate(keys)}
        batches = int(packed[-1])
    return {key: value / max(batches, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: CrystalVectorField,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    dataset: CrystalDataset,
    *,
    ema_model: CrystalVectorField | None = None,
    training_mode: str,
    geometry_validity_weight: float,
    coupling: str,
    fe_coordination_options: tuple[int, ...],
    training_arguments: dict[str, Any],
) -> None:
    payload = model.checkpoint_payload()
    sizes = torch.tensor(dataset.atom_counts, dtype=torch.long)
    metadata = dict(dataset.metadata)
    payload.update(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "n_histogram": torch.bincount(sizes, minlength=int(sizes.max()) + 1),
            "element_to_index": dict(dataset.element_to_index),
            "dataset_metadata": metadata,
            "dataset_source_hash": metadata.get("source_hash"),
            "dataset_version": metadata.get("version"),
            "dataset_filter": metadata.get("filter_definition"),
            "training_mode": training_mode,
            "training_objective": {
                "geometry_validity_weight": geometry_validity_weight,
                "coupling": coupling,
                "fe_coordination_options": list(fe_coordination_options),
            },
            "training_arguments": training_arguments,
        }
    )
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _dataset_option(value: str) -> tuple[str, Any]:
    key, separator, raw = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("dataset option must use key=value")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def _dataset_spec(args: argparse.Namespace, split: str) -> DatasetSpec:
    options = dict(args.dataset_option)
    return DatasetSpec(
        kind=args.dataset_kind,
        path=args.dataset,
        split=split,
        options=options,
        factory=args.dataset_factory,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-kind", default="packed")
    parser.add_argument("--dataset-factory", help="external dataset factory: module:callable")
    parser.add_argument(
        "--dataset-option",
        action="append",
        default=[],
        type=_dataset_option,
        metavar="KEY=VALUE",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means all batches per epoch")
    parser.add_argument("--val-max-batches", type=int, default=100)
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
    parser.add_argument(
        "--coupling",
        choices=("noise", "assignment", "polyhedral"),
        default="noise",
    )
    parser.add_argument(
        "--fe-coordination",
        type=parse_fe_coordination_options,
        default=DEFAULT_FE_COORDINATION_OPTIONS,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.epochs < 1 or args.atom_budget < 1 or args.num_workers < 0:
        raise ValueError("epochs, atom-budget and num-workers must be valid")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    if distributed:
        if device.type != "cuda":
            raise RuntimeError("distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_dataset = load_crystal_dataset(_dataset_spec(args, args.train_split))
    val_dataset = load_crystal_dataset(_dataset_spec(args, args.val_split))
    lattice_mean, lattice_std = lattice_statistics(train_dataset)
    config = FlowConfig(
        vocab_size=max(train_dataset.element_to_index.values()),
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
        ema_model.load_state_dict(checkpoint.get("ema_model", base_model.state_dict()))
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
    loader_kwargs = {
        "collate_fn": collate_crystals,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **loader_kwargs)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    log_path = args.out.with_suffix(".jsonl")
    best_validation = min(
        (record.get("val_loss", float("inf")) for record in history),
        default=float("inf"),
    )
    training_arguments = vars(args) | {
        "fe_coordination": list(args.fe_coordination),
        "world_size": world_size,
    }

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
            batch = prepare_batch(
                raw_batch,
                device,
                lattice_mean.to(device),
                lattice_std.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                loss, metrics = flow_matching_loss(
                    model,
                    batch["types"],
                    batch["frac_coords"],
                    batch["lattice_state"],
                    batch["mask"],
                    geometry_only=args.geometry_only,
                    geometry_validity_weight=args.geometry_validity_weight,
                    coupling=args.coupling,
                    fe_coordination_options=args.fe_coordination,
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
            if args.max_steps and steps >= args.max_steps:
                break
        if distributed:
            keys = tuple(totals)
            packed = torch.tensor(
                [*(totals[key] for key in keys), float(steps)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            totals = {key: float(packed[index]) for index, key in enumerate(keys)}
            steps = int(packed[-1])
        train_metrics = {
            f"train_{key}": value / max(steps, 1) for key, value in totals.items()
        }
        val_metrics = validate(
            model,
            val_loader,
            device,
            max_batches=args.val_max_batches,
            geometry_only=args.geometry_only,
            geometry_validity_weight=args.geometry_validity_weight,
            coupling=args.coupling,
            fe_coordination_options=args.fe_coordination,
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
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            save_checkpoint(
                args.out,
                base_model,
                optimizer,
                epoch,
                history,
                train_dataset,
                ema_model=ema_model,
                training_mode="geometry" if args.geometry_only else "joint",
                geometry_validity_weight=args.geometry_validity_weight,
                coupling=args.coupling,
                fe_coordination_options=args.fe_coordination,
                training_arguments=training_arguments,
            )
            if record["val_loss"] < best_validation:
                best_validation = record["val_loss"]
                save_checkpoint(
                    args.out.with_name(args.out.stem + ".best" + args.out.suffix),
                    base_model,
                    optimizer,
                    epoch,
                    history,
                    train_dataset,
                    ema_model=ema_model,
                    training_mode="geometry" if args.geometry_only else "joint",
                    geometry_validity_weight=args.geometry_validity_weight,
                    coupling=args.coupling,
                    fe_coordination_options=args.fe_coordination,
                    training_arguments=training_arguments,
                )
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

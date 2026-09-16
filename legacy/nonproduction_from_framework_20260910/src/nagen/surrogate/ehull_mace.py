"""Periodic equivariant message-passing surrogate for direct E_hull regression.

The model is trained from random initialization on NaGen labels.  It uses an
e3nn tensor-product message-passing trunk (the core operation used by
MACE-style models), mean pools atom features for an intensive target, and
combines them with composition and invariant lattice features.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.v2106.gate_points_message_passing import MessagePassing
from pymatgen.core import Lattice, Structure
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 11
    embedding_dim: int = 32
    hidden_mul: int = 16
    layers: int = 3
    lmax: int = 2
    radial_basis: int = 10
    radial_hidden: int = 64
    readout_hidden: int = 192
    cutoff_A: float = 6.0
    max_neighbors: int = 64
    average_neighbors: float = 48.0


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20260904
    batch_size: int = 6
    epochs: int = 80
    patience: int = 12
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-6
    huber_delta: float = 0.5
    num_workers: int = 4
    threshold_eV_atom: float = 0.15


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lean_labels(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for expected, line in enumerate(handle):
            row = json.loads(line)
            if int(row["source_index"]) != expected:
                raise ValueError(f"non-contiguous label index at {expected}")
            rows.append(row)
    return rows


def build_graph_dataset(
    packed_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    cutoff_A: float = 6.0,
    max_neighbors: int = 64,
) -> dict[str, Any]:
    """Attach labels and an exact periodic neighbor graph to the packed data."""
    packed_path = Path(packed_path)
    labels_path = Path(labels_path)
    output_path = Path(output_path)
    packed = torch.load(packed_path, map_location="cpu", weights_only=False)
    labels = _lean_labels(labels_path)
    if len(labels) != len(packed["ids"]):
        raise ValueError("packed structures and labels have different lengths")

    index_to_element = {
        int(index): element for element, index in packed["element_to_index"].items()
    }
    edge_offsets = [0]
    edge_src_rows: list[np.ndarray] = []
    edge_dst_rows: list[np.ndarray] = []
    edge_shift_rows: list[np.ndarray] = []
    neighbor_counts: list[int] = []
    total = len(labels)
    for structure_index in range(total):
        atom_start = int(packed["offsets"][structure_index])
        atom_stop = int(packed["offsets"][structure_index + 1])
        types = packed["types"][atom_start:atom_stop].numpy()
        frac = packed["frac_coords"][atom_start:atom_stop].numpy()
        lattice = packed["lattices"][structure_index].numpy()
        species = [index_to_element[int(item)] for item in types]
        structure = Structure(Lattice(lattice), species, frac, coords_are_cartesian=False)
        center, neighbor, shifts, distances = structure.get_neighbor_list(
            cutoff_A, numerical_tol=1.0e-8, exclude_self=True
        )
        if len(center):
            # Deterministically retain the closest max_neighbors around each
            # center.  This bounds memory while preserving the local shell.
            selected = []
            for atom in range(atom_stop - atom_start):
                candidates = np.where(center == atom)[0]
                if len(candidates) > max_neighbors:
                    order = np.lexsort((neighbor[candidates], distances[candidates]))
                    candidates = candidates[order[:max_neighbors]]
                selected.append(candidates)
            keep = np.concatenate(selected) if selected else np.empty(0, dtype=int)
            center = center[keep]
            neighbor = neighbor[keep]
            shifts = shifts[keep]
        edge_src_rows.append(np.asarray(neighbor, dtype=np.int16))
        edge_dst_rows.append(np.asarray(center, dtype=np.int16))
        edge_shift_rows.append(np.asarray(shifts, dtype=np.int8))
        edge_offsets.append(edge_offsets[-1] + len(center))
        neighbor_counts.append(len(center) // max(1, atom_stop - atom_start))
        if (structure_index + 1) % 500 == 0 or structure_index + 1 == total:
            print(json.dumps({
                "graph_structures": structure_index + 1,
                "total": total,
                "edges": edge_offsets[-1],
            }), flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "nagen-ehull-graph-v1",
        "packed_source": str(packed_path.resolve()),
        "labels_source": str(labels_path.resolve()),
        "cutoff_A": float(cutoff_A),
        "max_neighbors": int(max_neighbors),
        "element_to_index": packed["element_to_index"],
        "types": packed["types"],
        "frac_coords": packed["frac_coords"],
        "offsets": packed["offsets"],
        "lattices": packed["lattices"],
        "split": packed["split"],
        "ids": packed["ids"],
        "formulas": packed["formulas"],
        "E_hull": torch.tensor([row["E_hull"] for row in labels], dtype=torch.float32),
        "E_UMA": torch.tensor([row["E_UMA"] for row in labels], dtype=torch.float32),
        "E_MP": torch.tensor([row["E_MP"] for row in labels], dtype=torch.float32),
        "upstream_error_present": torch.tensor(
            [row["upstream_error_present"] for row in labels], dtype=torch.bool
        ),
        "chemical_systems": [row["chemical_system"] for row in labels],
        "edge_offsets": torch.tensor(edge_offsets, dtype=torch.int64),
        "edge_src": torch.from_numpy(np.concatenate(edge_src_rows)),
        "edge_dst": torch.from_numpy(np.concatenate(edge_dst_rows)),
        "edge_shift": torch.from_numpy(np.concatenate(edge_shift_rows)),
    }
    torch.save(payload, output_path)
    manifest = {
        "version": payload["version"],
        "structures": total,
        "atoms": int(packed["types"].numel()),
        "edges": int(payload["edge_src"].numel()),
        "cutoff_A": cutoff_A,
        "max_neighbors": max_neighbors,
        "mean_retained_neighbors": float(np.mean(neighbor_counts)),
        "split_counts": {
            name: int((packed["split"] == code).sum())
            for name, code in {"train": 0, "val": 1, "test": 2}.items()
        },
        "target": {
            "min": float(payload["E_hull"].min()),
            "mean": float(payload["E_hull"].mean()),
            "max": float(payload["E_hull"].max()),
        },
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


class CrystalGraphDataset(Dataset):
    def __init__(self, data: dict[str, Any], split_code: int) -> None:
        self.data = data
        self.indices = torch.where(data["split"] == split_code)[0].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        atom_start = int(self.data["offsets"][index])
        atom_stop = int(self.data["offsets"][index + 1])
        edge_start = int(self.data["edge_offsets"][index])
        edge_stop = int(self.data["edge_offsets"][index + 1])
        return {
            "index": index,
            "types": self.data["types"][atom_start:atom_stop].long(),
            "frac": self.data["frac_coords"][atom_start:atom_stop].float(),
            "lattice": self.data["lattices"][index].float(),
            "edge_src": self.data["edge_src"][edge_start:edge_stop].long(),
            "edge_dst": self.data["edge_dst"][edge_start:edge_stop].long(),
            "edge_shift": self.data["edge_shift"][edge_start:edge_stop].float(),
            "target": self.data["E_hull"][index].float(),
        }


def collate_graphs(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    node_offsets = np.cumsum([0] + [len(row["types"]) for row in rows])
    node_batch = torch.cat([
        torch.full((len(row["types"]),), batch, dtype=torch.long)
        for batch, row in enumerate(rows)
    ])
    return {
        "index": torch.tensor([row["index"] for row in rows], dtype=torch.long),
        "types": torch.cat([row["types"] for row in rows]),
        "frac": torch.cat([row["frac"] for row in rows]),
        "lattice": torch.stack([row["lattice"] for row in rows]),
        "node_batch": node_batch,
        "edge_src": torch.cat([
            row["edge_src"] + int(node_offsets[batch]) for batch, row in enumerate(rows)
        ]),
        "edge_dst": torch.cat([
            row["edge_dst"] + int(node_offsets[batch]) for batch, row in enumerate(rows)
        ]),
        "edge_shift": torch.cat([row["edge_shift"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
    }


class PeriodicMACERegressor(nn.Module):
    """MACE-style e3nn tensor-product network with intensive readout."""

    def __init__(
        self,
        config: ModelConfig,
        target_mean: float,
        target_std: float,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        irreps_in = o3.Irreps(f"{config.embedding_dim}x0e")
        irreps_hidden = o3.Irreps([
            (config.hidden_mul, (ell, parity))
            for ell in range(config.lmax + 1)
            for parity in (-1, 1)
        ])
        irreps_out = o3.Irreps(f"{config.embedding_dim}x0e")
        self.edge_irreps = o3.Irreps.spherical_harmonics(config.lmax)
        self.message_passing = MessagePassing(
            irreps_node_sequence=(
                [irreps_in]
                + [irreps_hidden] * config.layers
                + [irreps_out]
            ),
            irreps_node_attr="1x0e",
            irreps_edge_attr=self.edge_irreps,
            fc_neurons=[config.radial_basis, config.radial_hidden],
            num_neighbors=config.average_neighbors,
        )
        global_dim = config.embedding_dim + (config.vocab_size - 1) + 8
        self.readout = nn.Sequential(
            nn.Linear(global_dim, config.readout_hidden),
            nn.SiLU(),
            nn.LayerNorm(config.readout_hidden),
            nn.Linear(config.readout_hidden, config.readout_hidden // 2),
            nn.SiLU(),
            nn.Linear(config.readout_hidden // 2, 1),
        )
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))

    @staticmethod
    def _scatter_mean(
        values: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        result = values.new_zeros((batch_size, values.shape[-1]))
        result.index_add_(0, batch, values)
        counts = torch.bincount(batch, minlength=batch_size).to(values.dtype)
        return result / counts.clamp_min(1)[:, None]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        types = batch["types"]
        frac = batch["frac"]
        lattice = batch["lattice"]
        node_batch = batch["node_batch"]
        edge_src = batch["edge_src"]
        edge_dst = batch["edge_dst"]
        shift = batch["edge_shift"]
        batch_size = lattice.shape[0]

        delta = frac[edge_src] + shift - frac[edge_dst]
        edge_lattice = lattice[node_batch[edge_dst]]
        edge_vec = torch.einsum("ei,eij->ej", delta, edge_lattice)
        edge_length = torch.linalg.vector_norm(edge_vec, dim=-1)
        edge_attr = o3.spherical_harmonics(
            range(self.config.lmax + 1), edge_vec,
            normalize=True, normalization="component",
        )
        edge_scalars = soft_one_hot_linspace(
            edge_length, 0.0, self.config.cutoff_A,
            self.config.radial_basis, basis="smooth_finite", cutoff=True,
        ) * math.sqrt(self.config.radial_basis)
        node_attr = frac.new_ones((len(types), 1))
        if "species_probabilities" in batch:
            probabilities = batch["species_probabilities"].to(frac.dtype)
            expected_shape = (len(types), self.config.vocab_size - 1)
            if tuple(probabilities.shape) != expected_shape:
                raise ValueError(
                    f"species_probabilities must have shape {expected_shape}"
                )
            embedded_species = probabilities @ self.embedding.weight[1:]
            composition_input = probabilities
        else:
            embedded_species = self.embedding(types)
            composition_input = F.one_hot(
                types, num_classes=self.config.vocab_size
            ).to(frac.dtype)[:, 1:]
        node_features = self.message_passing(
            embedded_species, node_attr,
            edge_src, edge_dst, edge_attr, edge_scalars,
        )
        pooled = self._scatter_mean(node_features, node_batch, batch_size)

        composition = self._scatter_mean(
            composition_input, node_batch, batch_size
        )
        counts = torch.bincount(node_batch, minlength=batch_size).to(frac.dtype)
        gram = lattice @ lattice.transpose(-1, -2)
        lengths = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
        volume = torch.linalg.det(lattice).abs().clamp_min(1e-12)
        scale = volume.pow(1.0 / 3.0)
        shape_lengths = lengths / scale[:, None]
        cosines = torch.stack((
            gram[:, 1, 2] / (lengths[:, 1] * lengths[:, 2]).clamp_min(1e-12),
            gram[:, 0, 2] / (lengths[:, 0] * lengths[:, 2]).clamp_min(1e-12),
            gram[:, 0, 1] / (lengths[:, 0] * lengths[:, 1]).clamp_min(1e-12),
        ), dim=-1)
        global_features = torch.cat((
            pooled,
            composition,
            torch.log1p(counts)[:, None],
            torch.log1p(volume / counts.clamp_min(1))[:, None],
            shape_lengths,
            cosines,
        ), dim=-1)
        normalized = self.readout(global_features).squeeze(-1)
        return normalized * self.target_std + self.target_mean


def load_surrogate(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[PeriodicMACERegressor, dict[str, Any]]:
    """Load a trained surrogate and its complete manifest."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("version") != "nagen-ehull-mace-surrogate-v1":
        raise ValueError("unsupported E_hull surrogate checkpoint")
    model = PeriodicMACERegressor(
        ModelConfig(**payload["model_config"]),
        float(payload["target_mean"]),
        float(payload["target_std"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def structure_to_model_batch(
    structure: Structure,
    element_to_index: dict[str, int],
    cutoff_A: float,
    max_neighbors: int,
) -> dict[str, torch.Tensor]:
    """Convert one pymatgen Structure to the model's differentiable tensors."""
    elements = [str(site.specie) for site in structure]
    unknown = sorted(set(elements) - set(element_to_index))
    if unknown:
        raise ValueError(f"unsupported elements: {unknown}")
    center, neighbor, shifts, distances = structure.get_neighbor_list(
        cutoff_A, numerical_tol=1.0e-8, exclude_self=True
    )
    selected = []
    for atom in range(len(structure)):
        candidates = np.where(center == atom)[0]
        if len(candidates) > max_neighbors:
            order = np.lexsort((neighbor[candidates], distances[candidates]))
            candidates = candidates[order[:max_neighbors]]
        selected.append(candidates)
    keep = np.concatenate(selected) if selected else np.empty(0, dtype=int)
    return {
        "index": torch.zeros(1, dtype=torch.long),
        "types": torch.tensor([element_to_index[item] for item in elements]),
        "frac": torch.tensor(structure.frac_coords, dtype=torch.float32),
        "lattice": torch.tensor(structure.lattice.matrix, dtype=torch.float32)[None],
        "node_batch": torch.zeros(len(structure), dtype=torch.long),
        "edge_src": torch.tensor(neighbor[keep], dtype=torch.long),
        "edge_dst": torch.tensor(center[keep], dtype=torch.long),
        "edge_shift": torch.tensor(shifts[keep], dtype=torch.float32),
        "target": torch.full((1,), float("nan")),
    }


@torch.no_grad()
def predict_structure_e_hull(
    model: PeriodicMACERegressor,
    structure: Structure,
    element_to_index: dict[str, int],
) -> float:
    """Predict E_hull in eV/atom for one pymatgen Structure."""
    device = next(model.parameters()).device
    batch = structure_to_model_batch(
        structure,
        element_to_index,
        model.config.cutoff_A,
        model.config.max_neighbors,
    )
    return float(model(_to_device(batch, device)).item())


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _sample_weights(target: torch.Tensor, train_indices: torch.Tensor) -> torch.Tensor:
    # Equalize 25-meV target bands without allowing rare bins to dominate.
    edges = torch.arange(0.0, 0.575, 0.025)
    train_bins = torch.bucketize(target[train_indices], edges)
    counts = torch.bincount(train_bins, minlength=len(edges) + 1).float()
    weights = counts[train_bins].clamp_min(1).rsqrt()
    weights = weights / weights.mean()
    result = torch.ones_like(target)
    result[train_indices] = weights.clamp(0.35, 4.0)
    return result


def regression_metrics(
    truth: np.ndarray, prediction: np.ndarray, threshold: float = 0.15
) -> dict[str, float | int]:
    error = prediction - truth
    ss_res = float(np.square(error).sum())
    ss_tot = float(np.square(truth - truth.mean()).sum())
    true_high = truth > threshold
    pred_high = prediction > threshold
    tp = int(np.logical_and(true_high, pred_high).sum())
    fp = int(np.logical_and(~true_high, pred_high).sum())
    fn = int(np.logical_and(true_high, ~pred_high).sum())
    tn = int(np.logical_and(~true_high, ~pred_high).sum())
    return {
        "count": len(truth),
        "MAE_eV_atom": float(np.abs(error).mean()),
        "RMSE_eV_atom": float(np.sqrt(np.square(error).mean())),
        "R2": 1.0 - ss_res / max(ss_tot, 1.0e-12),
        "Spearman": float(spearmanr(truth, prediction).statistic),
        "threshold_eV_atom": threshold,
        "threshold_accuracy": float((true_high == pred_high).mean()),
        "high_precision": tp / max(tp + fp, 1),
        "high_recall": tp / max(tp + fn, 1),
        "high_F1": 2 * tp / max(2 * tp + fp + fn, 1),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    indices, truths, predictions = [], [], []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        prediction = model(batch)
        indices.append(batch["index"].cpu())
        truths.append(batch["target"].cpu())
        predictions.append(prediction.cpu())
    return (
        torch.cat(indices).numpy(),
        torch.cat(truths).numpy(),
        torch.cat(predictions).numpy(),
    )


def train_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device_name: str,
    overfit_count: int = 0,
) -> dict[str, Any]:
    seed_everything(train_config.seed)
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    train_set = CrystalGraphDataset(data, 0)
    val_set = CrystalGraphDataset(data, 1)
    test_set = CrystalGraphDataset(data, 2)
    if overfit_count:
        chosen = train_set.indices[:overfit_count]
        train_set.indices = chosen
        val_set.indices = chosen
        test_set.indices = chosen
    loader_args = {
        "batch_size": train_config.batch_size,
        "num_workers": train_config.num_workers,
        "collate_fn": collate_graphs,
        "pin_memory": device_name.startswith("cuda"),
        "persistent_workers": train_config.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    train_eval_loader = DataLoader(train_set, shuffle=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    train_indices = torch.tensor(train_set.indices)
    target_mean = float(data["E_hull"][train_indices].mean())
    target_std = float(data["E_hull"][train_indices].std().clamp_min(1.0e-4))
    weights = _sample_weights(data["E_hull"], train_indices)
    device = torch.device(device_name)
    model = PeriodicMACERegressor(model_config, target_mean, target_std).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1.0e-6
    )
    best_mae = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            prediction = model(batch)
            normalized_error = (prediction - batch["target"]) / target_std
            per_item = F.huber_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
                delta=train_config.huber_delta,
                reduction="none",
            )
            batch_weights = weights[raw_batch["index"]].to(device)
            loss = (per_item * batch_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(prediction)
            count += len(prediction)
        _, val_truth, val_prediction = predict_loader(model, val_loader, device)
        val_metrics = regression_metrics(
            val_truth, val_prediction, train_config.threshold_eV_atom
        )
        val_mae = float(val_metrics["MAE_eV_atom"])
        scheduler.step(val_mae)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
            "val_MAE_eV_atom": val_mae,
            "val_RMSE_eV_atom": val_metrics["RMSE_eV_atom"],
            "val_R2": val_metrics["R2"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_mae < best_mae - 1.0e-6:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= train_config.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        "version": "nagen-ehull-mace-surrogate-v1",
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "target_mean": target_mean,
        "target_std": target_std,
        "element_to_index": data["element_to_index"],
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "best_validation_MAE_eV_atom": best_mae,
        "dataset": str(dataset_path.resolve()),
    }
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(checkpoint, checkpoint_path)

    split_results = {}
    prediction_rows = []
    for name, loader in (
        ("train", train_eval_loader), ("validation", val_loader), ("test", test_loader)
    ):
        indices, truth, prediction = predict_loader(model, loader, device)
        split_results[name] = regression_metrics(
            truth, prediction, train_config.threshold_eV_atom
        )
        if name == "test":
            for index, target, predicted in zip(indices, truth, prediction, strict=True):
                prediction_rows.append({
                    "source_index": int(index),
                    "material_id": data["ids"][int(index)],
                    "formula": data["formulas"][int(index)],
                    "chemical_system": data["chemical_systems"][int(index)],
                    "E_hull_true": float(target),
                    "E_hull_predicted": float(predicted),
                    "absolute_error": float(abs(predicted - target)),
                })
    report = {
        "version": checkpoint["version"],
        "trained_from_scratch": True,
        "pretrained_weights": None,
        "architecture": "periodic e3nn tensor-product MACE-style scalar regressor",
        "dataset": str(dataset_path.resolve()),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "epochs_completed": len(history),
        "best_validation_MAE_eV_atom": best_mae,
        "metrics": split_results,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    with (output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--packed", required=True)
    build.add_argument("--labels", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--cutoff", type=float, default=6.0)
    build.add_argument("--max-neighbors", type=int, default=64)
    train = subparsers.add_parser("train")
    train.add_argument("--dataset", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--patience", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=6)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=3.0e-4)
    train.add_argument("--hidden-mul", type=int, default=16)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--overfit-count", type=int, default=0)
    args = parser.parse_args()
    if args.command == "build":
        build_graph_dataset(args.packed, args.labels, args.out, args.cutoff, args.max_neighbors)
    else:
        data = torch.load(args.dataset, map_location="cpu", weights_only=False)
        model_config = ModelConfig(
            cutoff_A=float(data["cutoff_A"]),
            average_neighbors=float(min(data["max_neighbors"], 48)),
            hidden_mul=args.hidden_mul,
            layers=args.layers,
        )
        train_config = TrainConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            num_workers=args.workers,
            learning_rate=args.learning_rate,
        )
        train_model(
            args.dataset, args.out, model_config, train_config,
            args.device, args.overfit_count,
        )


if __name__ == "__main__":
    main()

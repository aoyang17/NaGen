"""Differentiable ensemble for post-UMA-relaxation energy and geometry."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._io import sha256_file
from .novelty import DescriptorConfig, pair_channel_lookup


GEOMETRY_TARGETS = (
    "minimum_pbc_distances",
    "p_coordination_eq_4",
    "fe_coordination_in_4_5_6",
    "volume_per_atom_range",
)
ELEMENT_TO_INDEX = {"Na": 0, "Fe": 1, "P": 2, "O": 3}


def soft_crystal_descriptor(
    probabilities: torch.Tensor,
    frac: torch.Tensor,
    lattice: torch.Tensor,
    mask: torch.Tensor,
    config: DescriptorConfig = DescriptorConfig(),
) -> torch.Tensor:
    """Differentiable descriptor supporting hard or soft element identities."""
    dtype, device = frac.dtype, frac.device
    probabilities = probabilities.to(dtype) * mask.unsqueeze(-1)
    composition = probabilities.sum(dim=1)
    composition = composition / mask.sum(dim=1, keepdim=True).clamp_min(1)
    delta = frac[:, :, None, :] - frac[:, None, :, :]
    delta = delta - torch.floor(delta + 0.5)
    cart = torch.einsum("bijd,bdk->bijk", delta, lattice)
    distances = torch.linalg.vector_norm(cart, dim=-1)
    centers = torch.linspace(
        config.rdf_min_A, config.rdf_max_A, config.rdf_bins,
        device=device, dtype=dtype,
    )
    lookup = pair_channel_lookup(config.vocab_size, device)
    batch, n_max, _ = probabilities.shape
    rdf = torch.zeros(
        batch, config.pair_channels, config.rdf_bins,
        device=device, dtype=dtype,
    )
    upper = torch.triu(
        torch.ones(n_max, n_max, dtype=torch.bool, device=device), diagonal=1
    )
    for row in range(batch):
        pair_i, pair_j = torch.where(
            upper & mask[row, :, None] & mask[row, None, :]
        )
        if pair_i.numel() == 0:
            continue
        radial = torch.exp(-0.5 * (
            (distances[row, pair_i, pair_j, None] - centers[None, :])
            / config.rdf_sigma_A
        ).square())
        for first in range(config.vocab_size):
            for second in range(first, config.vocab_size):
                channel = int(lookup[first, second])
                if first == second:
                    weights = (
                        probabilities[row, pair_i, first]
                        * probabilities[row, pair_j, second]
                    )
                else:
                    weights = (
                        probabilities[row, pair_i, first]
                        * probabilities[row, pair_j, second]
                        + probabilities[row, pair_i, second]
                        * probabilities[row, pair_j, first]
                    )
                rdf[row, channel] = (weights[:, None] * radial).sum(dim=0)
        rdf[row] /= float(pair_i.numel())
    gram = lattice @ lattice.transpose(-1, -2)
    lengths = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
    volume = torch.linalg.det(lattice).abs().clamp_min(1e-12)
    scale = volume.pow(1.0 / 3.0)
    normalized_lengths = lengths / scale[:, None]
    cosines = torch.stack((
        gram[:, 1, 2] / (lengths[:, 1] * lengths[:, 2]).clamp_min(1e-12),
        gram[:, 0, 2] / (lengths[:, 0] * lengths[:, 2]).clamp_min(1e-12),
        gram[:, 0, 1] / (lengths[:, 0] * lengths[:, 1]).clamp_min(1e-12),
    ), dim=-1)
    volume_per_atom = volume / mask.sum(dim=1).clamp_min(1).to(dtype)
    lattice_features = torch.cat((
        normalized_lengths, cosines, torch.log1p(volume_per_atom)[:, None]
    ), dim=-1)
    return torch.cat((composition, rdf.flatten(1), lattice_features), dim=-1)


class RelaxationMember(nn.Module):
    def __init__(self, dimension: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dimension, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
        )
        self.energy = nn.Linear(hidden, 2)  # final E_UMA and delta E_relax
        self.geometry = nn.Linear(hidden, len(GEOMETRY_TARGETS))

    def forward(self, features: torch.Tensor):
        latent = self.trunk(features)
        return self.energy(latent), self.geometry(latent)


class RelaxationSurrogateEnsemble(nn.Module):
    def __init__(
        self, members: list[RelaxationMember], feature_mean: torch.Tensor,
        feature_std: torch.Tensor, target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)

    def forward_descriptor(self, descriptor: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = (descriptor - self.feature_mean) / self.feature_std
        energy_rows, geometry_rows = zip(
            *(member(normalized) for member in self.members), strict=True
        )
        energy = torch.stack(energy_rows) * self.target_std + self.target_mean
        logits = torch.stack(geometry_rows)
        probabilities = torch.sigmoid(logits)
        return {
            "final_energy_mean": energy[..., 0].mean(dim=0),
            "final_energy_std": energy[..., 0].std(dim=0, unbiased=False),
            "delta_energy_mean": energy[..., 1].mean(dim=0),
            "delta_energy_std": energy[..., 1].std(dim=0, unbiased=False),
            "geometry_probability_mean": probabilities.mean(dim=0),
            "geometry_probability_std": probabilities.std(dim=0, unbiased=False),
        }

    def forward_structure(
        self, probabilities: torch.Tensor, frac: torch.Tensor,
        lattice: torch.Tensor, mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.forward_descriptor(
            soft_crystal_descriptor(probabilities, frac, lattice, mask)
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        members = [
            RelaxationMember(payload["dimension"], payload["hidden"])
            for _ in payload["members"]
        ]
        for member, state in zip(members, payload["members"], strict=True):
            member.load_state_dict(state)
        model = cls(
            members, payload["feature_mean"], payload["feature_std"],
            payload["target_mean"], payload["target_std"],
        )
        return model.to(device).eval(), payload["manifest"]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2:
        return float("nan")
    return float(np.corrcoef(_rank(first), _rank(second))[0, 1])


def _load_pairs(label_roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for label_root in label_roots:
        paths = list(label_root.glob("shard_*/records/sample_*.json"))
        paths.extend(label_root.glob("records/sample_*.json"))
        for path in sorted(paths):
            row = json.loads(path.read_text())
            if row.get("status") == "complete" and row.get("converged"):
                if "delta_E_relax_eV_atom" not in row:
                    row["delta_E_relax_eV_atom"] = (
                        float(row["final_E_UMA_eV_atom"])
                        - float(row["initial_E_UMA_eV_atom"])
                    )
                row["_path"] = str(path)
                rows.append(row)
    return rows


def _descriptor(row: dict[str, Any]) -> torch.Tensor:
    structure = row["initial_structure"]
    elements = structure["elements"]
    indices = torch.tensor([[ELEMENT_TO_INDEX[item] for item in elements]])
    probabilities = F.one_hot(indices, num_classes=4).float()
    frac = torch.tensor(structure["frac_coords"], dtype=torch.float32)[None]
    lattice = torch.tensor(structure["lattice"], dtype=torch.float32)[None]
    mask = torch.ones(1, len(elements), dtype=torch.bool)
    with torch.no_grad():
        return soft_crystal_descriptor(probabilities, frac, lattice, mask)[0]


def train(args: argparse.Namespace) -> None:
    label_roots = [Path(path) for path in args.label_campaign]
    rows = _load_pairs(label_roots)
    if len(rows) < args.minimum_pairs:
        raise RuntimeError(
            f"only {len(rows)} converged pairs; need at least {args.minimum_pairs}"
        )
    features = torch.stack([_descriptor(row) for row in rows])
    targets = torch.tensor([
        [row["final_E_UMA_eV_atom"], row["delta_E_relax_eV_atom"]]
        for row in rows
    ], dtype=torch.float32)
    geometry = torch.tensor([
        [float(row["final_feasibility"]["checks"][name]) for name in GEOMETRY_TARGETS]
        for row in rows
    ], dtype=torch.float32)
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(rows), generator=generator)
    n_test = max(1, round(0.15 * len(rows)))
    n_validation = max(1, round(0.15 * len(rows)))
    test_indices = permutation[:n_test]
    validation_indices = permutation[n_test:n_test + n_validation]
    train_indices = permutation[n_test + n_validation:]
    feature_mean = features[train_indices].mean(dim=0)
    feature_std = features[train_indices].std(dim=0).clamp_min(1.0e-5)
    target_mean = targets[train_indices].mean(dim=0)
    target_std = targets[train_indices].std(dim=0).clamp_min(1.0e-5)
    normalized = (features - feature_mean) / feature_std
    normalized_target = (targets - target_mean) / target_std
    positive = geometry[train_indices].sum(dim=0)
    negative = len(train_indices) - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(0.25, 20.0)
    device = torch.device(args.device)
    members: list[RelaxationMember] = []
    histories = []
    for member_index in range(args.ensemble_size):
        torch.manual_seed(args.seed + member_index)
        member = RelaxationMember(features.shape[1], args.hidden).to(device)
        optimizer = torch.optim.AdamW(member.parameters(), lr=args.learning_rate)
        best_loss = float("inf")
        best_state = None
        stale = 0
        member_history = []
        for epoch in range(args.epochs):
            member.train()
            boot = train_indices[torch.randint(
                len(train_indices), (len(train_indices),), generator=generator
            )]
            for start in range(0, len(boot), args.batch_size):
                index = boot[start:start + args.batch_size]
                energy_pred, geometry_logits = member(normalized[index].to(device))
                energy_loss = F.smooth_l1_loss(
                    energy_pred, normalized_target[index].to(device)
                )
                geometry_loss = F.binary_cross_entropy_with_logits(
                    geometry_logits, geometry[index].to(device),
                    pos_weight=positive_weight.to(device),
                )
                loss = energy_loss + args.geometry_weight * geometry_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            member.eval()
            with torch.no_grad():
                val_energy, val_geometry = member(normalized[validation_indices].to(device))
                val_loss = F.smooth_l1_loss(
                    val_energy, normalized_target[validation_indices].to(device)
                ) + args.geometry_weight * F.binary_cross_entropy_with_logits(
                    val_geometry, geometry[validation_indices].to(device),
                    pos_weight=positive_weight.to(device),
                )
            value = float(val_loss)
            member_history.append(value)
            if value < best_loss - 1.0e-6:
                best_loss = value
                best_state = {key: value.detach().cpu().clone() for key, value in member.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        member.load_state_dict(best_state)
        members.append(member.cpu())
        histories.append(member_history)
    ensemble = RelaxationSurrogateEnsemble(
        members, feature_mean, feature_std, target_mean, target_std
    ).eval()
    with torch.no_grad():
        prediction = ensemble.forward_descriptor(features[test_indices])
    final_truth = targets[test_indices, 0].numpy()
    final_prediction = prediction["final_energy_mean"].numpy()
    delta_truth = targets[test_indices, 1].numpy()
    delta_prediction = prediction["delta_energy_mean"].numpy()
    probability = prediction["geometry_probability_mean"].numpy()
    geometry_truth = geometry[test_indices].numpy()
    metrics = {
        "pair_count": len(rows),
        "split": {
            "train": len(train_indices), "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "final_energy_MAE_eV_atom": float(np.abs(final_truth - final_prediction).mean()),
        "delta_energy_MAE_eV_atom": float(np.abs(delta_truth - delta_prediction).mean()),
        "final_energy_spearman": _spearman(final_truth, final_prediction),
        "delta_energy_spearman": _spearman(delta_truth, delta_prediction),
        "geometry_accuracy": {
            name: float(((probability[:, index] >= 0.5) == geometry_truth[:, index]).mean())
            for index, name in enumerate(GEOMETRY_TARGETS)
        },
        "geometry_positive_rate": {
            name: float(geometry[:, index].mean())
            for index, name in enumerate(GEOMETRY_TARGETS)
        },
    }
    manifest = {
        "version": "relaxation-surrogate-ensemble-v1",
        "label_campaigns": [str(path.resolve()) for path in label_roots],
        "label_record_hashes": [sha256_file(row["_path"]) for row in rows],
        "descriptor": asdict(DescriptorConfig()),
        "geometry_targets": GEOMETRY_TARGETS,
        "ensemble_size": args.ensemble_size,
        "metrics": metrics,
    }
    payload = {
        "dimension": features.shape[1], "hidden": args.hidden,
        "members": [member.state_dict() for member in members],
        "feature_mean": feature_mean, "feature_std": feature_std,
        "target_mean": target_mean, "target_std": target_std,
        "manifest": manifest,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-campaign", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum-pairs", type=int, default=100)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--geometry-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()

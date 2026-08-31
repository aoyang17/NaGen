"""Fixed invariant descriptor phi(C) and exact nearest-reference novelty score."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .dataset import PackedCrystalDataset, collate_crystals


@dataclass(frozen=True)
class DescriptorConfig:
    vocab_size: int = 4
    rdf_bins: int = 32
    rdf_min_A: float = 1.0
    rdf_max_A: float = 8.0
    rdf_sigma_A: float = 0.16

    @property
    def pair_channels(self) -> int:
        return self.vocab_size * (self.vocab_size + 1) // 2

    @property
    def dimension(self) -> int:
        # composition + element-resolved RDF + canonical lattice shape/volume
        return self.vocab_size + self.pair_channels * self.rdf_bins + 7


def pair_channel_lookup(vocab_size: int, device: torch.device) -> torch.Tensor:
    lookup = torch.empty(vocab_size, vocab_size, dtype=torch.long, device=device)
    channel = 0
    for i in range(vocab_size):
        for j in range(i, vocab_size):
            lookup[i, j] = channel
            lookup[j, i] = channel
            channel += 1
    return lookup


def crystal_descriptor(
    types: torch.Tensor,
    frac: torch.Tensor,
    lattice: torch.Tensor,
    mask: torch.Tensor,
    config: DescriptorConfig = DescriptorConfig(),
) -> torch.Tensor:
    """Permutation/translation/rotation-invariant differentiable phi(C)."""
    dtype = frac.dtype
    device = frac.device
    batch, n_max = types.shape
    one_hot = torch.nn.functional.one_hot(
        (types - 1).clamp_min(0), num_classes=config.vocab_size
    ).to(dtype)
    composition = (one_hot * mask.unsqueeze(-1)).sum(dim=1)
    composition = composition / mask.sum(dim=1, keepdim=True).clamp_min(1)

    delta = frac[:, :, None, :] - frac[:, None, :, :]
    delta = delta - torch.floor(delta + 0.5)
    cart = torch.einsum("bijd,bdk->bijk", delta, lattice)
    distances = torch.linalg.vector_norm(cart, dim=-1)
    centers = torch.linspace(
        config.rdf_min_A, config.rdf_max_A, config.rdf_bins, device=device, dtype=dtype
    )
    lookup = pair_channel_lookup(config.vocab_size, device)
    rdf = torch.zeros(
        batch, config.pair_channels, config.rdf_bins, device=device, dtype=dtype
    )
    upper = torch.triu(
        torch.ones(n_max, n_max, dtype=torch.bool, device=device), diagonal=1
    )
    for row in range(batch):
        valid = upper & mask[row, :, None] & mask[row, None, :]
        pair_i, pair_j = torch.where(valid)
        if pair_i.numel() == 0:
            continue
        pair_types_i = types[row, pair_i] - 1
        pair_types_j = types[row, pair_j] - 1
        channels = lookup[pair_types_i, pair_types_j]
        radial = torch.exp(
            -0.5
            * (
                (distances[row, pair_i, pair_j, None] - centers[None, :])
                / config.rdf_sigma_A
            ).square()
        )
        rdf[row].index_add_(0, channels, radial)
        rdf[row] /= float(pair_i.numel())

    gram = lattice @ lattice.transpose(-1, -2)
    lengths = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
    volume = torch.linalg.det(lattice).abs().clamp_min(1e-12)
    volume_scale = volume.pow(1.0 / 3.0)
    normalized_lengths = lengths / volume_scale[:, None]
    cos_alpha = gram[:, 1, 2] / (lengths[:, 1] * lengths[:, 2]).clamp_min(1e-12)
    cos_beta = gram[:, 0, 2] / (lengths[:, 0] * lengths[:, 2]).clamp_min(1e-12)
    cos_gamma = gram[:, 0, 1] / (lengths[:, 0] * lengths[:, 1]).clamp_min(1e-12)
    volume_per_atom = volume / mask.sum(dim=1).clamp_min(1).to(dtype)
    lattice_features = torch.cat(
        (
            normalized_lengths,
            torch.stack((cos_alpha, cos_beta, cos_gamma), dim=-1),
            torch.log1p(volume_per_atom)[:, None],
        ),
        dim=-1,
    )
    return torch.cat((composition, rdf.flatten(1), lattice_features), dim=-1)


class NoveltyIndex:
    def __init__(self, payload: dict[str, Any], device: torch.device | str = "cpu") -> None:
        self.config = DescriptorConfig(**payload["config"])
        self.descriptors = payload["descriptors"].to(device)
        self.mean = payload["mean"].to(device)
        self.std = payload["std"].to(device)
        self.ids = payload["ids"]
        self.formulas = payload["formulas"]

    @classmethod
    def load(cls, path: str, device: torch.device | str = "cpu") -> "NoveltyIndex":
        return cls(torch.load(path, map_location="cpu", weights_only=False), device)

    def standardize(self, descriptor: torch.Tensor) -> torch.Tensor:
        return (descriptor - self.mean) / self.std

    def score(
        self, descriptor: torch.Tensor, reference_chunk: int = 2048
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.standardize(descriptor)
        best_distance = torch.full(
            (query.shape[0],), float("inf"), device=query.device, dtype=query.dtype
        )
        best_index = torch.full(
            (query.shape[0],), -1, device=query.device, dtype=torch.long
        )
        dimension_scale = query.shape[1] ** 0.5
        for start in range(0, self.descriptors.shape[0], reference_chunk):
            reference = self.descriptors[start : start + reference_chunk]
            # sqrt(mean standardized squared feature difference)
            distance = torch.cdist(query, reference) / dimension_scale
            values, indices = distance.min(dim=1)
            improve = values < best_distance
            best_distance = torch.where(improve, values, best_distance)
            best_index = torch.where(improve, indices + start, best_index)
        return best_distance, best_index

    def smooth_score(
        self,
        descriptor: torch.Tensor,
        temperature: float = 0.10,
        top_k: int = 32,
        reference_chunk: int = 2048,
    ) -> torch.Tensor:
        """Differentiable top-k soft minimum reference distance."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        query = self.standardize(descriptor)
        dimension_scale = query.shape[1] ** 0.5
        nearest = torch.full(
            (query.shape[0], min(top_k, self.descriptors.shape[0])),
            float("inf"), device=query.device, dtype=query.dtype,
        )
        for start in range(0, self.descriptors.shape[0], reference_chunk):
            distance = torch.cdist(
                query, self.descriptors[start : start + reference_chunk]
            ) / dimension_scale
            nearest = torch.topk(
                torch.cat((nearest, distance), dim=1),
                k=nearest.shape[1], largest=False, dim=1,
            ).values
        return -temperature * torch.logsumexp(-nearest / temperature, dim=1)


def build_reference_index(
    dataset_path: str,
    output_path: str,
    device: torch.device,
    atom_budget: int = 3000,
) -> dict[str, Any]:
    packed = torch.load(dataset_path, map_location="cpu", weights_only=False)
    dataset = PackedCrystalDataset(dataset_path, "train")
    # The objective reference set is all original structures, not only train.
    dataset.indices = list(range(len(packed["ids"])))
    sizes = torch.diff(packed["offsets"])
    order = torch.argsort(sizes).tolist()
    descriptors: list[torch.Tensor] = []
    ordered_indices: list[int] = []
    current: list[int] = []
    current_max = 0

    def process(indices: list[int]) -> None:
        items = [dataset[index] for index in indices]
        batch = collate_crystals(items)
        with torch.no_grad():
            descriptor = crystal_descriptor(
                batch["types"].to(device),
                batch["frac_coords"].to(device),
                batch["lattice"].to(device),
                batch["mask"].to(device),
            )
        descriptors.append(descriptor.cpu())
        ordered_indices.extend(indices)

    for index in order:
        size = int(sizes[index])
        proposal_max = max(current_max, size)
        if current and proposal_max * (len(current) + 1) > atom_budget:
            process(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, size)
    if current:
        process(current)
    stacked_ordered = torch.cat(descriptors)
    inverse = torch.empty(len(ordered_indices), dtype=torch.long)
    inverse[torch.tensor(ordered_indices)] = torch.arange(len(ordered_indices))
    stacked = stacked_ordered[inverse]
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    active = std > 1e-6
    # Constant dimensions carry no metric information and stay exactly zero.
    std = torch.where(active, std, torch.full_like(std, float("inf")))
    standardized = ((stacked - mean) / std).float()
    payload = {
        "version": "nagen-novelty-v1",
        "config": asdict(DescriptorConfig()),
        "descriptors": standardized,
        "mean": mean.float(),
        "std": std.float(),
        "ids": packed["ids"],
        "formulas": packed["formulas"],
        "dataset_source_hash": packed["source_first_1MiB_sha256"],
        "active_dimensions": int(active.sum()),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    manifest = {
        key: payload[key]
        for key in ("version", "config", "dataset_source_hash", "active_dimensions")
    }
    manifest["n_reference"] = len(payload["ids"])
    manifest["dimension"] = DescriptorConfig().dimension
    with open(output.with_suffix(".manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--atom-budget", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    build_reference_index(args.dataset, args.out, torch.device(args.device), args.atom_budget)


if __name__ == "__main__":
    main()

"""Shared tensor preparation and serialization for shooting campaigns."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .model import CrystalState


def state_payload(state: CrystalState) -> dict[str, Any]:
    return {
        "atom": state.atom.detach().cpu().tolist(),
        "frac": state.frac.detach().cpu().tolist(),
        "lattice": state.lattice.detach().cpu().tolist(),
    }


def fixed_composition_source(
    model: torch.nn.Module,
    composition: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[CrystalState, torch.Tensor]:
    indices = torch.tensor(
        composition["type_indices"], dtype=torch.long, device=device
    )[None]
    mask = torch.ones_like(indices, dtype=torch.bool)
    atom = F.one_hot(indices - 1, num_classes=model.config.vocab_size).to(
        dtype=next(model.parameters()).dtype
    ) * 2.0
    generator = torch.Generator(device=device).manual_seed(seed)
    return CrystalState(
        atom,
        torch.rand(1, indices.shape[1], 3, device=device, generator=generator),
        torch.randn(1, 6, device=device, generator=generator),
    ), mask

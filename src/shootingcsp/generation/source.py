"""Source construction and pre-generation safety checks for ShootingFlow."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from shootingcsp.config.schema import SourcePrecheckPolicy
from shootingcsp.inverse.model import CrystalState


def make_source(model, type_indices, seed: int, source_mode: str):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    count = len(type_indices)
    mask = torch.ones(1, count, dtype=torch.bool, device=device)
    fixed_atom = F.one_hot(
        type_indices[None] - 1,
        num_classes=model.config.vocab_size,
    ).to(device=device, dtype=next(model.parameters()).dtype) * 2.0
    lattice = torch.randn(1, 6, device=device, generator=generator)
    if source_mode == "uniform":
        frac = torch.rand(1, count, 3, device=device, generator=generator)
    elif source_mode == "polyhedral":
        from shootingcsp.inverse.polyhedral_prior import polyhedral_source_frac

        source_matrix = model._lattice_matrix(lattice, torch.tensor([count], device=device))
        frac = polyhedral_source_frac(
            type_indices[None], mask, source_matrix, generator=generator
        )
    else:
        raise ValueError(f"unsupported source mode: {source_mode}")
    return CrystalState(fixed_atom, frac, lattice), mask


def catastrophic_reason(crystal, policy: SourcePrecheckPolicy | None = None) -> str | None:
    from shootingcsp.selection.pipeline import neighbors

    if policy is not None and not policy.enabled:
        return None
    volume_min = policy.volume_per_atom_A3.min if policy is not None else 2.0
    volume_max = policy.volume_per_atom_A3.max if policy is not None else 60.0
    max_condition = policy.max_cell_condition_number if policy is not None else 40.0
    minimum_distance = policy.minimum_distance_A if policy is not None else 0.4
    volume_per_atom = abs(np.linalg.det(crystal.lattice)) / len(crystal.elements)
    if not np.isfinite(crystal.frac).all() or not np.isfinite(crystal.lattice).all():
        return "nonfinite"
    if not volume_min <= volume_per_atom <= volume_max or np.linalg.cond(crystal.lattice) > max_condition:
        return "unsafe_cell"
    if any(
        distance < minimum_distance
        for site in range(len(crystal.elements))
        for _, _, distance in neighbors(crystal, site, minimum_distance)
    ):
        return f"overlap_below_{minimum_distance:g}A"
    return None


__all__ = ["catastrophic_reason", "make_source"]

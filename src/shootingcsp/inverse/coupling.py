"""Source/target couplings for periodic crystal flow matching."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def assignment_source_frac(
    target_frac: torch.Tensor,
    target_types: torch.Tensor,
    mask: torch.Tensor,
    lattice: torch.Tensor,
) -> torch.Tensor:
    """Draw uniform sites and optimally relabel them within each species.

    Relabeling iid uniform sites preserves the unordered source distribution.
    The assignment uses minimum-image Cartesian squared distance and never
    changes species, composition, target coordinates, or the source law.
    """
    if target_frac.ndim != 3 or target_frac.shape[-1] != 3:
        raise ValueError("target_frac must have shape (B,N,3)")
    if target_types.shape != target_frac.shape[:2] or mask.shape != target_types.shape:
        raise ValueError("incompatible type/mask shapes")
    if lattice.shape != (len(target_frac), 3, 3):
        raise ValueError("lattice must have shape (B,3,3)")
    source = torch.rand_like(target_frac)
    matched = torch.zeros_like(source)
    for batch in range(len(target_frac)):
        active_types = target_types[batch, mask[batch]].unique()
        for species in active_types.tolist():
            indices = torch.where(mask[batch] & (target_types[batch] == species))[0]
            delta = target_frac[batch, indices, None] - source[batch, None, indices]
            delta = delta - torch.floor(delta + 0.5)
            cart = torch.einsum("ijd,dk->ijk", delta, lattice[batch])
            cost = cart.square().sum(-1).detach().cpu().numpy()
            rows, columns = linear_sum_assignment(cost)
            row_tensor = torch.as_tensor(rows, device=indices.device)
            column_tensor = torch.as_tensor(columns, device=indices.device)
            matched[batch, indices[row_tensor]] = source[batch, indices[column_tensor]]
    return matched * mask.unsqueeze(-1)

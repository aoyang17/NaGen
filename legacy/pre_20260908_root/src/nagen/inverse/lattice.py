"""Positive-volume lattice parameterization for crystal flow matching."""

from __future__ import annotations

import torch


def encode_lattice(lattice: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
    """Encode row-vector lattices as six unconstrained Cholesky variables.

    The Gram matrix removes global Cartesian rotation. Dividing by N^(2/3)
    keeps the representation approximately size independent.
    """
    lattice = lattice.double()
    n_scale = n_atoms.double().clamp_min(1.0).pow(2.0 / 3.0)
    gram = lattice @ lattice.transpose(-1, -2)
    gram = gram / n_scale[..., None, None]
    eye = torch.eye(3, dtype=gram.dtype, device=gram.device)
    chol = torch.linalg.cholesky(gram + 1e-8 * eye)
    encoded = torch.stack(
        (
            chol[..., 0, 0].log(),
            chol[..., 1, 1].log(),
            chol[..., 2, 2].log(),
            chol[..., 1, 0],
            chol[..., 2, 0],
            chol[..., 2, 1],
        ),
        dim=-1,
    )
    return encoded.float()


def decode_lattice(encoded: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
    """Decode six unconstrained variables to det(L)>0 canonical lattices."""
    output_shape = (*encoded.shape[:-1], 3, 3)
    chol = torch.zeros(output_shape, dtype=encoded.dtype, device=encoded.device)
    # Clamping protects ODE transients without changing the positive determinant invariant.
    chol[..., 0, 0] = encoded[..., 0].clamp(-4.0, 4.0).exp()
    chol[..., 1, 1] = encoded[..., 1].clamp(-4.0, 4.0).exp()
    chol[..., 2, 2] = encoded[..., 2].clamp(-4.0, 4.0).exp()
    chol[..., 1, 0] = encoded[..., 3]
    chol[..., 2, 0] = encoded[..., 4]
    chol[..., 2, 1] = encoded[..., 5]
    scale = n_atoms.to(encoded).clamp_min(1.0).pow(1.0 / 3.0)
    return chol * scale[..., None, None]


def standardize_lattice(
    encoded: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (encoded - mean) / std.clamp_min(1e-6)


def unstandardize_lattice(
    encoded: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return encoded * std.clamp_min(1e-6) + mean

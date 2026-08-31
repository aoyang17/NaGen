"""Differentiable ODE sampling utilities for crystal D-Flow."""

from __future__ import annotations

import torch

from .lattice import decode_lattice, unstandardize_lattice
from .model import CrystalState, CrystalVectorField


def _add(state: CrystalState, velocity: CrystalState, factor: float) -> CrystalState:
    return CrystalState(
        state.atom + factor * velocity.atom,
        torch.remainder(state.frac + factor * velocity.frac, 1.0),
        state.lattice + factor * velocity.lattice,
    )


def integrate_flow(
    model: CrystalVectorField,
    source: CrystalState,
    mask: torch.Tensor,
    steps: int = 24,
    method: str = "midpoint",
    freeze_atom: bool = False,
) -> CrystalState:
    state = source
    dt = 1.0 / steps
    for step in range(steps):
        t0 = torch.full(
            (mask.shape[0], 1), step * dt, dtype=source.atom.dtype, device=source.atom.device
        )
        if method == "euler":
            velocity = model(*state, t0, mask)
        elif method == "midpoint":
            first = model(*state, t0, mask)
            middle = _add(state, first, 0.5 * dt)
            velocity = model(*middle, t0 + 0.5 * dt, mask)
        else:
            raise ValueError(f"unsupported integrator: {method}")
        state = _add(state, velocity, dt)
        if freeze_atom:
            state = CrystalState(source.atom, state.frac, state.lattice)
    return state


def random_source(
    batch_size: int,
    n_atoms: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[CrystalState, torch.Tensor]:
    mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device)
    source = CrystalState(
        torch.randn(batch_size, n_atoms, vocab_size, dtype=dtype, device=device),
        torch.rand(batch_size, n_atoms, 3, dtype=dtype, device=device),
        torch.randn(batch_size, 6, dtype=dtype, device=device),
    )
    return source, mask


def decode_terminal(
    model: CrystalVectorField, state: CrystalState, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    types = state.atom.argmax(dim=-1) + 1
    types = types.masked_fill(~mask, 0)
    physical_lattice = unstandardize_lattice(
        state.lattice, model.lattice_mean, model.lattice_std
    )
    lattice = decode_lattice(physical_lattice, mask.sum(dim=1))
    return types, torch.remainder(state.frac, 1.0), lattice

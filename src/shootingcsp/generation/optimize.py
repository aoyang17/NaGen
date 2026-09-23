"""Source-space ShootingFlow optimization over a swappable surrogate."""

from __future__ import annotations

from collections.abc import Callable

import torch

from shootingcsp.inverse.guidance import _source_prior, soft_constraint_terms
from shootingcsp.inverse.model import CrystalState
from shootingcsp.inverse.sample import decode_terminal, integrate_flow
from shootingcsp.inverse.spec import (
    DEFAULT_FE_COORDINATION_OPTIONS,
    normalize_fe_coordination_options,
)


EnergyFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def optimize_source_with_surrogate(
    model,
    source: CrystalState,
    mask: torch.Tensor,
    energy_fn: EnergyFunction,
    core_probabilities: torch.Tensor,
    steps: int,
    lr: float,
    ode_steps: int,
    fe_coordination_options=DEFAULT_FE_COORDINATION_OPTIONS,
):
    """Optimize Flow source X/L through the complete terminal ODE.

    The surrogate only needs to expose a differentiable scalar
    ``energy_fn(frac, lattice)``. Model parameters are frozen; gradients flow
    from the terminal energy and geometric penalties back to the source.
    """
    if steps < 0 or ode_steps < 1 or lr <= 0:
        raise ValueError("steps must be nonnegative, lr positive, and ode_steps positive")
    allowed_fe = normalize_fe_coordination_options(fe_coordination_options)
    frac = source.frac.detach().clone().requires_grad_(True)
    lattice = source.lattice.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam((frac, lattice), lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state = CrystalState(source.atom, frac, lattice)
        terminal = integrate_flow(
            model,
            state,
            mask,
            steps=ode_steps,
            method="midpoint",
            freeze_atom=True,
        )
        _, final_frac, final_lattice = decode_terminal(model, terminal, mask)
        energy = energy_fn(final_frac, final_lattice).mean()
        soft = soft_constraint_terms(
            model,
            terminal,
            mask,
            probabilities_override=core_probabilities,
            fe_coordination_options=allowed_fe,
        )
        prior = _source_prior(state, mask)
        loss = (
            0.10 * energy
            + 20.0 * soft["distance"]
            + 3.0 * soft["p_coordination"]
            + 3.0 * soft["fe_coordination"]
            + 2.0 * soft["volume"]
            + 0.02 * prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_((frac, lattice), 5.0)
        optimizer.step()
        with torch.no_grad():
            frac.remainder_(1.0)
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "surrogate_energy_eV_atom": float(energy.detach()),
                **{f"soft_{name}": float(value.detach()) for name, value in soft.items()},
                "source_prior": float(prior.detach()),
                "fe_coordination_options": list(allowed_fe),
            }
        )
    final_source = CrystalState(source.atom, frac.detach(), lattice.detach())
    with torch.no_grad():
        terminal = integrate_flow(
            model,
            final_source,
            mask,
            steps=ode_steps,
            method="midpoint",
            freeze_atom=True,
        )
        _, final_frac, final_lattice = decode_terminal(model, terminal, mask)
    return final_source, final_frac, final_lattice, history


optimize_source_with_uma = optimize_source_with_surrogate


__all__ = ["optimize_source_with_surrogate", "optimize_source_with_uma"]

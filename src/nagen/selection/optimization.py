"""Energy-only DFlow source optimization with a frozen generator callback.

The generator owns fixed N/A/family conditioning. No chemistry, novelty,
composition, or force penalties are added to the objective. Evaluate every
saved endpoint through final hard gates; optimization does not imply validity.
"""
import torch


def optimize_source(source, generate, energy_per_atom, steps=20, lr=.01):
    """generate(*source) -> (frac[1,N,3], lattice[1,3,3]).

    energy_per_atom can be a CHGNet or UMA force/stress VJP adapter. Source
    tensors must include only continuous coordinate/lattice variables. The
    returned snapshots include the baseline and each updated endpoint for
    matched-budget comparisons and feasibility retention by the caller.
    """
    if steps < 0 or lr <= 0:
        raise ValueError("invalid optimization budget")
    variables = [x.detach().clone().requires_grad_(True) for x in source]
    optimizer = torch.optim.Adam(variables, lr=lr)
    snapshots = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        frac, cell = generate(*variables)
        energy = energy_per_atom(frac, cell).reshape(())
        if not torch.isfinite(energy):
            raise ValueError("nonfinite potential energy")
        snapshots.append({"step": step, "energy_eV_atom": float(energy.detach()),
                          "frac": frac.detach().clone(), "lattice": cell.detach().clone()})
        if step == steps:
            break
        gradients = torch.autograd.grad(energy, variables)
        if any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("nonfinite source gradient")
        for variable, gradient in zip(variables, gradients):
            variable.grad = gradient
        optimizer.step()
    return snapshots

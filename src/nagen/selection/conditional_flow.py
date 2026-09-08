"""Fixed-N/A geometry flow matching, without physical-property penalties.

Species-wise assignment only permutes uniform source points. The unordered
point-cloud source law is unchanged; ordered coordinates are NOT claimed iid
after target-dependent matching. Inference uses independent uniform points.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


ATOM_SCALE = 2.0


def assigned_uniform_source(target, types, mask):
    source = torch.rand_like(target)
    matched = source.clone()
    for b in range(len(target)):
        for species in types[b, mask[b]].unique().tolist():
            indices = torch.where(mask[b] & (types[b] == species))[0]
            delta = target[b, indices, None] - source[b, None, indices]
            delta = delta - torch.floor(delta + .5)
            cost = delta.square().sum(-1).detach().cpu().numpy()
            rows, columns = linear_sum_assignment(cost)
            matched[b, indices[torch.as_tensor(rows, device=indices.device)]] = source[
                b, indices[torch.as_tensor(columns, device=indices.device)]]
    return matched * mask[..., None]


def conditional_flow_loss(model, types, frac, lattice, mask):
    atom = ATOM_SCALE * F.one_hot((types-1).clamp_min(0), model.config.vocab_size).to(frac)
    atom = atom * mask[..., None]
    source_frac = assigned_uniform_source(frac, types, mask)
    source_lattice = torch.randn_like(lattice)
    delta = frac-source_frac
    delta -= torch.floor(delta+.5)
    t = torch.rand(len(frac), 1, device=frac.device, dtype=frac.dtype)
    x = (source_frac + t[..., None]*delta) % 1
    h = source_lattice+t*(lattice-source_lattice)
    prediction = model(atom, x, h, t, mask)
    frac_mse = ((prediction.frac-delta).square()*mask[..., None]).sum()/(3*mask.sum())
    lattice_mse = (prediction.lattice-(lattice-source_lattice)).square().mean()
    # Ordinary regression of normalized geometry velocities, not a weighted
    # composition/energy/force/motif/novelty objective. A is never optimized.
    return frac_mse+lattice_mse, {'frac_mse': frac_mse.detach(), 'lattice_mse': lattice_mse.detach()}

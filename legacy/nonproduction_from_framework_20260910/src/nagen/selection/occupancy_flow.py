"""Rectified flow for fixed-backbone, fixed-count Na occupancy generation.

The physical design specification is conditioning, not a penalty.  The model
maps a Gaussian source to continuous site logits; exact Na count is imposed by
top-k decoding.  Physics and novelty are deliberately handled downstream.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OccupancyFlowConfig:
    sites: int
    hidden: int = 128
    depth: int = 4


class OccupancyFlow(nn.Module):
    def __init__(self, config: OccupancyFlowConfig):
        super().__init__()
        self.config = config
        layers = []
        width = config.sites + 2  # state, time, conditioned Na fraction
        for _ in range(config.depth):
            layers += [nn.Linear(width, config.hidden), nn.SiLU()]
            width = config.hidden
        layers.append(nn.Linear(width, config.sites))
        self.net = nn.Sequential(*layers)

    def forward(self, state, time, target_count):
        if not torch.is_tensor(target_count):
            target_count = torch.as_tensor(target_count, device=state.device, dtype=state.dtype)
        fraction = target_count.to(state).reshape(-1, 1) / self.config.sites
        if len(fraction) == 1 and len(state) != 1:
            fraction = fraction.expand(len(state), 1)
        return self.net(torch.cat((state, time.reshape(-1, 1), fraction), dim=-1))

    def metadata(self):
        return asdict(self.config)


def flow_matching_loss(model, occupancy, target_count):
    """Standard conditional rectified-flow regression; no physics penalties."""
    target = occupancy.to(dtype=torch.float32) * 2 - 1
    source = torch.randn_like(target)
    time = torch.rand(len(target), 1, device=target.device)
    state = (1-time)*source + time*target
    velocity = target-source
    return (model(state, time, target_count)-velocity).square().mean()


@torch.no_grad()
def sample_logits(model, count, target_count, steps=64, generator=None, device=None):
    device = device or next(model.parameters()).device
    state = torch.randn(count, model.config.sites, generator=generator, device=device)
    dt = 1.0/steps
    conditioned = torch.full((count,), float(target_count), device=device)
    for index in range(steps):
        time = torch.full((count, 1), (index+.5)*dt, device=device)
        state = state + dt*model(state, time, conditioned)
    return state


def topk_occupancy(logits, target_count):
    if not 0 <= target_count <= logits.shape[-1]:
        raise ValueError("target_count outside number of sites")
    result = torch.zeros_like(logits, dtype=torch.bool)
    if target_count:
        result.scatter_(1, logits.topk(target_count, dim=1).indices, True)
    return result

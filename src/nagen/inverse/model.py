"""Unconditional periodic flow matching model for complete crystals."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .lattice import decode_lattice, unstandardize_lattice
from .spec import DEFAULT_SPEC


class CrystalState(NamedTuple):
    atom: torch.Tensor  # (B, N, V), continuous element state
    frac: torch.Tensor  # (B, N, 3), periodic fractional coordinates
    lattice: torch.Tensor  # (B, 6), standardized lattice state


@dataclass(frozen=True)
class FlowConfig:
    vocab_size: int = 4
    hidden_dim: int = 192
    layers: int = 5
    heads: int = 8
    radial_basis: int = 24
    radial_max_A: float = 10.0
    dropout: float = 0.0
    time_dim: int = 64


def time_embedding(t: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        torch.linspace(0.0, -math.log(10000.0), half, device=t.device, dtype=t.dtype)
    )
    phase = 2.0 * math.pi * t * frequencies[None, :]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class PeriodicAttentionBlock(nn.Module):
    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.heads = config.heads
        self.head_dim = hidden // config.heads
        if hidden % config.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden)
        self.pair_bias = nn.Sequential(
            nn.Linear(config.radial_basis, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.heads),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.SiLU(),
            nn.Linear(4 * hidden, hidden),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, hidden: torch.Tensor, radial: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, atoms, width = hidden.shape
        normalized = self.norm1(hidden)
        qkv = self.qkv(normalized).reshape(batch, atoms, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        scores = torch.einsum("bihd,bjhd->bhij", query, key) / math.sqrt(self.head_dim)
        scores = scores + self.pair_bias(radial).permute(0, 3, 1, 2)
        valid_pairs = mask[:, None, :, None] & mask[:, None, None, :]
        scores = scores.masked_fill(~valid_pairs, -torch.finfo(scores.dtype).max)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(valid_pairs, attention, torch.zeros_like(attention))
        update = torch.einsum("bhij,bjhd->bihd", attention, value).reshape(batch, atoms, width)
        hidden = hidden + self.dropout(self.out(update))
        hidden = hidden + self.dropout(self.ff(self.norm2(hidden)))
        return hidden * mask.unsqueeze(-1)


class CrystalVectorField(nn.Module):
    def __init__(
        self,
        config: FlowConfig,
        lattice_mean: torch.Tensor,
        lattice_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("lattice_mean", lattice_mean.float().reshape(6))
        self.register_buffer("lattice_std", lattice_std.float().reshape(6))
        coordinate_features = 12  # sin/cos at two periodic frequencies
        global_features = config.time_dim + 6 + 2
        self.atom_input = nn.Linear(config.vocab_size + coordinate_features, config.hidden_dim)
        self.global_input = nn.Sequential(
            nn.Linear(global_features, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.blocks = nn.ModuleList(PeriodicAttentionBlock(config) for _ in range(config.layers))
        self.atom_norm = nn.LayerNorm(config.hidden_dim)
        self.atom_velocity = nn.Linear(config.hidden_dim, config.vocab_size)
        self.frac_velocity = nn.Linear(config.hidden_dim, 3)
        self.lattice_velocity = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 6),
        )
        centers = torch.linspace(0.0, config.radial_max_A, config.radial_basis)
        self.register_buffer("radial_centers", centers)
        spacing = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
        self.radial_gamma = 1.0 / (spacing * spacing)

    def _lattice_matrix(self, state: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
        physical = unstandardize_lattice(state, self.lattice_mean, self.lattice_std)
        return decode_lattice(physical, n_atoms)

    def _radial_features(
        self, frac: torch.Tensor, lattice: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        delta = frac[:, :, None, :] - frac[:, None, :, :]
        delta = delta - torch.floor(delta + 0.5)
        cart = torch.einsum("bijd,bdk->bijk", delta, lattice)
        distance = torch.linalg.vector_norm(cart, dim=-1)
        radial = torch.exp(
            -self.radial_gamma
            * (distance.unsqueeze(-1) - self.radial_centers.to(distance)) ** 2
        )
        pair_mask = mask[:, :, None] & mask[:, None, :]
        return radial * pair_mask.unsqueeze(-1)

    def forward(
        self,
        atom: torch.Tensor,
        frac: torch.Tensor,
        lattice_state: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
    ) -> CrystalState:
        batch, atoms, _ = atom.shape
        if t.ndim == 1:
            t = t[:, None]
        n_atoms = mask.sum(dim=1)
        lattice = self._lattice_matrix(lattice_state, n_atoms)
        angles = 2.0 * math.pi * frac
        coord_features = torch.cat(
            (angles.sin(), angles.cos(), (2.0 * angles).sin(), (2.0 * angles).cos()),
            dim=-1,
        )
        global_features = torch.cat(
            (
                time_embedding(t, self.config.time_dim),
                lattice_state,
                torch.log1p(n_atoms.to(atom))[:, None] / 6.0,
                (n_atoms.to(atom) / 184.0)[:, None],
            ),
            dim=-1,
        )
        global_hidden = self.global_input(global_features)
        hidden = self.atom_input(torch.cat((atom, coord_features), dim=-1))
        hidden = (hidden + global_hidden[:, None, :]) * mask.unsqueeze(-1)
        radial = self._radial_features(frac, lattice, mask)
        for block in self.blocks:
            hidden = block(hidden, radial, mask)
        hidden = self.atom_norm(hidden) * mask.unsqueeze(-1)
        atom_velocity = self.atom_velocity(hidden) * mask.unsqueeze(-1)
        frac_velocity = self.frac_velocity(hidden).tanh() * mask.unsqueeze(-1)
        pooled = hidden.sum(dim=1) / n_atoms.clamp_min(1).to(hidden)[:, None]
        lattice_velocity = self.lattice_velocity(torch.cat((pooled, global_hidden), dim=-1))
        return CrystalState(atom_velocity, frac_velocity, lattice_velocity)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "lattice_mean": self.lattice_mean.detach().cpu(),
            "lattice_std": self.lattice_std.detach().cpu(),
            "model": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, object]) -> "CrystalVectorField":
        model = cls(
            FlowConfig(**payload["config"]),
            payload["lattice_mean"],
            payload["lattice_std"],
        )
        model.load_state_dict(payload["model"])
        return model


def flow_matching_loss(
    model: nn.Module,
    target_types: torch.Tensor,
    target_frac: torch.Tensor,
    target_lattice: torch.Tensor,
    mask: torch.Tensor,
    type_scale: float = 2.0,
    coordinate_noise_sigma: float = 0.50,
    geometry_only: bool = False,
    geometry_validity_weight: float = 0.0,
    coupling: str = "noise",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint rectified flow loss with a minimum-image torus path for X.

    ``coupling`` selects how the X source is paired with the target:

    ``"noise"``
        The historical behaviour, ``wrap(target + coordinate_noise_sigma * eps)``.
    ``"assignment"``
        Draw the source exactly uniform -- the same law `sample.random_source`
        draws at inference -- then permute it within each element species to
        minimise total squared periodic displacement.  Permuting i.i.d. uniform
        points leaves the marginal exactly uniform, so this buys a short,
        low-variance regression target at no cost to the source distribution.
        See `nagen.inverse.coupling` for why the ``"noise"`` coupling carries
        almost no information at sigma = 0.5.
    """
    batch, atoms = target_types.shape
    device = target_frac.device
    dtype = target_frac.dtype
    base_model = model.module if hasattr(model, "module") else model
    target_atom = F.one_hot(
        (target_types - 1).clamp_min(0), num_classes=base_model.config.vocab_size
    ).to(dtype) * type_scale
    source_atom = target_atom if geometry_only else torch.randn_like(target_atom)
    if coupling == "assignment":
        from .coupling import assignment_source_frac

        source_frac = assignment_source_frac(
            target_frac,
            target_types,
            mask,
            base_model._lattice_matrix(target_lattice, mask.sum(dim=1)),
        )
    elif coupling == "noise":
        # Wrapped N(0, 0.5^2) has an almost-uniform torus marginal.  It was also
        # meant to retain a site correspondence, but at sigma = 0.5 the pair
        # offset law is uniform to within 1e-4, so that correspondence is not
        # actually present -- see nagen.inverse.coupling.
        source_frac = torch.remainder(
            target_frac + coordinate_noise_sigma * torch.randn_like(target_frac), 1.0
        )
    else:
        raise ValueError(f"unsupported coupling: {coupling!r}")
    source_lattice = torch.randn_like(target_lattice)
    t = torch.rand(batch, 1, device=device, dtype=dtype).clamp_(1e-4, 1.0 - 1e-4)
    atom_t = (
        target_atom
        if geometry_only
        else (1.0 - t[:, None, :]) * source_atom + t[:, None, :] * target_atom
    )
    frac_delta = target_frac - source_frac
    frac_delta = frac_delta - torch.floor(frac_delta + 0.5)
    frac_t = torch.remainder(source_frac + t[:, None, :] * frac_delta, 1.0)
    lattice_t = (1.0 - t) * source_lattice + t * target_lattice
    prediction = model(atom_t, frac_t, lattice_t, t, mask)
    target_atom_velocity = torch.zeros_like(target_atom) if geometry_only else target_atom - source_atom
    atom_error = (prediction.atom - target_atom_velocity).square().sum(dim=-1)
    frac_error = (prediction.frac - frac_delta).square().sum(dim=-1)
    atom_loss = (atom_error * mask).sum() / mask.sum().clamp_min(1)
    frac_loss = (frac_error * mask).sum() / mask.sum().clamp_min(1)
    lattice_loss = (prediction.lattice - (target_lattice - source_lattice)).square().mean()
    endpoint_atom = atom_t + (1.0 - t[:, None, :]) * prediction.atom
    endpoint_type_loss = (
        torch.zeros((), device=device, dtype=dtype)
        if geometry_only
        else F.cross_entropy(endpoint_atom[mask], (target_types[mask] - 1).long())
    )

    # Endpoint distance reconstruction directly supervises geometry while
    # remaining invariant to global Cartesian rotation and periodic translation.
    endpoint_frac = torch.remainder(
        frac_t + (1.0 - t[:, None, :]) * prediction.frac, 1.0
    )
    endpoint_lattice_state = lattice_t + (1.0 - t) * prediction.lattice
    target_matrix = base_model._lattice_matrix(target_lattice, mask.sum(dim=1))
    endpoint_matrix = base_model._lattice_matrix(endpoint_lattice_state, mask.sum(dim=1))

    def pair_distances(coords: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        delta = coords[:, :, None, :] - coords[:, None, :, :]
        delta = delta - torch.floor(delta + 0.5)
        cart = torch.einsum("bijd,bdk->bijk", delta, matrix)
        return torch.linalg.vector_norm(cart, dim=-1)

    target_distances = pair_distances(target_frac, target_matrix)
    endpoint_distances = pair_distances(endpoint_frac, endpoint_matrix)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    pair_mask &= torch.triu(
        torch.ones(atoms, atoms, dtype=torch.bool, device=device), diagonal=1
    )
    pair_distance_loss = F.smooth_l1_loss(
        endpoint_distances[pair_mask] / 4.0,
        target_distances[pair_mask] / 4.0,
    )
    geometry_validity_loss = endpoint_distances.sum() * 0.0
    if geometry_validity_weight > 0.0:
        elements = DEFAULT_SPEC.elements
        minima = torch.empty(
            len(elements), len(elements), dtype=dtype, device=device
        )
        for first, element_first in enumerate(elements):
            for second, element_second in enumerate(elements):
                key = "-".join(sorted((element_first, element_second)))
                reverse_key = "-".join((element_first, element_second))
                direct_key = "-".join((element_second, element_first))
                minimum = DEFAULT_SPEC.pair_minima.get(
                    reverse_key,
                    DEFAULT_SPEC.pair_minima.get(direct_key),
                )
                if minimum is None:
                    minimum = DEFAULT_SPEC.generic_min_radius_fraction * (
                        DEFAULT_SPEC.covalent_radii[element_first]
                        + DEFAULT_SPEC.covalent_radii[element_second]
                    )
                minima[first, second] = minimum
        type_offsets = (target_types - 1).clamp_min(0)
        pair_minimum = minima[
            type_offsets[:, :, None], type_offsets[:, None, :]
        ]
        collision = F.relu(pair_minimum + 0.035 - endpoint_distances).square()
        collision_values = collision[pair_mask]
        collision_worst = max(1, math.ceil(0.10 * collision_values.numel()))
        collision_loss = (
            collision_values.mean()
            + collision_values.topk(collision_worst).values.mean()
        )

        oxygen = (target_types == DEFAULT_SPEC.element_to_index["O"]) & mask
        phosphorus = (target_types == DEFAULT_SPEC.element_to_index["P"]) & mask
        iron = (target_types == DEFAULT_SPEC.element_to_index["Fe"]) & mask
        oxygen_distances = endpoint_distances.masked_fill(
            ~oxygen[:, None, :], 1.0e3
        )
        oxygen_distances = oxygen_distances.masked_fill(
            torch.eye(atoms, dtype=torch.bool, device=device)[None], 1.0e3
        )
        sorted_oxygen = oxygen_distances.sort(dim=-1).values

        def shell_site_loss(count: int, lower: float, cutoff: float) -> torch.Tensor:
            included = sorted_oxygen[..., :count]
            excluded = sorted_oxygen[..., count]
            return (
                F.relu(lower + 0.035 - included).square()
                + F.relu(included - (cutoff - 0.035)).square()
            ).sum(dim=-1) + F.relu(cutoff + 0.035 - excluded).square()

        def worst_sites(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
            active = values[centers]
            if active.numel() == 0:
                return values.sum() * 0.0
            count = max(1, math.ceil(0.25 * active.numel()))
            return active.mean() + active.topk(count).values.mean()

        p_loss = worst_sites(
            shell_site_loss(
                4, DEFAULT_SPEC.pair_minima["P-O"],
                DEFAULT_SPEC.p_o_bond_cutoff,
            ),
            phosphorus,
        )
        fe_site_loss = torch.stack(
            [
                shell_site_loss(
                    count, DEFAULT_SPEC.pair_minima["Fe-O"],
                    DEFAULT_SPEC.fe_o_bond_cutoff,
                )
                for count in (4, 5, 6)
            ],
            dim=0,
        ).amin(dim=0)
        fe_loss = worst_sites(fe_site_loss, iron)
        geometry_validity_loss = collision_loss + p_loss + fe_loss
    total = (
        (0.1 if geometry_only else 1.0) * atom_loss
        + 4.0 * frac_loss
        + lattice_loss
        + (0.0 if geometry_only else 0.5) * endpoint_type_loss
        + 0.25 * pair_distance_loss
        + geometry_validity_weight * geometry_validity_loss
    )
    return total, {
        "loss": total.detach(),
        "atom_loss": atom_loss.detach(),
        "frac_loss": frac_loss.detach(),
        "lattice_loss": lattice_loss.detach(),
        "endpoint_type_loss": endpoint_type_loss.detach(),
        "pair_distance_loss": pair_distance_loss.detach(),
        "geometry_validity_loss": geometry_validity_loss.detach(),
    }

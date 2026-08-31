"""Independent MLFF energy plus executable hard-constraint barrier forces."""

from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet

from .constraints import minimum_distance
from .guidance import _coordination_assignment
from .spec import DEFAULT_SPEC


class ConstraintBiasedCHGNetCalculator(Calculator):
    """CHGNet calculator augmented with differentiable feasibility barriers.

    The cell is held fixed.  The barrier uses the same 27-image PBC distance
    convention and the same pair minima/bond cutoffs as the exact verifier.
    Its energy is reported separately and is never called a hull energy.
    """

    implemented_properties = ("energy", "free_energy", "forces")

    def __init__(
        self,
        model: CHGNet,
        use_device: str = "cuda",
        distance_k_eV_A2: float = 120.0,
        coordination_k_eV_A2: float = 80.0,
        margin_A: float = 0.04,
        fixed_topology: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base = CHGNetCalculator(model=model, use_device=use_device)
        self.device = torch.device(use_device)
        self.distance_k = distance_k_eV_A2
        self.coordination_k = coordination_k_eV_A2
        self.margin = margin_A
        self.fixed_topology = fixed_topology
        self.coordination_assignment: torch.Tensor | None = None
        self.assignment_symbols: tuple[str, ...] | None = None

    def reset_topology(self, atoms) -> None:
        """Freeze a capacity-aware coordination graph from the input A, X, L."""
        dtype = torch.float32
        positions = torch.tensor(
            atoms.get_positions(), dtype=dtype, device=self.device
        )
        cell = torch.tensor(atoms.cell.array, dtype=dtype, device=self.device)
        frac = positions @ torch.linalg.inv(cell)
        delta = frac[:, None, :] - frac[None, :, :]
        delta = delta - torch.floor(delta + 0.5)
        translations = torch.tensor(
            [
                (i, j, k)
                for i in (-1.0, 0.0, 1.0)
                for j in (-1.0, 0.0, 1.0)
                for k in (-1.0, 0.0, 1.0)
            ],
            dtype=dtype,
            device=self.device,
        )
        distances = torch.linalg.vector_norm(
            (delta[:, :, None, :] + translations) @ cell, dim=-1
        ).amin(dim=-1)
        symbols = tuple(atoms.get_chemical_symbols())
        offsets = {element: index for index, element in enumerate(DEFAULT_SPEC.elements)}
        type_index = torch.tensor(
            [[offsets[symbol] for symbol in symbols]],
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.ones(
            1, len(symbols), dtype=torch.bool, device=self.device
        )
        self.coordination_assignment = _coordination_assignment(
            type_index, mask, distances[None]
        )[0]
        self.assignment_symbols = symbols

    def _barrier(self, atoms) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.float32
        positions = torch.tensor(
            atoms.get_positions(), dtype=dtype, device=self.device, requires_grad=True
        )
        cell = torch.tensor(atoms.cell.array, dtype=dtype, device=self.device)
        frac = positions @ torch.linalg.inv(cell)
        delta = frac[:, None, :] - frac[None, :, :]
        delta = delta - torch.floor(delta + 0.5)
        translations = torch.tensor(
            [
                (i, j, k)
                for i in (-1.0, 0.0, 1.0)
                for j in (-1.0, 0.0, 1.0)
                for k in (-1.0, 0.0, 1.0)
            ],
            dtype=dtype,
            device=self.device,
        )
        images = delta[:, :, None, :] + translations
        distances = torch.linalg.vector_norm(images @ cell, dim=-1).amin(dim=-1)

        symbols = atoms.get_chemical_symbols()
        assignment = None
        if self.fixed_topology:
            if (
                self.coordination_assignment is None
                or self.assignment_symbols != tuple(symbols)
            ):
                self.reset_topology(atoms)
            assignment = self.coordination_assignment
            assert assignment is not None
        minima = torch.tensor(
            [[minimum_distance(a, b) for b in symbols] for a in symbols],
            dtype=dtype,
            device=self.device,
        )
        upper = torch.triu(
            torch.ones(len(symbols), len(symbols), dtype=torch.bool, device=self.device),
            diagonal=1,
        )
        distance_violation = torch.relu(minima + self.margin - distances)
        distance_loss = distance_violation.square().masked_fill(~upper, 0.0).sum()

        oxygen = torch.tensor(
            [symbol == "O" for symbol in symbols],
            dtype=torch.bool,
            device=self.device,
        )
        coordination_loss = torch.zeros((), dtype=dtype, device=self.device)
        phosphorus = torch.tensor(
            [symbol == "P" for symbol in symbols],
            dtype=torch.bool,
            device=self.device,
        )
        iron = torch.tensor(
            [symbol == "Fe" for symbol in symbols],
            dtype=torch.bool,
            device=self.device,
        )
        oxygen_distances = distances.masked_fill(~oxygen[None, :], 1.0e3)
        oxygen_distances.fill_diagonal_(1.0e3)
        for index, symbol in enumerate(symbols):
            if symbol == "P":
                lower = DEFAULT_SPEC.pair_minima["P-O"]
                upper_cutoff = DEFAULT_SPEC.p_o_bond_cutoff
                bond_target = 1.55
                default_count = 4
            elif symbol == "Fe":
                lower = DEFAULT_SPEC.pair_minima["Fe-O"]
                upper_cutoff = DEFAULT_SPEC.fe_o_bond_cutoff
                bond_target = 2.00
                default_count = 6
            else:
                continue
            if assignment is None:
                neighbor_indices = oxygen_distances[index].argsort()[:default_count]
            else:
                neighbor_indices = torch.nonzero(
                    assignment[index], as_tuple=False
                ).flatten()
            count = int(neighbor_indices.numel())
            if count == 0:
                continue
            included = distances[index, neighbor_indices]
            coordination_loss = coordination_loss + torch.relu(
                lower + self.margin - included
            ).square().sum()
            coordination_loss = coordination_loss + torch.relu(
                included - (upper_cutoff - self.margin)
            ).square().sum()
            unassigned_oxygen = oxygen.clone()
            unassigned_oxygen[neighbor_indices] = False
            if bool(unassigned_oxygen.any()):
                excluded = distances[index, unassigned_oxygen].amin()
                coordination_loss = coordination_loss + torch.relu(
                    upper_cutoff + self.margin - excluded
                ).square()
            neighbor_pair = distances[neighbor_indices[:, None], neighbor_indices[None, :]]
            pair_upper = torch.triu(
                torch.ones(count, count, dtype=torch.bool, device=self.device),
                diagonal=1,
            )
            pair_values = neighbor_pair[pair_upper].sort().values
            coordination_loss = coordination_loss + 0.25 * (
                included - bond_target
            ).square().sum()
            if count == 4:
                pair_target = torch.full_like(
                    pair_values, bond_target * (8.0 / 3.0) ** 0.5
                )
            elif count == 5:
                pair_target = torch.tensor(
                    [bond_target * 2.0**0.5] * 6
                    + [bond_target * 3.0**0.5] * 3
                    + [2.0 * bond_target],
                    dtype=dtype,
                    device=self.device,
                )
            elif count == 6:
                pair_target = torch.tensor(
                    [bond_target * 2.0**0.5] * 12
                    + [2.0 * bond_target] * 3,
                    dtype=dtype,
                    device=self.device,
                )
            else:
                pair_target = pair_values.detach()
            if pair_values.numel():
                coordination_loss = coordination_loss + (
                    pair_values - pair_target
                ).square().mean()
        barrier = self.distance_k * distance_loss + self.coordination_k * coordination_loss
        gradient = torch.autograd.grad(barrier, positions)[0]
        return barrier.detach(), gradient.detach()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        self.base.calculate(atoms, ["energy", "forces"], system_changes)
        barrier, gradient = self._barrier(atoms)
        physical_energy = float(self.base.results["energy"])
        physical_forces = np.asarray(self.base.results["forces"], dtype=np.float64)
        self.results = {
            "energy": physical_energy + float(barrier),
            "free_energy": physical_energy + float(barrier),
            "forces": physical_forces - gradient.cpu().numpy(),
            "physical_energy": physical_energy,
            "constraint_barrier_energy": float(barrier),
            "physical_forces": physical_forces,
        }

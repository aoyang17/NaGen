"""Differentiable MP2020 hull reference for soft Na/Fe/P/O compositions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DifferentiableMP2020Hull(nn.Module):
    """Exact piecewise-linear MP hull with an optional smooth maximum."""

    elements = ("Na", "Fe", "P", "O")

    def __init__(self, planes: torch.Tensor, temperature_eV_atom: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("planes", planes)
        # Candidate correction is exactly linear for the locked OMat24
        # metadata over Na/Fe/P/O: Fe=-2.256 and O=-0.687 eV/atom fraction.
        self.register_buffer(
            "candidate_correction_coefficients",
            torch.tensor((0.0, -2.256, 0.0, -0.687), dtype=planes.dtype),
        )
        self.temperature_eV_atom = float(temperature_eV_atom)

    @classmethod
    def from_phase_cache(
        cls, path: str | Path, temperature_eV_atom: float = 0.0
    ) -> "DifferentiableMP2020Hull":
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        payload = json.loads(Path(path).read_text())
        if payload.get("compatibility") != "MaterialsProject2020Compatibility":
            raise ValueError("phase cache is not MP2020-compatible")
        diagram = PhaseDiagram([
            ComputedEntry(
                Composition(row["composition"]), float(row["energy_eV"]),
                entry_id=row["entry_id"],
            )
            for row in payload["entries"]
        ])
        if tuple(str(element) for element in diagram.elements) != cls.elements:
            raise ValueError("phase diagram element order is not Na,Fe,P,O")
        planes = []
        for facet in diagram.facets:
            points = np.asarray(diagram.qhull_data[facet], dtype=np.float64)
            matrix = np.column_stack((points[:, :-1], np.ones(len(points))))
            planes.append(np.linalg.solve(matrix, points[:, -1]))
        return cls(torch.tensor(np.stack(planes), dtype=torch.float32), temperature_eV_atom)

    def composition_fractions(
        self, probabilities: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        counts = (probabilities * mask.unsqueeze(-1)).sum(dim=1)
        return counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    def reference_energy(self, fractions: torch.Tensor) -> torch.Tensor:
        coordinates = torch.cat((
            fractions[:, 1:], torch.ones_like(fractions[:, :1])
        ), dim=-1)
        values = coordinates @ self.planes.T
        if self.temperature_eV_atom > 0:
            temperature = self.temperature_eV_atom
            return temperature * torch.logsumexp(values / temperature, dim=-1)
        return values.max(dim=-1).values

    def candidate_correction(self, fractions: torch.Tensor) -> torch.Tensor:
        return fractions @ self.candidate_correction_coefficients

    def raw_e_hull(
        self, final_energy_eV_atom: torch.Tensor,
        probabilities: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        fractions = self.composition_fractions(probabilities, mask)
        return (
            final_energy_eV_atom + self.candidate_correction(fractions)
            - self.reference_energy(fractions)
        )

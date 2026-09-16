"""Differentiable all-UMA same-energy-scale hull reference for soft compositions.

`UMAHullCache` holds the official caliber but exposes it only as a
`{reduced_formula -> float}` lookup, so `d E_ref / d z_A` is identically zero and
the composition variables receive no gradient.  `DifferentiableMP2020Hull` has
the right math but is built on the cross-scale MP2020 caliber the 8/27 protocol
forbids.  This module is the intersection: the exact piecewise-linear hull of
the *same* all-UMA phase diagram the cache was built from.

The lower convex hull is a convex piecewise-linear function, so it equals the
pointwise maximum of its facet-supporting hyperplanes.  Both caliber and support
come from the cache itself, so no separate provenance can drift.

No compatibility correction appears here: candidate and support energies are on
one energy scale by construction, which is the whole point of the caliber.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .uma_hull import UMAHullCache


class DifferentiableUMAHull(nn.Module):
    """Exact piecewise-linear all-UMA hull with an optional smooth maximum."""

    elements = ("Na", "Fe", "P", "O")

    def __init__(
        self,
        planes: torch.Tensor,
        temperature_eV_atom: float = 0.0,
        provenance: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("planes", planes)
        self.temperature_eV_atom = float(temperature_eV_atom)
        self.provenance = dict(provenance or {})

    @classmethod
    def from_hull_cache(
        cls,
        path: str | Path,
        temperature_eV_atom: float = 0.0,
        expected_model_sha256: str | None = None,
        expected_task_name: str | None = None,
    ) -> "DifferentiableUMAHull":
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        cache = UMAHullCache(
            str(path),
            expected_model_sha256=expected_model_sha256,
            expected_task_name=expected_task_name,
        )
        support = cache.data["support_entries"]
        if len(support) != int(cache.data["relaxed_support_entry_count"]):
            raise ValueError("hull cache support_entries is not the relaxed support")
        entries = []
        for row in support:
            composition = Composition(row["composition"])
            entries.append(ComputedEntry(
                composition,
                float(row["final_energy_eV_atom"]) * float(composition.num_atoms),
                entry_id=str(row["entry_id"]),
            ))
        diagram = PhaseDiagram(entries)
        if tuple(str(element) for element in diagram.elements) != cls.elements:
            raise ValueError("phase diagram element order is not Na,Fe,P,O")
        planes = []
        for facet in diagram.facets:
            points = np.asarray(diagram.qhull_data[facet], dtype=np.float64)
            matrix = np.column_stack((points[:, :-1], np.ones(len(points))))
            planes.append(np.linalg.solve(matrix, points[:, -1]))
        provenance = {
            "hull_cache": str(Path(path).resolve()),
            "hull_cache_content_sha256": cache.data.get("content_sha256"),
            "uma_model_sha256": cache.data.get("uma_model_sha256"),
            "task_name": cache.data.get("task_name"),
            "caliber": "all_UMA_same_energy_scale",
            "n_support_entries": len(entries),
            "n_facets": len(planes),
        }
        return cls(
            torch.tensor(np.stack(planes), dtype=torch.float64),
            temperature_eV_atom,
            provenance,
        )

    def composition_fractions(
        self, probabilities: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        counts = (probabilities * mask.unsqueeze(-1)).sum(dim=1)
        return counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    def reference_energy(self, fractions: torch.Tensor) -> torch.Tensor:
        planes = self.planes.to(fractions.dtype)
        coordinates = torch.cat((
            fractions[:, 1:], torch.ones_like(fractions[:, :1])
        ), dim=-1)
        values = coordinates @ planes.T
        if self.temperature_eV_atom > 0:
            temperature = self.temperature_eV_atom
            return temperature * torch.logsumexp(values / temperature, dim=-1)
        return values.max(dim=-1).values

    def e_hull(
        self, final_energy_eV_atom: torch.Tensor, fractions: torch.Tensor
    ) -> torch.Tensor:
        """E_hull on the official caliber: both terms are UMA energies."""
        return final_energy_eV_atom - self.reference_energy(fractions)

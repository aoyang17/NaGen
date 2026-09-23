"""Interfaces for swappable differentiable and evaluation surrogate models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import torch


EnergyFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EnergyFactory = Callable[[torch.Tensor], EnergyFunction]


@runtime_checkable
class EvaluationSurrogate(Protocol):
    """Structure evaluator used for relaxation and final independent labels."""

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    @property
    def calculator(self): ...

    def evaluate(self, elements, frac, cell) -> dict[str, Any]: ...

    def evaluate_atoms(self, atoms) -> dict[str, Any]: ...

    def evaluate_batch(self, structures) -> list[dict[str, Any]]: ...


@runtime_checkable
class SurrogateAdapter(Protocol):
    """A surrogate backend exposed to generation-time optimization.

    Implementations may provide only ``guidance_energy``. ``evaluator`` is
    required only when the backend is also used for FIRE and final validation.
    """

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def guidance_energy(self, atomic_numbers: torch.Tensor) -> EnergyFunction: ...

    def evaluator(self, inference_settings: str = "default") -> EvaluationSurrogate: ...


@dataclass(frozen=True)
class SurrogateSpec:
    """Serializable surrogate selection passed to a registered factory."""

    backend: str
    checkpoint: str | None = None
    device: str = "cuda"
    task_name: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    factory: str | None = None


__all__ = [
    "EnergyFactory",
    "EnergyFunction",
    "EvaluationSurrogate",
    "SurrogateAdapter",
    "SurrogateSpec",
]

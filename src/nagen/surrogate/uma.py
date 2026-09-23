"""UMA adapter implementing the swappable surrogate interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from nagen.inverse.uma_guidance import load_uma_calculator_vjp_factory
from .base import EnergyFunction, SurrogateSpec
from .registry import register_surrogate


@dataclass
class UMASurrogate:
    spec: SurrogateSpec
    _factory: Any = None
    _evaluator: Any = None
    _provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.spec.checkpoint:
            raise ValueError("UMA surrogate requires checkpoint")
        unknown = set(self.spec.options) - {"inference_settings"}
        if unknown:
            raise ValueError(f"unsupported UMA surrogate options: {sorted(unknown)}")
        self._provenance = {
            "backend": "uma",
            "checkpoint": self.spec.checkpoint,
            "device": self.spec.device,
            "task_name": self.spec.task_name or "omat",
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance or {})

    def evaluator(self, inference_settings: str = "default"):
        from nagen.selection.uma import UMAEvaluator

        settings = (
            inference_settings
            if inference_settings != "default"
            else self.spec.options.get("inference_settings", "default")
        )
        if self._evaluator is None or self._evaluator.provenance.get("inference_settings") != settings:
            self._evaluator = UMAEvaluator(
                str(self.spec.checkpoint),
                self.spec.device,
                self.spec.task_name or "omat",
                inference_settings=settings,
            )
            self._provenance = {**self._provenance, "evaluation": self._evaluator.provenance}
        return self._evaluator

    def guidance_energy(self, atomic_numbers: torch.Tensor) -> EnergyFunction:
        if self._factory is None:
            calculator = getattr(self._evaluator, "calculator", None)
            self._factory, metadata = load_uma_calculator_vjp_factory(
                str(self.spec.checkpoint),
                self.spec.device,
                self.spec.task_name or "omat",
                calculator=calculator,
            )
            self._provenance = {**self._provenance, "guidance": metadata}
        return self._factory(atomic_numbers)


def load_uma_surrogate(spec: SurrogateSpec) -> UMASurrogate:
    return UMASurrogate(spec)


register_surrogate("uma", load_uma_surrogate)


__all__ = ["UMASurrogate", "load_uma_surrogate"]

"""Swappable surrogate interfaces and built-in backends."""

from .base import (
    EnergyFactory,
    EnergyFunction,
    EvaluationSurrogate,
    SurrogateAdapter,
    SurrogateSpec,
)
from .registry import available_surrogates, load_surrogate, register_surrogate
from . import uma as _uma  # register the built-in backend

__all__ = [
    "EnergyFactory",
    "EnergyFunction",
    "EvaluationSurrogate",
    "SurrogateAdapter",
    "SurrogateSpec",
    "available_surrogates",
    "load_surrogate",
    "register_surrogate",
]

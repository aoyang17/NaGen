"""Registry for independently packaged surrogate backends."""

from __future__ import annotations

from importlib import import_module
from typing import Callable

from .base import SurrogateAdapter, SurrogateSpec


SurrogateLoader = Callable[[SurrogateSpec], SurrogateAdapter]
_SURROGATE_REGISTRY: dict[str, SurrogateLoader] = {}


def register_surrogate(name: str, loader: SurrogateLoader, *, replace: bool = False) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("surrogate backend name cannot be empty")
    if normalized in _SURROGATE_REGISTRY and not replace:
        raise ValueError(f"surrogate backend already registered: {normalized}")
    _SURROGATE_REGISTRY[normalized] = loader


def available_surrogates() -> tuple[str, ...]:
    return tuple(sorted(_SURROGATE_REGISTRY))


def _external_loader(reference: str) -> SurrogateLoader:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("surrogate factory must use 'module:callable' syntax")
    loader = getattr(import_module(module_name), attribute)
    if not callable(loader):
        raise TypeError(f"surrogate factory is not callable: {reference}")
    return loader


def load_surrogate(spec: SurrogateSpec) -> SurrogateAdapter:
    loader = _external_loader(spec.factory) if spec.factory else _SURROGATE_REGISTRY.get(spec.backend.lower())
    if loader is None:
        raise ValueError(
            f"unknown surrogate backend {spec.backend!r}; "
            f"available: {available_surrogates()}"
        )
    adapter = loader(spec)
    if not isinstance(adapter, SurrogateAdapter):
        raise TypeError(
            "surrogate loader must return an object with provenance, "
            "guidance_energy(atomic_numbers), and evaluator(inference_settings)"
        )
    return adapter


__all__ = ["available_surrogates", "load_surrogate", "register_surrogate"]

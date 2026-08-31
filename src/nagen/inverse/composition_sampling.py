"""Exact rejection collection for generated ``N,A`` proposals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from .constraints import composition_metrics
from .spec import DEFAULT_SPEC


CompositionProposal = Callable[[int], Iterable[list[str]]]


@dataclass(frozen=True)
class CompositionPool:
    unique_sequences: tuple[tuple[str, ...], ...]
    proposals: int
    accepted_proposals: int
    status: str


def count_signature(elements: Iterable[str]) -> tuple[int, int, int, int]:
    counts = Counter(elements)
    return tuple(counts[element] for element in DEFAULT_SPEC.elements)


def exact_composition_passes(elements: list[str]) -> bool:
    metrics = composition_metrics(elements)
    return bool(
        DEFAULT_SPEC.n_primitive_min <= len(elements) <= DEFAULT_SPEC.n_primitive_max
        and metrics["domain_ok"] and metrics["charge_ok"] and metrics["capacity_ok"]
    )


def collect_unique_compositions(
    proposal_fn: CompositionProposal, maximum_proposals: int = 200_000,
    maximum_unique: int = 32, batch_size: int = 512,
) -> CompositionPool:
    unique: dict[tuple[int, int, int, int], tuple[str, ...]] = {}
    proposals = 0
    accepted = 0
    while proposals < maximum_proposals and len(unique) < maximum_unique:
        request = min(batch_size, maximum_proposals - proposals)
        batch = list(proposal_fn(request))
        if len(batch) > request:
            raise ValueError("proposal_fn returned more proposals than requested")
        if not batch:
            break
        proposals += len(batch)
        for elements in batch:
            if exact_composition_passes(elements):
                accepted += 1
                signature = count_signature(elements)
                unique.setdefault(signature, tuple(elements))
                if len(unique) >= maximum_unique:
                    break
    status = "complete" if len(unique) >= 6 else "insufficient_unique_compositions"
    return CompositionPool(tuple(unique.values()), proposals, accepted, status)

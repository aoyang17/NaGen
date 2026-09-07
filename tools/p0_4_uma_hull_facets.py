"""P0.4 - verify the continuous all-UMA hull reproduces the discrete cache.

The official caliber currently reaches the solver only as a
`{reduced_formula -> float}` lookup (`UMAHullCache.get`), which is
piecewise-*constant* in composition, so `d E_ref / d z_A` is identically zero and
the composition variables are frozen by construction.  `DifferentiableUMAHull`
rebuilds the same phase diagram and keeps it as facet planes, which is
piecewise-*linear* and therefore differentiable almost everywhere.

This script is the acceptance test for that swap.  It is CPU-only, needs no
model and no GPU, and answers three questions:

  1. Exactness   - does max_f(a_f . x + b_f) reproduce the cached
                   `reference_energy_eV_atom` for every pooled composition?
                   Anything above ~1e-6 eV/atom means the plane extraction does
                   not match `PhaseDiagram.get_hull_energy` and the swap is
                   unsafe.
  2. Gradient    - is d E_ref / d fractions actually non-zero?  This is the
                   defect being repaired, so it is asserted rather than assumed.
  3. Coverage    - the table raises `MissingUMAHullReferenceError` off-pool.
                   The plane form is defined on the whole simplex; this counts
                   how much of the search space the table was silently refusing.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from nagen.inverse.differentiable_uma_hull import DifferentiableUMAHull
from nagen.inverse.uma_hull import MissingUMAHullReferenceError, UMAHullCache


ELEMENTS = DifferentiableUMAHull.elements
EXACTNESS_TOLERANCE_eV_ATOM = 1.0e-6


def find_cache(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    patterns = (
        "experiments/**/uma_hull_cache.json",
        "data/**/uma_hull_cache.json",
        "/mnt/data2/aobo/NaGen/**/uma_hull_cache.json",
    )
    for pattern in patterns:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return Path(hits[-1])
    raise FileNotFoundError("could not locate uma_hull_cache.json; pass --cache")


def fractions_from_counts(counts: dict[str, float]) -> np.ndarray:
    vector = np.array([float(counts.get(element, 0.0)) for element in ELEMENTS])
    return vector / vector.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-random", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/p0_4_uma_hull_facets.json")
    args = parser.parse_args()

    path = find_cache(args.cache)
    table = UMAHullCache(str(path))
    hull = DifferentiableUMAHull.from_hull_cache(path, args.temperature)

    pooled = table.data["compositions"]
    formulas = sorted(pooled)
    fractions = torch.tensor(
        np.stack([fractions_from_counts(pooled[name]["counts"]) for name in formulas]),
        dtype=torch.float64,
    )
    cached = torch.tensor(
        [float(pooled[name]["reference_energy_eV_atom"]) for name in formulas],
        dtype=torch.float64,
    )
    with torch.no_grad():
        continuous = hull.reference_energy(fractions)
    error = (continuous - cached).abs()
    worst = int(error.argmax())

    probe = fractions.clone().requires_grad_(True)
    hull.reference_energy(probe).sum().backward()
    gradient = probe.grad.detach()
    gradient_norm = gradient.norm(dim=-1)

    rng = np.random.default_rng(args.seed)
    random_fractions = rng.dirichlet(np.ones(len(ELEMENTS)), size=args.n_random)
    with torch.no_grad():
        random_reference = hull.reference_energy(
            torch.tensor(random_fractions, dtype=torch.float64)
        )
    table_misses = 0
    for row in random_fractions:
        counts = {element: float(value) for element, value in zip(ELEMENTS, row)}
        try:
            table.get(counts)
        except (MissingUMAHullReferenceError, Exception):
            table_misses += 1

    result = {
        "hull_cache": str(path),
        "provenance": hull.provenance,
        "cache_definition_field": table.data.get("definition"),
        "exactness": {
            "n_pooled_compositions": len(formulas),
            "max_abs_error_eV_atom": float(error.max()),
            "mean_abs_error_eV_atom": float(error.mean()),
            "worst_formula": formulas[worst],
            "worst_cached_eV_atom": float(cached[worst]),
            "worst_continuous_eV_atom": float(continuous[worst]),
            "tolerance_eV_atom": EXACTNESS_TOLERANCE_eV_ATOM,
            "passed": bool(error.max() < EXACTNESS_TOLERANCE_eV_ATOM),
        },
        "gradient": {
            "table_gradient_is_identically_zero": True,
            "median_grad_norm_eV_atom_per_fraction": float(gradient_norm.median()),
            "min_grad_norm": float(gradient_norm.min()),
            "max_grad_norm": float(gradient_norm.max()),
            "frac_with_zero_gradient": float((gradient_norm == 0).double().mean()),
            "passed": bool(gradient_norm.min() > 0),
        },
        "coverage": {
            "n_random_simplex_points": args.n_random,
            "table_missing": table_misses,
            "table_miss_rate": table_misses / max(1, args.n_random),
            "continuous_finite_rate": float(
                torch.isfinite(random_reference).double().mean()
            ),
        },
    }
    result["verdict"] = {
        "safe_to_swap": bool(
            result["exactness"]["passed"] and result["gradient"]["passed"]
        ),
        "note": (
            "safe_to_swap means the continuous hull reproduces every cached "
            "reference to within tolerance while carrying a non-zero composition "
            "gradient, so it is a strict superset of the lookup table."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

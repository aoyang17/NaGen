"""Exact Na--Fe--P--O composition and geometry checks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from .spec import DEFAULT_SPEC, OptimizationSpec


def _pair_key(a: str, b: str) -> str:
    order = {element: index for index, element in enumerate(DEFAULT_SPEC.elements)}
    return "-".join(sorted((a, b), key=order.__getitem__))


def pbc_distance_matrix(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    frac = np.asarray(frac, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    delta = frac[:, None, :] - frac[None, :, :]
    delta -= np.floor(delta + 0.5)
    best_squared = np.full(delta.shape[:2], np.inf, dtype=np.float64)
    for i in (-1.0, 0.0, 1.0):
        for j in (-1.0, 0.0, 1.0):
            for k in (-1.0, 0.0, 1.0):
                cart = (delta + np.asarray((i, j, k))) @ lattice
                best_squared = np.minimum(
                    best_squared, np.einsum("ijk,ijk->ij", cart, cart)
                )
    return np.sqrt(np.maximum(best_squared, 0.0))


def minimum_distance(a: str, b: str, spec: OptimizationSpec = DEFAULT_SPEC) -> float:
    key = _pair_key(a, b)
    reverse = "-".join(reversed(key.split("-")))
    if key in spec.pair_minima:
        return spec.pair_minima[key]
    if reverse in spec.pair_minima:
        return spec.pair_minima[reverse]
    return spec.generic_min_radius_fraction * (
        spec.covalent_radii[a] + spec.covalent_radii[b]
    )


def bond_cutoff(a: str, b: str, spec: OptimizationSpec = DEFAULT_SPEC) -> float:
    pair = frozenset((a, b))
    if pair == {"P", "O"}:
        return spec.p_o_bond_cutoff
    if pair == {"Fe", "O"}:
        return spec.fe_o_bond_cutoff
    return minimum_distance(a, b, spec)


def composition_metrics(
    elements: Iterable[str], spec: OptimizationSpec = DEFAULT_SPEC
) -> dict[str, Any]:
    """Compute the normalized quantities in OPTIMIZATION Appendix A1--A4."""
    elements = list(elements)
    counts = Counter(elements)
    support_ok = set(counts) == set(spec.elements)
    p = counts["P"]
    if p <= 0:
        return {
            "support_ok": False, "m": None, "n": None, "y": None,
            "fe_valence": None, "kappa_fe": None, "k": None,
            "capacity_mAh_g": 0.0, "molar_mass_formula_g_mol": None,
            "domain_ok": False, "charge_ok": False, "capacity_ok": False,
        }
    m = counts["Na"] / p
    n = counts["Fe"] / p
    y = 4.0 - counts["O"] / p
    fe_valence = (3.0 - 2.0 * y - m) / n if n > 0 else float("nan")
    kappa_fe = m + spec.fe_valence_max * n + 2.0 * y - 3.0
    k = min(m, kappa_fe)
    formula_mass = (
        m * spec.atomic_masses["Na"] + n * spec.atomic_masses["Fe"]
        + spec.atomic_masses["P"] + (4.0 - y) * spec.atomic_masses["O"]
    )
    capacity = max(0.0, k) * spec.faraday_C_mol / (3.6 * formula_mass)
    domain_ok = bool(
        support_ok and spec.y_min <= y <= spec.y_max and m >= 1.0
        and m >= n and n <= 1.0
    )
    charge_ok = bool(
        np.isfinite(fe_valence)
        and spec.fe_valence_min <= fe_valence <= spec.fe_valence_max
        and kappa_fe >= 0.0
    )
    return {
        "support_ok": support_ok, "m": m, "n": n, "y": y,
        "fe_valence": fe_valence, "kappa_fe": kappa_fe, "k": k,
        "capacity_mAh_g": float(capacity),
        "molar_mass_formula_g_mol": float(formula_mass),
        "domain_ok": domain_ok, "charge_ok": charge_ok,
        "capacity_ok": capacity >= spec.capacity_min_mAh_g,
    }


def charge_neutrality(
    elements: Iterable[str], spec: OptimizationSpec = DEFAULT_SPEC
) -> tuple[bool, dict[str, Any]]:
    metrics = composition_metrics(elements, spec)
    return bool(metrics["support_ok"] and metrics["charge_ok"]), metrics


def theoretical_capacity(
    elements: Iterable[str], spec: OptimizationSpec = DEFAULT_SPEC
) -> tuple[float, float, float]:
    elements = list(elements)
    metrics = composition_metrics(elements, spec)
    cell_mass = sum(spec.atomic_masses.get(element, 0.0) for element in elements)
    extracted = max(0.0, float(metrics.get("k") or 0.0)) * elements.count("P")
    return float(metrics["capacity_mAh_g"]), float(cell_mass), extracted


def _coordination_checks(
    elements: list[str], distances: np.ndarray, spec: OptimizationSpec
) -> tuple[dict[str, bool], dict[str, Any]]:
    p_ok = True
    fe_ok = True
    sites: list[dict[str, Any]] = []
    for i, element in enumerate(elements):
        if element not in {"P", "Fe"}:
            continue
        cutoff = spec.p_o_bond_cutoff if element == "P" else spec.fe_o_bond_cutoff
        neighbours = [
            j for j, other in enumerate(elements)
            if j != i and other == "O" and distances[i, j] <= cutoff + 1e-10
        ]
        coordination = len(neighbours)
        if element == "P":
            p_ok &= coordination == 4
        else:
            fe_ok &= coordination in spec.fe_coordination_options
        sites.append({"site": i, "element": element, "coordination": coordination})
    return {
        "p_coordination_eq_4": bool(p_ok),
        "fe_coordination_allowed": bool(fe_ok),
    }, {
        "sites": sites,
        "allowed_fe_coordination": list(spec.fe_coordination_options),
    }


def evaluate_feasibility(
    elements: Iterable[str], frac_coords: np.ndarray, lattice: np.ndarray,
    delta_e_hull_eV_atom: float | None = None,
    spec: OptimizationSpec = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Evaluate all final hard constraints; an unknown hull value fails."""
    elements = list(elements)
    frac = np.asarray(frac_coords, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    if frac.shape != (len(elements), 3):
        raise ValueError("frac_coords must have shape (len(elements), 3)")
    composition = composition_metrics(elements, spec)
    finite = bool(np.isfinite(frac).all() and np.isfinite(lattice).all())
    determinant = float(np.linalg.det(lattice)) if finite else float("nan")
    positive_lattice = bool(np.isfinite(determinant) and determinant > 0.0)
    volume_per_atom = determinant / len(elements) if elements else float("nan")
    known = set(elements).issubset(spec.elements)
    distances = (
        pbc_distance_matrix(frac, lattice)
        if elements and finite and known else np.zeros((len(elements), len(elements)))
    )
    violations: list[dict[str, Any]] = []
    minimum_observed = float("inf")
    if known:
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                observed = float(distances[i, j])
                threshold = minimum_distance(elements[i], elements[j], spec)
                minimum_observed = min(minimum_observed, observed)
                if observed + 1e-10 < threshold:
                    violations.append({
                        "i": i, "j": j, "pair": f"{elements[i]}-{elements[j]}",
                        "distance_A": observed, "minimum_A": threshold,
                    })
    coordination, coordination_details = _coordination_checks(elements, distances, spec)
    hull_known = bool(
        delta_e_hull_eV_atom is not None and np.isfinite(delta_e_hull_eV_atom)
    )
    checks = {
        "element_support_exact": bool(composition["support_ok"]),
        "atom_count_range": spec.n_primitive_min <= len(elements) <= spec.n_primitive_max,
        "finite_structure": finite,
        "fractional_coordinate_domain": bool(finite and ((frac >= 0.0) & (frac < 1.0)).all()),
        "positive_lattice_determinant": positive_lattice,
        "volume_per_atom_range": bool(
            positive_lattice and spec.volume_per_atom_min_A3 <= volume_per_atom
            <= spec.volume_per_atom_max_A3
        ),
        "composition_domain": bool(composition["domain_ok"]),
        "fe_average_valence": bool(composition["charge_ok"]),
        "specific_capacity": bool(composition["capacity_ok"]),
        "minimum_pbc_distances": not violations,
        **coordination,
        "hull_threshold": bool(
            hull_known and delta_e_hull_eV_atom <= spec.hull_max_eV_atom
        ),
    }
    return {
        "feasible": bool(all(checks.values())), "checks": checks,
        "values": {
            "N": len(elements), "det_lattice_A3": determinant,
            "volume_per_atom_A3": volume_per_atom,
            "minimum_distance_A": None if minimum_observed == float("inf") else minimum_observed,
            "delta_e_hull_eV_atom": delta_e_hull_eV_atom,
            "hull_evaluated": hull_known, **composition,
        },
        "details": {"distance_violations": violations, "coordination": coordination_details},
    }

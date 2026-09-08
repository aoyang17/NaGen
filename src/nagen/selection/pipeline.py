"""Conditioning -> hard gates -> lexicographic minimax -> farthest selection.

Energy values carry their evaluator provenance, never implied hull energies. Geometry
uses reciprocal-cell bounds so periodic images and skew cells are included.
The external PolyAnionCathodeFilter is an explicit adapter, not impersonated.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from math import gcd
from functools import reduce

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import rankdata


@dataclass
class Crystal:
    id: str
    elements: list[str]
    frac: np.ndarray
    lattice: np.ndarray
    family: str = "phosphate"

    @property
    def composition(self):
        counts = Counter(self.elements)
        divisor = reduce(gcd, counts.values())
        return tuple((e, n // divisor) for e, n in sorted(counts.items()))


@dataclass(frozen=True)
class Conditioning:
    counts: dict[str, int]
    family: str = "phosphate"

    def __post_init__(self):
        if (not {"Fe", "P", "O"}.issubset(self.counts)
            or not set(self.counts).issubset({"Na", "Fe", "P", "O"})
            or any(type(n) is not int or n < (0 if e == "Na" else 1)
                   for e, n in self.counts.items())):
            raise ValueError("conditioning requires integer counts; only Na may be zero")
        if self.family != "phosphate":
            raise ValueError("only phosphate family implemented")

    def matches(self, crystal):
        return Counter(crystal.elements) == self.counts and crystal.family == self.family


def neighbors(crystal, index, radius):
    """All periodic neighbors, including nonzero self images, within radius."""
    inv = np.linalg.inv(crystal.lattice)
    bounds = radius * np.linalg.norm(inv, axis=0)
    delta = np.asarray(crystal.frac) - crystal.frac[index]
    # Vectorize atoms for each exact reciprocal-bounded image. This retains
    # self images and skew-cell correctness without O(N) Python work per image.
    lower = np.ceil(-delta - bounds - 1e-10).astype(int).min(axis=0)
    upper = np.floor(-delta + bounds + 1e-10).astype(int).max(axis=0)
    result = []
    for shift in product(*(range(a, b + 1) for a, b in zip(lower, upper))):
        vectors = (delta + shift) @ crystal.lattice
        lengths = np.linalg.norm(vectors, axis=1)
        included = lengths <= radius + 1e-10
        if shift == (0, 0, 0):
            included[index] = False
        for j in np.flatnonzero(included):
            result.append((int(j), vectors[j], float(lengths[j])))
    return result


def motifs(crystal, cutoffs=None):
    cutoffs = cutoffs or {"P": 2.0, "Fe": 2.5}
    rows = []
    for i, element in enumerate(crystal.elements):
        if element not in cutoffs:
            continue
        oxygen = [n for n in neighbors(crystal, i, cutoffs[element])
                  if crystal.elements[n[0]] == "O"]
        vectors = np.array([n[1] for n in oxygen])
        lengths = np.array([n[2] for n in oxygen])
        cn = len(oxygen)
        inside = False
        if cn >= 4:
            try:
                hull = ConvexHull(vectors)
                # Strictly inside, not on a flat face or a degenerate polyhedron.
                inside = bool(np.all(hull.equations[:, -1] < -1e-6))
            except QhullError:
                pass
        angles = []
        if cn >= 2 and lengths.min() > 0:
            unit = vectors / lengths[:, None]
            angles = np.sort(np.degrees(np.arccos(np.clip(
                (unit @ unit.T)[np.triu_indices(cn, 1)], -1, 1)))).tolist()
        rows.append({
            "site": i, "element": element, "cn": cn, "inside": inside,
            "oxygen_indices": sorted({n[0] for n in oxygen}),
            "bond_min": float(lengths.min()) if cn else None,
            "bond_max": float(lengths.max()) if cn else None,
            "off_center": float(np.linalg.norm(vectors.mean(0)) / lengths.mean())
                if cn and lengths.mean() > 0 else None,
            "bond_distortion": float(np.std(lengths) / lengths.mean())
                if cn and lengths.mean() > 0 else None,
            "angles": angles,
        })
    return rows


def calibrate(training, min_structures=20, cutoffs=None):
    """Train-only support by independent structure counts, stratified by CN.

    No rare CN is silently mapped to six. Insufficient support stays unresolved.
    Bond ranges are broad empirical quantiles with 0.1 A margin; must be audited
    on validation and cutoff-sensitivity runs before production use.
    """
    cutoffs = cutoffs or {"P": 2.0, "Fe": 2.5}
    grouped, support = defaultdict(list), defaultdict(set)
    ids = []
    for crystal in training:
        if crystal.id in ids:
            raise ValueError("duplicate training ID")
        ids.append(crystal.id)
        for row in motifs(crystal, cutoffs):
            key = f"{crystal.family}:{row['element']}:{row['cn']}"
            grouped[key].append(row)
            support[key].add(crystal.id)
    profiles = {}
    for key, rows in grouped.items():
        valid = [r for r in rows if r["inside"] and r["angles"]]
        profiles[key] = {"structures": len(support[key]), "sites": len(rows),
                         "inside_fraction": len(valid) / len(rows)}
        if valid:
            profiles[key].update(
                bond_min=float(np.quantile([r["bond_min"] for r in valid], .001) - .1),
                bond_max=float(np.quantile([r["bond_max"] for r in valid], .999) + .1),
                angle_reference=np.median([r["angles"] for r in valid], axis=0).tolist(),
            )
    return {"version": "four-stage-v1", "training_ids": ids,
            "min_structures": min_structures, "cutoffs": cutoffs, "profiles": profiles}


def hard_gates(crystal, condition, profile, forces=None, force_threshold=.05,
               external_filter=None, fe_valences=(2, 3), motif_backend="external"):
    """Final force = max atom Euclidean force norm, in eV/A; unknown fails.

    external_filter(crystal) must return {'passed': bool, 'details': ...}.
    Missing external tool is explicitly unverified, never silently accepted.
    """
    if not np.isfinite(force_threshold) or force_threshold <= 0:
        raise ValueError("force_threshold must be positive")
    if motif_backend not in {"external", "native"}:
        raise ValueError("unknown motif backend")
    frac, cell = np.asarray(crystal.frac), np.asarray(crystal.lattice)
    finite = (frac.shape == (len(crystal.elements), 3) and cell.shape == (3, 3)
              and np.isfinite(frac).all() and np.isfinite(cell).all())
    cell_ok = bool(finite and abs(np.linalg.det(cell)) > 1e-6
                   and np.linalg.cond(cell) < 100)
    counts = Counter(crystal.elements)
    # Na+, P5+, O2-; the explicitly configured Fe oxidation-state model.
    if not fe_valences or any(type(v) is not int or v <= 0 for v in fe_valences):
        raise ValueError("Fe valences must be positive integers")
    required = 2 * counts["O"] - counts["Na"] - 5 * counts["P"]
    totals = {0}
    for _ in range(counts["Fe"]):
        totals = {q + v for q in totals for v in fe_valences}
    charge = counts["Fe"] > 0 and required in totals
    checks = {"conditioning": condition.matches(crystal), "cell": cell_ok,
              "charge": bool(charge)}
    f = np.asarray(forces) if forces is not None else np.array([])
    force_known = f.shape == (len(crystal.elements), 3) and np.isfinite(f).all()
    fmax = float(np.linalg.norm(f, axis=1).max()) if force_known and len(f) else None
    checks["final_force"] = fmax is not None and fmax <= force_threshold
    checks["overlap"] = checks["motif"] = checks["oxygen_coverage"] = False
    rows, violations = [], []
    if cell_ok and checks["conditioning"]:
        # Conservative catastrophic-overlap limits, not equilibrium bond targets.
        radii = {"Na": 1.66, "Fe": 1.32, "P": 1.07, "O": .66}
        for i, e in enumerate(crystal.elements):
            for j, _, distance in neighbors(crystal, i, 2.0):
                threshold = .6 * (radii[e] + radii[crystal.elements[j]])
                if distance < threshold:
                    violations.append({"i": i, "j": j, "distance": distance})
        checks["overlap"] = not violations
        rows = motifs(crystal, profile["cutoffs"])
        for row in rows:
            key = f"{crystal.family}:{row['element']}:{row['cn']}"
            ref = profile["profiles"].get(key, {})
            supported = (ref.get("structures", 0) >= profile["min_structures"]
                         and "bond_min" in ref)
            row["supported"] = supported
            row["passed"] = bool(supported and row["inside"]
                and (row["element"] != "P" or row["cn"] == 4)
                and row["bond_min"] >= ref["bond_min"]
                and row["bond_max"] <= ref["bond_max"])
            if supported and len(row["angles"]) == len(ref["angle_reference"]):
                target = ([109.4712206] * 6 if row["element"] == "P"
                          else ref["angle_reference"])
                row["angle_distortion"] = float(np.sqrt(np.mean(
                    (np.asarray(row["angles"]) - target) ** 2)) / 180)
        checks["motif"] = bool(rows and all(r["passed"] for r in rows))
        covered = {i for r in rows for i in r["oxygen_indices"]}
        checks["oxygen_coverage"] = covered == {i for i, e in enumerate(crystal.elements) if e == "O"}
    external = {"passed": False, "details": "PolyAnionCathodeFilter unavailable"}
    if external_filter is not None and cell_ok and checks["conditioning"]:
        external = external_filter(crystal)
        if type(external.get("passed")) is not bool:
            raise ValueError("external filter must return an explicit boolean passed")
    if motif_backend == "external":
        checks["external_motif"] = external["passed"]
    else:
        external = {"passed": None, "details": "native periodic motif implementation; external tool not invoked"}
    metrics = {}
    if checks["motif"]:
        for element in ("P", "Fe"):
            sites = [r for r in rows if r["element"] == element]
            for metric in ("off_center", "bond_distortion", "angle_distortion"):
                metrics[f"{element}_{metric}"] = max(r[metric] for r in sites)
            # Worst site support rarity; continuous local CN quality proxy.
            frequencies = [profile["profiles"][f"{crystal.family}:{element}:{r['cn']}"]
                           ["structures"] / len(profile["training_ids"]) for r in sites]
            metrics[f"{element}_coordination_rarity"] = 1 - min(frequencies)
    return {"id": crystal.id, "passed": all(checks.values()), "checks": checks,
            "motif_backend": motif_backend,
            "charge_model": {"Na": 1, "P": 5, "O": -2, "Fe": list(fe_valences),
                             "required_Fe_average": required / counts["Fe"] if counts["Fe"] else None},
            "force_max_eV_A": fmax, "sites": rows, "overlaps": violations,
            "external": external, "metrics": metrics}


def robust_rank(candidates, metrics):
    """Within composition/family, rank lower-is-better metrics; worst first.

    Candidates need id, passed, group, metrics. Missing/nonfinite quality fails
    loudly. Ties use all remaining worst ranks, then energy, then stable ID.
    """
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("provide unique ranking metrics")
    groups = defaultdict(list)
    for row in candidates:
        if row["passed"]:
            groups[row["group"]].append(row)
    output = {}
    for group, rows in groups.items():
        values = np.array([[r["metrics"][m] for m in metrics] for r in rows], float)
        if not np.isfinite(values).all():
            raise ValueError("ranking metrics must be finite")
        percentiles = (rankdata(values, axis=0, method="average") - 1) / max(1, len(rows)-1)
        ranked = []
        for r, percent in zip(rows, percentiles):
            key = tuple(sorted(percent.tolist(), reverse=True))
            ranked.append({**r, "percentiles": dict(zip(metrics, percent.tolist())),
                           "minimax_key": key})
        output[group] = sorted(ranked, key=lambda r: (
            r["minimax_key"], r["metrics"].get("energy_eV_atom", r["metrics"].get("E_UMA_eV_atom", 0)), r["id"]))
    return output


def diversity_select(ranked, distances, training_distances, k, quality_pool,
                     duplicate_threshold=.02):
    """Farthest-first ONLY within the supplied ranked quality pool.

    Distances use ranked order; training_distances are nearest TRAIN distances.
    Caller must use a periodic, permutation/rotation/cell-invariant descriptor
    and record its version. No novelty enters source optimization.
    """
    n = len(ranked)
    d, train = np.asarray(distances, float), np.asarray(training_distances, float)
    if k < 0 or quality_pool < 0 or duplicate_threshold < 0:
        raise ValueError("selection budgets/threshold must be nonnegative")
    if d.shape != (n, n) or train.shape != (n,) or not np.isfinite(d).all() or not np.isfinite(train).all():
        raise ValueError("finite candidate and train distances required")
    if (d < 0).any() or (train < 0).any() or not np.allclose(d, d.T) or not np.allclose(np.diag(d), 0):
        raise ValueError("invalid distance matrix")
    eligible = []
    for i in range(min(n, quality_pool)):
        if train[i] < duplicate_threshold:
            continue
        if all(d[i, j] >= duplicate_threshold for j in eligible):
            eligible.append(i)
    chosen = []
    if eligible and k:
        chosen.append(eligible.pop(0))  # anchor the best-quality structure
    while eligible and len(chosen) < k:
        i = max(eligible, key=lambda i: (min(train[i], min(d[i, j] for j in chosen)), -i))
        chosen.append(i)
        eligible.remove(i)
    return [ranked[i] for i in chosen]

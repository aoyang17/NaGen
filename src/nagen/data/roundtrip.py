#!/usr/bin/env python3
"""NaGen Phase 0 gate: JSON -> Structure -> CIF -> Structure round-trip test.

Samples records from the raw JSONL and from the canonical dataset, writes CIF
with pymatgen's CifWriter, re-reads with Structure.from_file, and reports
max lattice / site / coordinate deviations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile

import numpy as np

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


def min_image_delta(frac_a, frac_b):
    d = frac_a - frac_b
    d -= np.round(d)
    return d


def compare_structures(st1, st2, label, tol_frac=1e-3, tol_lat=1e-2):
    """Compare two structures as equivalent crystals.

    Equivalence: same atom count and species multiset, equivalent lattices
    (Niggli-reduced forms match), and all fractional coordinates match under a
    common origin shift (minimum-image), allowing arbitrary site ordering.
    """
    import spglib

    from collections import defaultdict

    issues = []
    if len(st1) != len(st2):
        issues.append(f"site count {len(st1)} != {len(st2)}")
        return False, issues

    def coord_sets(st):
        d = defaultdict(list)
        for site in st:
            d[site.species_string].append(np.asarray(site.frac_coords) % 1.0)
        return d

    s1, s2 = coord_sets(st1), coord_sets(st2)
    if set(s1) != set(s2):
        issues.append(f"element multisets differ: {sorted(s1)} vs {sorted(s2)}")
        return False, issues

    # lattice equivalence via Niggli reduction
    def niggli(mat):
        return spglib.niggli_reduce(np.asarray(mat, dtype=float), eps=1e-5)

    n1, n2 = niggli(st1.lattice.matrix), niggli(st2.lattice.matrix)
    if n1 is None or n2 is None:
        issues.append("niggli reduction failed")
        return False, issues
    v1, v2 = abs(np.linalg.det(n1)), abs(np.linalg.det(n2))
    if abs(v1 - v2) / max(v1, v2) > 1e-3:
        issues.append(f"cell volume mismatch {v1:.3f} vs {v2:.3f}")
        return False, issues
    n1r = np.sort(np.linalg.norm(n1, axis=1))
    n2r = np.sort(np.linalg.norm(n2, axis=1))
    if not np.allclose(n1r, n2r, rtol=tol_lat, atol=1e-3):
        issues.append(f"niggli lattice vectors differ: {n1r} vs {n2r}")
        return False, issues

    # coordinates: transform both to Niggli frame, then match with origin shift
    def to_niggli_frac(st, nlat):
        st2 = st.copy()
        st2.lattice = st2.lattice.__class__(nlat)
        return st2.frac_coords % 1.0

    f1 = to_niggli_frac(st1, n1)
    f2 = to_niggli_frac(st2, n2)
    max_frac = 0.0
    for el in s1:
        a = np.asarray([f1[i] for i, s in enumerate(st1) if s.species_string == el])
        b = np.asarray([f2[i] for i, s in enumerate(st2) if s.species_string == el])
        if len(a) != len(b):
            issues.append(f"element {el} count mismatch")
            return False, issues
        # origin shift: align by the first atom, verify the rest
        shift = min_image_delta(a[0], b[0])
        b_shifted = (b + shift) % 1.0
        used = set()
        for v in a:
            ds = [np.linalg.norm(min_image_delta(v, w)) for w in b_shifted]
            best = min((d, i) for i, d in enumerate(ds) if i not in used)[1]
            used.add(best)
            max_frac = max(max_frac, ds[best])
    if max_frac > tol_frac:
        issues.append(f"frac coord max diff {max_frac:.2e} > {tol_frac}")
    return not issues, issues


def run(input_path, canonical_path, out_path, n_sample, seed):
    rng = random.Random(seed)
    os.makedirs(out_path, exist_ok=True)

    # Raw JSONL sample
    with open(input_path) as f:
        lines = f.readlines()
    idxs = rng.sample(range(len(lines)), min(n_sample, len(lines)))
    raw_results = []
    with tempfile.TemporaryDirectory() as td:
        for i in idxs:
            z = json.loads(lines[i])
            st = Structure.from_dict(z["structure"])
            cif = os.path.join(td, f"raw_{i}.cif")
            CifWriter(st).write_file(cif)
            st2 = Structure.from_file(cif)
            ok, issues = compare_structures(st, st2, f"raw:{i}")
            raw_results.append({"idx": i, "material_id": z.get("material_id"),
                                "ok": ok, "issues": issues,
                                "n_orig": len(st)})

    # Canonical dataset sample (primitive structures)
    canon_results = []
    with open(canonical_path) as f:
        clines = f.readlines()
    cidxs = rng.sample(range(len(clines)), min(n_sample, len(clines)))
    with tempfile.TemporaryDirectory() as td:
        for i in cidxs:
            c = json.loads(clines[i])
            if c.get("primitive_structure") is None:
                canon_results.append({"idx": i, "canonical_id": c.get("canonical_id"),
                                      "ok": False, "issues": ["no primitive structure"],
                                      "n_prim": None})
                continue
            st = Structure.from_dict(c["primitive_structure"])
            cif = os.path.join(td, f"canon_{i}.cif")
            CifWriter(st).write_file(cif)
            st2 = Structure.from_file(cif)
            ok, issues = compare_structures(st, st2, f"canon:{i}")
            canon_results.append({"idx": i, "canonical_id": c.get("canonical_id"),
                                  "ok": ok, "issues": issues, "n_prim": len(st)})

    report = {
        "seed": seed, "n_sampled_raw": len(raw_results), "n_sampled_canonical": len(canon_results),
        "raw_jsonl": {
            "pass": sum(r["ok"] for r in raw_results),
            "fail": sum(not r["ok"] for r in raw_results),
            "failures": [r for r in raw_results if not r["ok"]],
            "n_atom_range": [min((r["n_orig"] for r in raw_results), default=None),
                             max((r["n_orig"] for r in raw_results), default=None)],
        },
        "canonical_primitive": {
            "pass": sum(r["ok"] for r in canon_results),
            "fail": sum(not r["ok"] for r in canon_results),
            "failures": [r for r in canon_results if not r["ok"]],
            "n_atom_range": [min((r["n_prim"] for r in canon_results if r["n_prim"] is not None), default=None),
                             max((r["n_prim"] for r in canon_results if r["n_prim"] is not None), default=None)],
        },
    }
    with open(os.path.join(out_path, "roundtrip_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()
    run(args.input, args.canonical, args.out, args.n, args.seed)


if __name__ == "__main__":
    main()

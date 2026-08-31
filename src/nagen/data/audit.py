#!/usr/bin/env python3
"""NaGen Phase 0: streaming full-data audit for Na-TM-P-O cathode JSONL.

Reads /mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl strictly read-only and
writes per-record CSV, canonical dataset JSONL, aggregate JSON and provenance
under --out.  Uses pymatgen + spglib for primitive/Niggli reduction, space
groups and duplicate fingerprints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter

import numpy as np

from pymatgen.core import Composition, Element, Structure

TARGET_TM = ("Fe", "Mn", "Ti", "V")
ROLE = {"P": "P", "O": "O", "Na": "Na", "F": "F", "C": "C", "N": "N"}


def role(e):
    return ROLE.get(e, "TM" if e in TARGET_TM else "OTHER")


COVALENT_RADIUS = {
    "Na": 1.66, "Fe": 1.32, "Mn": 1.39, "Ti": 1.60, "V": 1.53,
    "P": 1.07, "O": 0.66, "F": 0.57, "C": 0.76, "N": 0.71, "H": 0.31,
}

PAIR_TYPES = ("P-O", "P-P", "Na-O", "TM-O", "O-O", "Na-Na", "TM-TM", "Na-TM", "ANY")


def pair_name(a, b):
    ra, rb = role(a), role(b)
    if ra == "OTHER" or rb == "OTHER":
        return None
    key = tuple(sorted((ra, rb)))
    if key == ("O", "P"):
        return "P-O"
    if key == ("P", "P"):
        return "P-P"
    if key == ("Na", "O"):
        return "Na-O"
    if key == ("O", "TM"):
        return "TM-O"
    if key == ("O", "O"):
        return "O-O"
    if key == ("Na", "Na"):
        return "Na-Na"
    if key == ("TM", "TM"):
        return "TM-TM"
    if key == ("Na", "TM"):
        return "Na-TM"
    return None


def pbc_dist_matrix(frac, lat):
    d = frac[:, None, :] - frac[None, :, :]
    d -= np.round(d)
    cart = d @ lat
    return np.sqrt(np.maximum(np.einsum("ijm,ijm->ij", cart, cart), 0.0))


def lattice_params(M):
    a = float(np.linalg.norm(M[0]))
    b = float(np.linalg.norm(M[1]))
    c = float(np.linalg.norm(M[2]))

    def ang(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v)) / (nu * nv)))))

    return a, b, c, ang(M[1], M[2]), ang(M[0], M[2]), ang(M[0], M[1]), float(abs(np.linalg.det(M)))


def site_elements(st):
    elems = []
    fracs = []
    disorder = 0
    for site in st:
        sp = site.species
        if len(sp) != 1 or abs(sp.num_atoms - 1.0) > 1e-6:
            disorder = 1
        dominant = max(sp, key=lambda x: sp[x])
        elems.append(dominant.symbol)
        fracs.append(site.frac_coords)
    return elems, np.asarray(fracs, dtype=float), disorder


def min_pair_dists(D, elems):
    n = len(elems)
    iu = np.triu_indices(n, k=1)
    out = {pt: np.inf for pt in PAIR_TYPES}
    for i, j in zip(iu[0], iu[1]):
        pn = pair_name(elems[i], elems[j])
        d = D[i, j]
        if d < out["ANY"]:
            out["ANY"] = d
        if pn is not None and d < out[pn]:
            out[pn] = d
    return {k: (None if v == np.inf else float(v)) for k, v in out.items()}


def classify_po4(D, elems, cutoff=2.15):
    ps = [i for i, e in enumerate(elems) if role(e) == "P"]
    oxy = [i for i, e in enumerate(elems) if role(e) == "O"]
    p_neigh = [[] for _ in ps]
    o_degree = [0] * len(oxy)
    for oi, o in enumerate(oxy):
        for pi, p in enumerate(ps):
            if D[p, o] <= cutoff:
                p_neigh[pi].append(oi)
                o_degree[oi] += 1
    q = [sum(1 for oi in ns if o_degree[oi] >= 2) for ns in p_neigh]
    qhist = Counter(q)
    unbound = sum(d == 0 for d in o_degree) / max(1, len(oxy))
    p4 = sum(len(ns) == 4 for ns in p_neigh) / max(1, len(ps))
    if not ps:
        label = "no_P"
    elif p4 < 0.75:
        label = "irregular_PO"
    elif len(qhist) == 1:
        label = "Q" + str(next(iter(qhist)))
    else:
        label = "mixed_Q"
    return label, dict(qhist), unbound, p4


def tm_coord_hist(D, elems, cutoff=2.4):
    tm_idx = [i for i, e in enumerate(elems) if role(e) == "TM"]
    o_idx = [i for i, e in enumerate(elems) if role(e) == "O"]
    if not tm_idx:
        return {}, 0
    cn = Counter()
    for i in tm_idx:
        cn[int((D[i][o_idx] <= cutoff).sum())] += 1
    return dict(cn), len(tm_idx)


def abnormal_neighbors(D, elems, frac=0.75):
    n = len(elems)
    cnt = 0
    min_ratio = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            r = COVALENT_RADIUS.get(elems[i], 1.5) + COVALENT_RADIUS.get(elems[j], 1.5)
            ratio = D[i, j] / r
            if ratio < min_ratio:
                min_ratio = ratio
            if D[i, j] < frac * r:
                cnt += 1
    return cnt, None if min_ratio == np.inf else float(min_ratio)


_OXI_CACHE = {}


def oxi_guess_count(reduced_formula):
    if reduced_formula in _OXI_CACHE:
        return _OXI_CACHE[reduced_formula]
    try:
        guesses = Composition(reduced_formula).oxi_state_guesses()
        _OXI_CACHE[reduced_formula] = len(guesses)
    except Exception:
        _OXI_CACHE[reduced_formula] = -1
    return _OXI_CACHE[reduced_formula]


def canonical_key(lat, frac, nums):
    a, b, c, al, be, ga, _ = lattice_params(lat)
    params = tuple(round(x, 4) for x in (a, b, c, al, be, ga))
    sites = sorted((int(n), tuple(round(x, 3) for x in f)) for n, f in zip(nums, frac))
    return json.dumps({"lat": params, "sites": sites}, sort_keys=True)


def prototype_key(lat, frac, nums, coord_q=0.05):
    a, b, c, al, be, ga, _ = lattice_params(lat)
    params = tuple(round(x, 2) for x in (a, b, c, al, be, ga))
    sites = sorted((int(n), tuple(round(x / coord_q) for x in f)) for n, f in zip(nums, frac))
    return json.dumps({"lat": params, "sites": sites}, sort_keys=True)


ENERGY_FIELDS = ("optimized_energy_per_atom", "energy_above_hull",
                 "energy_per_atom", "formation_energy_per_atom")


def audit_record(z, idx):
    """Audit one raw record -> (row dict, canonical dict). Raises on malformed."""
    import spglib

    rec = {"idx": idx, "canonical_id": f"NaGen-{idx:05d}",
           "material_id": z.get("material_id"), "formula_field": z.get("formula")}
    for f in ENERGY_FIELDS:
        v = z.get(f)
        rec[f + "_present"] = int(v is not None)
        rec[f + "_value"] = v
    num_fields = {k: v for k, v in z.items() if k.startswith("num_")}
    rec["num_field_count"] = len(num_fields)
    rec["num_fields_present"] = int(all(v is not None for v in num_fields.values())) if num_fields else 0

    st = Structure.from_dict(z["structure"])
    rec["charge"] = float(st.charge) if st.charge is not None else None
    elems, fracs, disorder = site_elements(st)
    rec["disorder"] = disorder
    counts = Counter(elems)
    rec["n_orig"] = len(elems)
    rec["element_counts"] = json.dumps(dict(counts), sort_keys=True)
    red = Composition.from_dict({e: c for e, c in counts.items()}).reduced_formula
    rec["reduced_formula"] = red
    try:
        field_red = Composition(rec["formula_field"]).reduced_formula
        rec["formula_mismatch"] = int(field_red != red)
    except Exception:
        rec["formula_mismatch"] = -1

    a, b, c, al, be, ga, vol = lattice_params(st.lattice.matrix)
    rec.update({"a": a, "b": b, "c": c, "alpha": al, "beta": be, "gamma": ga,
                "volume": vol, "vol_per_atom": vol / max(1, len(elems))})

    cell = (np.asarray(st.lattice.matrix, dtype=float), fracs,
            np.asarray([Element(e).Z for e in elems], dtype=int))
    rec["prim_ok"] = 1
    rec["niggli_ok"] = 1
    prim_lat = None
    prim_frac = None
    prim_num = None
    try:
        prim = spglib.find_primitive(cell, symprec=1e-3)
        if prim is None:
            rec["prim_ok"] = 0
        else:
            p_lat, p_frac, p_num = prim
            prim_lat, prim_frac, prim_num = p_lat, p_frac, p_num
            rec["n_prim"] = len(p_num)
            pa, pb, pc, pal, pbe, pga, pvol = lattice_params(p_lat)
            rec.update({"prim_a": pa, "prim_b": pb, "prim_c": pc,
                        "prim_alpha": pal, "prim_beta": pbe, "prim_gamma": pga,
                        "prim_volume": pvol, "prim_vol_per_atom": pvol / max(1, len(p_num))})
            niggli = spglib.niggli_reduce(p_lat, eps=1e-5)
            if niggli is None:
                rec["niggli_ok"] = 0
                niggli = p_lat
            rec["n_niggli"] = len(p_num)
            na, nb, nc, nal, nbe, nga, nvol = lattice_params(niggli)
            rec.update({"niggli_a": na, "niggli_b": nb, "niggli_c": nc,
                        "niggli_alpha": nal, "niggli_beta": nbe, "niggli_gamma": nga,
                        "niggli_volume": nvol})
            rec["dup_key"] = canonical_key(niggli, p_frac, p_num)
            rec["proto_key"] = prototype_key(niggli, p_frac, p_num)
    except Exception as e:
        rec["prim_ok"] = 0
        rec["niggli_ok"] = 0
        rec["n_prim"] = None
        rec["dup_key"] = None
        rec["proto_key"] = None
        rec["prim_error"] = str(e)[:80]

    try:
        ds = spglib.get_symmetry_dataset(cell, symprec=1e-3)
        rec["spacegroup_number"] = ds.number
        rec["spacegroup_symbol"] = ds.international
    except Exception:
        rec["spacegroup_number"] = None
        rec["spacegroup_symbol"] = None

    D = pbc_dist_matrix(fracs, st.lattice.matrix)
    md = min_pair_dists(D, elems)
    for k, v in md.items():
        rec["min_d_" + k.lower().replace("-", "_")] = v
    abn, min_ratio = abnormal_neighbors(D, elems)
    rec["abnormal_neighbors"] = abn
    rec["min_cov_ratio"] = min_ratio

    label, qhist, unbound, p4 = classify_po4(D, elems)
    rec.update({"q_label": label, "q_hist": json.dumps(qhist, sort_keys=True),
                "unbound_o_frac": unbound, "po4_frac": p4})
    tmc, ntm = tm_coord_hist(D, elems)
    rec["n_tm"] = ntm
    rec["tm_coord_hist"] = json.dumps(tmc, sort_keys=True)
    rec["oxi_guess_count"] = oxi_guess_count(red)

    flags = []
    if rec["formula_mismatch"] == 1:
        flags.append("formula_mismatch")
    if disorder:
        flags.append("disorder")
    if rec["oxi_guess_count"] == 0:
        flags.append("no_oxi_guess")
    if abn > 0:
        flags.append("abnormal_neighbors")
    if p4 < 0.75:
        flags.append("irregular_po4")
    if unbound > 0.05:
        flags.append("unbound_o")
    if not rec["optimized_energy_per_atom_present"]:
        flags.append("missing_energy")
    ev = rec["optimized_energy_per_atom_value"]
    if ev is not None and not (-10.0 <= ev <= -3.0):
        flags.append("energy_outlier")
    if not rec.get("prim_ok"):
        flags.append("primitive_failed")
    if vol <= 0:
        flags.append("nonpositive_volume")
    if any(role(e) == "OTHER" for e in elems):
        flags.append("other_elements")
    if counts.get("F", 0) > 0:
        flags.append("contains_F")
    if counts.get("C", 0) + counts.get("N", 0) > 0:
        flags.append("contains_CN")
    rec["flags"] = ";".join(flags)

    canonical = {"idx": idx, "canonical_id": rec["canonical_id"],
                 "raw_id": rec["material_id"], "reduced_formula": red,
                 "q_label": label, "flags": rec["flags"],
                 "original_structure": st.as_dict()}
    if prim_lat is not None:
        try:
            prim_species = [Element.from_Z(int(n)).symbol for n in prim_num]
            prim_st = Structure(prim_lat, prim_species, prim_frac)
            canonical["primitive_structure"] = prim_st.as_dict()
        except Exception:
            canonical["primitive_structure"] = None
    else:
        canonical["primitive_structure"] = None
    return rec, canonical


CSV_COLS = [
    "idx", "canonical_id", "material_id", "formula_field", "reduced_formula",
    "n_orig", "n_prim", "n_niggli", "spacegroup_number", "spacegroup_symbol",
    "a", "b", "c", "alpha", "beta", "gamma", "volume", "vol_per_atom",
    "prim_a", "prim_b", "prim_c", "prim_volume", "prim_vol_per_atom",
    "niggli_a", "niggli_b", "niggli_c", "niggli_volume",
    "optimized_energy_per_atom_present", "optimized_energy_per_atom_value",
    "energy_above_hull_present", "energy_above_hull_value",
    "energy_per_atom_present", "energy_per_atom_value",
    "formation_energy_per_atom_present", "formation_energy_per_atom_value",
    "num_field_count", "num_fields_present", "charge", "disorder", "formula_mismatch",
    "q_label", "q_hist", "po4_frac", "unbound_o_frac", "n_tm", "tm_coord_hist",
    "oxi_guess_count", "min_d_any", "min_d_p_o", "min_d_p_p", "min_d_na_o",
    "min_d_tm_o", "min_d_o_o", "min_d_na_na", "min_d_tm_tm", "min_d_na_tm",
    "abnormal_neighbors", "min_cov_ratio",
    "dup_key", "proto_key", "flags",
]


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _stats(vals, pct=(10, 25, 50, 75, 90)):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    out = {"n": len(s), "min": s[0], "max": s[-1],
           "mean": statistics.fmean(s), "median": s[len(s) // 2]}
    for p in pct:
        out[f"p{p}"] = s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]
    return out


def aggregate(rows, input_path, input_digest):
    n = len(rows)
    missing = {}
    for f in ENERGY_FIELDS:
        present = sum(r[f + "_present"] for r in rows)
        missing[f] = {"present": present, "missing": n - present,
                      "missing_rate": (n - present) / max(1, n)}
    missing["num_*_fields"] = {
        "records_with_zero_num_fields": sum(r["num_field_count"] == 0 for r in rows),
        "records_with_any_null_num_field": sum(
            r["num_field_count"] > 0 and not r["num_fields_present"] for r in rows)}

    elem_total = Counter()
    elem_records = Counter()
    formulas = Counter()
    for r in rows:
        ec = json.loads(r["element_counts"])
        for e, c in ec.items():
            elem_total[e] += c
            elem_records[e] += 1
        formulas[r["reduced_formula"]] += 1

    atom_orig = [r["n_orig"] for r in rows]
    atom_prim = [r["n_prim"] for r in rows if r.get("n_prim") is not None]
    atom_niggli = [r["n_niggli"] for r in rows if r.get("n_niggli") is not None]
    energies = [r["optimized_energy_per_atom_value"] for r in rows
                if r.get("optimized_energy_per_atom_value") is not None]
    vpa_orig = [r["vol_per_atom"] for r in rows]
    vpa_prim = [r["prim_vol_per_atom"] for r in rows if r.get("prim_vol_per_atom") is not None]

    def frac_over(vals, thresh):
        return sum(1 for v in vals if v is not None and v > thresh) / max(1, len(vals))

    sg = Counter(r["spacegroup_number"] for r in rows if r.get("spacegroup_number"))
    q = Counter(r["q_label"] for r in rows)
    flags = Counter()
    for r in rows:
        for fl in (r["flags"].split(";") if r["flags"] else []):
            flags[fl] += 1
    tm_coord = Counter()
    for r in rows:
        for k, v in json.loads(r["tm_coord_hist"] or "{}").items():
            tm_coord[int(k)] += v

    min_d = {}
    for pt in PAIR_TYPES:
        col = "min_d_" + pt.lower().replace("-", "_")
        vals = [r[col] for r in rows if r.get(col) is not None]
        if vals:
            min_d[pt] = _stats(vals)

    return {
        "input": input_path, "input_first_1MiB_sha256": input_digest,
        "n_records": n,
        "missing_fields": missing,
        "unique_reduced_formulas": len(formulas),
        "formula_histogram_top20": formulas.most_common(20),
        "element_total_counts": dict(elem_total.most_common()),
        "element_record_coverage": dict(elem_records.most_common()),
        "atom_count": {"original": _stats(atom_orig), "primitive": _stats(atom_prim),
                       "niggli": _stats(atom_niggli)},
        "atom_count_primitive_fractions": {
            ">20": frac_over(atom_prim, 20), ">40": frac_over(atom_prim, 40),
            ">80": frac_over(atom_prim, 80)},
        "vol_per_atom": {"original": _stats(vpa_orig), "primitive": _stats(vpa_prim)},
        "spacegroup_top20": sg.most_common(20),
        "q_label_counts": dict(q),
        "tm_coordination_hist": dict(sorted(tm_coord.items())),
        "min_pair_distances_A": min_d,
        "disorder_records": sum(r["disorder"] for r in rows),
        "formula_mismatch_records": sum(1 for r in rows if r["formula_mismatch"] == 1),
        "no_oxi_guess_records": sum(1 for r in rows if r.get("oxi_guess_count") == 0),
        "abnormal_neighbor_pairs": sum(r["abnormal_neighbors"] for r in rows),
        "abnormal_neighbor_records": sum(1 for r in rows if r["abnormal_neighbors"] > 0),
        "unbound_o_gt5pct_records": sum(1 for r in rows if r["unbound_o_frac"] > 0.05),
        "irregular_po4_records": sum(1 for r in rows if r["po4_frac"] < 0.75),
        "energy_optimized_eV_per_atom": _stats(energies),
        "flag_counts": dict(flags),
    }


def _run(t):
    return t[0], audit_record(t[1], t[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    hasher = hashlib.sha256()
    with open(args.input, "rb") as f:
        hasher.update(f.read(1024 * 1024))
    input_digest = hasher.hexdigest()

    lines = []
    with open(args.input) as f:
        lines = f.readlines()
    n_lines = len(lines)
    print(f"[audit] {n_lines} lines, {args.workers} workers", flush=True)

    tasks = []
    malformed_lines = []
    for i, line in enumerate(lines):
        try:
            tasks.append((i, json.loads(line)))
        except Exception as e:
            malformed_lines.append({"line": i + 1, "error": str(e)[:120]})

    results = []
    with mp.Pool(args.workers) as pool:
        imap = pool.imap_unordered(_run, tasks, chunksize=8)
        for idx, (rec, canonical) in imap:
            results.append((idx, rec, canonical))
            if len(results) % 2000 == 0:
                print(f"[audit] {len(results)}/{len(tasks)} in {time.time()-t0:.0f}s",
                      flush=True)

    results.sort(key=lambda x: x[0])
    rows = [r for _, r, _ in results]
    canonicals = [c for _, _, c in results]

    csv_path = os.path.join(args.out, "records.csv")
    write_csv(rows, csv_path)
    with open(os.path.join(args.out, "canonical_v1.jsonl"), "w") as f:
        for c in canonicals:
            f.write(json.dumps(c) + "\n")

    summary = aggregate(rows, args.input, input_digest)
    summary["malformed_lines"] = malformed_lines
    summary["n_malformed"] = len(malformed_lines)
    summary["code"] = os.path.abspath(__file__)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        from importlib.metadata import version
        json.dump({
            "input": args.input, "input_first_1MiB_sha256": input_digest,
            "lines": n_lines, "seed": args.seed,
            "pymatgen": version("pymatgen"),
            "spglib": version("spglib"),
            "numpy": np.__version__, "python": sys.version.split()[0],
            "script": os.path.abspath(__file__),
            "out": os.path.abspath(args.out),
        }, f, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000], flush=True)
    print(f"[audit] done in {time.time()-t0:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

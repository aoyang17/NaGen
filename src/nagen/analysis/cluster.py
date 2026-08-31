#!/usr/bin/env python3
"""Dependency-free structural cluster audit for Na cathode JSONL data.

The input is read-only. All summaries are emitted under --out.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict


def elem(site):
    sp = site.get("species", [])
    return sp[0].get("element", site.get("label", "?")) if sp else site.get("label", "?")


def dist_frac(a, b, lattice):
    # Fast minimum-image convention; appropriate for these conventional cells.
    d = [a[i] - b[i] for i in range(3)]
    d = [x - round(x) for x in d]
    c = [sum(d[i] * lattice[i][j] for i in range(3)) for j in range(3)]
    return math.sqrt(sum(x*x for x in c))


def percentile(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    z = (len(ys) - 1) * p
    lo = int(z)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (z - lo)


def classify_q(sites, lattice, cutoff=2.15):
    ps = [s for s in sites if elem(s) == "P"]
    oxy = [s for s in sites if elem(s) == "O"]
    p_neigh = [[] for _ in ps]; o_degree = []
    for oi, o in enumerate(oxy):
        near = []
        for pi, p in enumerate(ps):
            if dist_frac(o["abc"], p["abc"], lattice) <= cutoff:
                near.append(pi); p_neigh[pi].append(oi)
        o_degree.append(len(near))
    q = []
    for ns in p_neigh:
        q.append(sum(1 for oi in ns if o_degree[oi] >= 2))
    qhist = Counter(q)
    unbound = sum(d == 0 for d in o_degree) / max(1, len(oxy))
    p4 = sum(len(ns) == 4 for ns in p_neigh) / max(1, len(ps))
    if not ps: label = "no_P"
    elif p4 < 0.75: label = "irregular_PO"
    elif len(qhist) == 1: label = "Q" + str(next(iter(qhist)))
    else: label = "mixed_Q"
    return label, qhist, unbound, p4


PAIR_TYPES = ("P-O", "P-P", "Na-O", "TM-O", "O-O")
NBINS, BINW = 30, 0.2


def role(e):
    if e in ("P", "O", "Na"): return e
    return "TM"


def pair_name(a, b):
    a, b = role(a), role(b)
    s = {a, b}
    if s == {"P", "O"}: return "P-O"
    if a == b == "P": return "P-P"
    if s == {"Na", "O"}: return "Na-O"
    if s == {"TM", "O"}: return "TM-O"
    if a == b == "O": return "O-O"
    return None


def features(struct):
    sites = struct["sites"]; latd = struct["lattice"]; lattice = latd["matrix"]
    counts = Counter(elem(s) for s in sites); n = max(1, len(sites)); vol = latd["volume"]
    base = [latd[x] / (vol ** (1/3)) for x in ("a", "b", "c")]
    base += [latd[x] / 180.0 for x in ("alpha", "beta", "gamma")]
    base += [vol/n, counts["Na"]/n, counts["P"]/n, counts["O"]/n,
             len([x for x in counts if x not in ("Na","P","O")])/max(1,n)]
    h = {p: [0.0]*NBINS for p in PAIR_TYPES}
    for i in range(len(sites)):
        for j in range(i+1, len(sites)):
            pn = pair_name(elem(sites[i]), elem(sites[j]))
            if pn is None: continue
            d = dist_frac(sites[i]["abc"], sites[j]["abc"], lattice)
            k = int(d/BINW)
            if k < NBINS: h[pn][k] += 1
    # Per-atom normalization reduces supercell-size leakage.
    return base + [v/n for p in PAIR_TYPES for v in h[p]]


def standardize(rows):
    d=len(rows[0]); mu=[]; sd=[]
    for j in range(d):
        x=[r[j] for r in rows]; m=statistics.fmean(x); s=statistics.pstdev(x) or 1.0
        mu.append(m); sd.append(s)
    return [[(r[j]-mu[j])/sd[j] for j in range(d)] for r in rows]


def d2(a,b): return sum((x-y)**2 for x,y in zip(a,b))


def metrics(x, y, rng, pair_samples=100000, sil_samples=1200):
    by=defaultdict(list)
    for i,c in enumerate(y): by[c].append(i)
    valid={c:v for c,v in by.items() if len(v)>=10}
    keep=[i for c in valid for i in valid[c]]; labels=list(valid)
    within=[]; between=[]
    for _ in range(pair_samples):
        i,j=rng.sample(keep,2); z=math.sqrt(d2(x[i],x[j]))
        (within if y[i]==y[j] else between).append(z)
    # Approximate silhouette against up to 80 representatives per class.
    sil=[]
    for i in rng.sample(keep, min(sil_samples,len(keep))):
        av={}
        for c, inds in valid.items():
            refs=rng.sample(inds,min(80,len(inds))); ds=[math.sqrt(d2(x[i],x[j])) for j in refs if j!=i]
            if ds: av[c]=statistics.fmean(ds)
        a=av[y[i]]; b=min(v for c,v in av.items() if c!=y[i]); sil.append((b-a)/max(a,b,1e-12))
    # Stratified 80/20 nearest-centroid evaluation.
    train=[]; test=[]
    for c,inds in valid.items():
        z=inds[:]; rng.shuffle(z); k=max(1,int(.8*len(z))); train += z[:k]; test += z[k:]
    cents={}
    for c in labels:
        ids=[i for i in train if y[i]==c]
        cents[c]=[statistics.fmean(x[i][j] for i in ids) for j in range(len(x[0]))]
    pred=[min(labels,key=lambda c:d2(x[i],cents[c])) for i in test]
    acc=sum(p==y[i] for p,i in zip(pred,test))/max(1,len(test))
    base=max(len(valid[c]) for c in labels)/sum(len(valid[c]) for c in labels)
    return {"classes":{c:len(valid[c]) for c in labels}, "within_mean":statistics.fmean(within),
            "between_mean":statistics.fmean(between), "between_within_ratio":statistics.fmean(between)/statistics.fmean(within),
            "silhouette_approx":statistics.fmean(sil), "nearest_centroid_accuracy":acc, "majority_baseline":base}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    labels = []
    meta = []
    malformed = 0
    with open(args.input) as handle:
        for line in handle:
            try:
                record = json.loads(line)
                structure = record["structure"]
                label, q_hist, unbound, p4 = classify_q(
                    structure["sites"], structure["lattice"]["matrix"]
                )
                rows.append(features(structure))
                labels.append(label)
                meta.append(
                    (
                        record.get("material_id"),
                        record.get("formula"),
                        label,
                        json.dumps(dict(q_hist), sort_keys=True),
                        unbound,
                        p4,
                        record.get("optimized_energy_per_atom"),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                malformed += 1
    xs = standardize(rows)
    observed = metrics(xs, labels, rng)
    null = []
    for _ in range(20):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        null.append(
            metrics(xs, shuffled, rng, pair_samples=15000, sil_samples=300)[
                "silhouette_approx"
            ]
        )
    observed["permutation_silhouette"] = {
        "n": len(null),
        "mean": statistics.fmean(null),
        "p95": percentile(null, 0.95),
        "max": max(null),
    }
    observed["records_total"] = len(rows)
    observed["malformed"] = malformed
    observed["label_counts_all"] = dict(Counter(labels))
    observed["oxygen_unbound_fraction"] = {
        "median": percentile([m[4] for m in meta], 0.5),
        "p90": percentile([m[4] for m in meta], 0.9),
        "count_gt_0.05": sum(m[4] > 0.05 for m in meta),
    }
    observed["method"] = {
        "P_O_cutoff_A": 2.15,
        "rdf_bin_A": BINW,
        "rdf_max_A": NBINS * BINW,
        "distance": "z-scored Euclidean",
    }
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(observed, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "records.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "material_id",
                "formula",
                "q_label",
                "q_hist",
                "unbound_O_fraction",
                "PO4_fraction",
                "optimized_energy_per_atom",
            ]
        )
        writer.writerows(meta)
    with open(args.input, "rb") as handle:
        digest = hashlib.sha256(handle.read(1024 * 1024)).hexdigest()
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(
            {"input": args.input, "first_1MiB_sha256": digest, "seed": args.seed},
            f,
            indent=2,
        )
    with open(os.path.join(args.out, "README.md"), "w") as handle:
        handle.write(
            "# Structural cluster analysis\n\n"
            "Initial dependency-free audit of whether Q-type structural labels form "
            "separable clusters in a fixed lattice/composition/RDF feature space.\n\n"
            f"- Input: `{args.input}` (read-only)\n"
            f"- Seed: `{args.seed}`\n"
            "- Command: `python -m nagen.analysis.cluster INPUT --out OUTPUT`\n"
            "- Outputs: `summary.json`, `records.csv`, and `provenance.json`\n"
        )
    print(json.dumps(observed, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

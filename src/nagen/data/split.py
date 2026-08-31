#!/usr/bin/env python3
"""NaGen Phase 0: group split v1.

Groups records by (reduced_formula, prototype_key) so that no chemical system
leaks across splits.  All prototype groups of the same reduced formula are
assigned together (80/10/10, largest groups first).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records_csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    formulas = defaultdict(lambda: defaultdict(list))  # formula -> proto -> [idx]
    rows = []
    with open(args.records_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            formulas[r["reduced_formula"]][r.get("proto_key") or "NONE"].append(int(r["idx"]))

    n = len(rows)
    targets = {"train": round(n * args.train), "val": round(n * args.val),
               "test": n - round(n * args.train) - round(n * args.val)}

    # deterministic formula order: largest first, tie-break by formula string
    formula_list = sorted(formulas, key=lambda f: (-sum(len(v) for v in formulas[f].values()), f))
    assigned = {}
    counts = Counter()
    for formula in formula_list:
        size = sum(len(v) for v in formulas[formula].values())
        # pick split with most remaining capacity (normalized by target)
        best = min(("train", "val", "test"),
                   key=lambda s: (counts[s] + size) / targets[s])
        assigned[formula] = best
        counts[best] += size

    splits = {"train": [], "val": [], "test": []}
    proto_map = {}
    for formula, protos in formulas.items():
        s = assigned[formula]
        for proto, idxs in protos.items():
            splits[s].extend(idxs)
            for i in idxs:
                proto_map[i] = proto
    for s in splits:
        splits[s].sort()

    # leakage checks
    formula_splits = defaultdict(set)
    for r in rows:
        formula_splits[r["reduced_formula"]].add(assigned[r["reduced_formula"]])
    leak_formulas = {f: sorted(v) for f, v in formula_splits.items() if len(v) > 1}

    proto_in = {}
    for s, idxs in splits.items():
        for i in idxs:
            proto_in[i] = s
    proto_of = {int(r["idx"]): r.get("proto_key") or "NONE" for r in rows}

    stats = {
        "n_records": n,
        "targets": targets,
        "actual_counts": {s: len(v) for s, v in splits.items()},
        "actual_fractions": {s: round(len(v) / n, 4) for s, v in splits.items()},
        "unique_formulas_per_split": {
            s: len({r["reduced_formula"] for r in rows if proto_in[int(r["idx"])] == s})
            for s in splits},
        "unique_prototypes_per_split": {
            s: len({proto_of[int(r["idx"])] for r in rows if proto_in[int(r["idx"])] == s})
            for s in splits},
        "q_label_counts_per_split": {
            s: dict(Counter(r["q_label"] for r in rows if proto_in[int(r["idx"])] == s))
            for s in splits},
        "formula_leakage": leak_formulas,
        "n_formula_leakage": len(leak_formulas),
    }

    out = {"version": "v1", "seed": args.seed,
           "split_fractions": {"train": args.train, "val": args.val, "test": args.test},
           "splits": splits, "stats": stats}
    with open(os.path.join(args.out, "split_v1.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "split_by_idx.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "split"])
        for i in range(n):
            w.writerow([i, proto_in[i]])
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

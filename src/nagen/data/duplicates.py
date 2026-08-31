#!/usr/bin/env python3
"""NaGen Phase 0: duplicate / prototype clustering from audit fingerprints.

Groups records by exact fingerprint (dup_key) and coarse prototype key
(proto_key), confirms a sample of duplicate groups with pymatgen
StructureMatcher, and writes duplicate + prototype reports.
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
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirm-max", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    by_dup = defaultdict(list)
    by_proto = defaultdict(list)
    no_key = 0
    with open(args.records_csv, newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["idx"])
            if r.get("dup_key"):
                by_dup[r["dup_key"]].append(i)
                by_proto[r.get("proto_key") or "NONE"].append(i)
            else:
                no_key += 1

    dup_groups = {k: v for k, v in by_dup.items() if len(v) > 1}
    dup_records = sum(len(v) for v in dup_groups.values())
    proto_groups = {k: v for k, v in by_proto.items() if len(v) > 1}
    proto_records = sum(len(v) for v in proto_groups.values())

    # StructureMatcher confirmation on a sample of duplicate groups
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=False)
    canon = {}
    with open(args.canonical) as f:
        for line in f:
            c = json.loads(line)
            canon[c["idx"]] = c

    confirmed = 0
    rejected = []
    sample_groups = sorted(dup_groups.values(), key=len, reverse=True)[:args.confirm_max]
    for g in sample_groups:
        a = Structure.from_dict(canon[g[0]]["primitive_structure"])
        b = Structure.from_dict(canon[g[1]]["primitive_structure"])
        if sm.fit(a, b):
            confirmed += 1
        else:
            rejected.append({"group_size": len(g), "idx_a": g[0], "idx_b": g[1]})

    report = {
        "n_records": sum(len(v) for v in by_dup.values()) + no_key,
        "records_without_key": no_key,
        "unique_dup_keys": len(by_dup),
        "duplicate_groups": len(dup_groups),
        "records_in_duplicate_groups": dup_records,
        "max_duplicate_group_size": max((len(v) for v in dup_groups.values()), default=0),
        "duplicate_group_size_hist": dict(sorted(
            Counter(len(v) for v in dup_groups.values()).items())),
        "top_duplicate_groups": [
            {"size": len(v), "idx_sample": v[:8]} for v in
            sorted(dup_groups.values(), key=len, reverse=True)[:15]],
        "unique_prototypes": len(by_proto),
        "prototype_groups_with_gt1": len(proto_groups),
        "records_in_prototype_groups_gt1": proto_records,
        "prototype_group_size_hist": dict(sorted(
            Counter(len(v) for v in proto_groups.values()).items())),
        "structure_matcher_confirmation": {
            "n_tested": len(sample_groups), "confirmed": confirmed,
            "rejected": rejected},
    }
    with open(os.path.join(args.out, "duplicates.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

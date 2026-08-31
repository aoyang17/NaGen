#!/usr/bin/env python3
"""NaGen Phase 0: label provenance summary (field-level presence / value stats).

Streams the raw JSONL once and reports every top-level field: type, presence,
null rate, and numeric stats where applicable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fields = defaultdict(lambda: {"type": None, "present": 0, "null": 0,
                                  "non_numeric": 0, "values": []})
    families = Counter()
    n = 0
    with open(args.input) as f:
        for line in f:
            z = json.loads(line)
            n += 1
            fam = (z.get("material_id") or "?").split("_")[0]
            families[fam] += 1
            for k, v in z.items():
                st = fields[k]
                st["present"] += 1
                if v is None:
                    st["null"] += 1
                    continue
                t = type(v).__name__
                if st["type"] is None:
                    st["type"] = t
                elif st["type"] != t:
                    st["type"] += f"+{t}"
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    st["values"].append(v)
                elif isinstance(v, str):
                    st["non_numeric"] += 1
                    if "str_sample" not in st:
                        st["str_sample"] = v[:80]

    report = {"n_records": n, "material_id_families": dict(families.most_common()),
              "fields": {}}
    for k, st in sorted(fields.items()):
        vals = st.pop("values")
        if vals:
            st["n_numeric"] = len(vals)
            st["min"] = min(vals)
            st["max"] = max(vals)
            st["mean"] = statistics.fmean(vals)
            s = sorted(vals)
            st["median"] = s[len(s) // 2]
        st["present_rate"] = round(st["present"] / n, 4)
        st["null_rate"] = round(st["null"] / n, 4)
        report["fields"][k] = st

    with open(os.path.join(args.out, "label_provenance.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

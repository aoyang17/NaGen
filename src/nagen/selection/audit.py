"""CPU data audit. JSONL (raw pymatgen or NAXL), or trusted packed .pt."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from nagen.inverse.spec import (
    DEFAULT_FE_COORDINATION_OPTIONS,
    parse_fe_coordination_options,
)
from .pipeline import Crystal, Conditioning, calibrate, hard_gates


def load_crystals(path, split=None):
    path = Path(path)
    if path.suffix == ".pt":
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        names = {int(v): k for k, v in data["element_to_index"].items()}
        for i, name in enumerate(data["ids"]):
            if split is not None and int(data["split"][i]) != {"train": 0, "val": 1, "test": 2}[split]:
                continue
            a, b = int(data["offsets"][i]), int(data["offsets"][i+1])
            yield Crystal(str(name), [names[int(v)] for v in data["types"][a:b]],
                          data["frac_coords"][a:b].numpy().astype(float),
                          data["lattices"][i].numpy().astype(float))
        return
    with path.open() as handle:
        for i, line in enumerate(handle):
            r = json.loads(line)
            if split is not None:
                if "split" not in r:
                    raise ValueError("JSONL calibration needs explicit split labels or separate train file")
                if r["split"] != split:
                    continue
            if "structure" in r:
                s = r["structure"]
                if any(len(x["species"]) != 1 or x["species"][0].get("occu", 1) != 1 for x in s["sites"]):
                    raise ValueError("partial occupancy requires an explicit disorder model")
                elements = [x["species"][0]["element"] for x in s["sites"]]
                frac = [x["abc"] for x in s["sites"]]
                cell = s["lattice"]["matrix"]
            else:
                elements, frac, cell = r["A"], r["X_frac"], r["L_matrix"]
            yield Crystal(str(r.get("material_id", r.get("sample_id", i))), elements,
                          np.asarray(frac, float) % 1, np.asarray(cell, float),
                          r.get("family", "phosphate"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--split", choices=["train"])
    parser.add_argument("--candidates")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-structures", type=int, default=20)
    parser.add_argument("--fe-cutoff", type=float, default=2.5)
    parser.add_argument(
        "--fe-coordination",
        type=parse_fe_coordination_options,
        default=DEFAULT_FE_COORDINATION_OPTIONS,
        help="Allowed Fe-O coordination numbers, e.g. 4, 4,5, or 4,5,6.",
    )
    args = parser.parse_args()
    if args.min_structures < 1 or args.fe_cutoff <= 0:
        parser.error("positive calibration support and cutoff required")
    train = list(load_crystals(args.train, args.split))
    if not train:
        parser.error("training set is empty")
    profile = calibrate(train, args.min_structures, {"P": 2., "Fe": args.fe_cutoff})
    digest = hashlib.sha256()
    with open(args.train, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    profile["source_sha256"] = digest.hexdigest()
    profile["source"] = str(Path(args.train).resolve())
    profile["split"] = args.split or "caller-declared training-only file"
    candidates = list(load_crystals(args.candidates)) if args.candidates else train
    reports = [
        hard_gates(
            c,
            Conditioning(dict(Counter(c.elements)), c.family),
            profile,
            fe_coordination_options=args.fe_coordination,
        )
        for c in candidates
    ]
    summary = {
        "status": "geometry_audit_only_no_UMA_no_external_filter",
        "training_count": len(train), "candidate_count": len(candidates),
        "composition_groups": len({c.composition for c in train}),
        "calibration_resubstitution": not bool(args.candidates),
        "note": "Self-derived conditions audit geometry only, not requested design compliance.",
        "passed_counts": {name: sum(r["checks"][name] for r in reports)
                          for name in reports[0]["checks"]} if reports else {},
        "final_passed": sum(r["passed"] for r in reports),
        "fe_coordination_options": list(args.fe_coordination),
        "profile": profile, "candidates": reports,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"profile", "candidates"}}, indent=2))


if __name__ == "__main__":
    main()

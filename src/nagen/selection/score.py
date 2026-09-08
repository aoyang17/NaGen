"""Re-label structures with CHGNet, including structures failing final gates.

This is separate from selection: labeling does not certify physical validity.
Existing input energy labels are ignored. Optional reference score JSONL must
have been made with this same backend hash, never with UMA or legacy MP energy.
"""
import argparse
from itertools import islice
import json
from pathlib import Path
import time
import torch

from .audit import load_crystals
from .energy import CHGNetEvaluator
from .experiment import sha256, json_default
from .hull import ReferenceHull


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--reference-scores")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("refusing to overwrite previous labels")
    evaluator = CHGNetEvaluator(args.device)
    model_hash = evaluator.provenance["state_sha256"]
    reference, reference_ids = None, set()
    if args.reference_scores:
        with open(args.reference_scores) as handle:
            records = [json.loads(line) for line in handle]
        if any(r.get("status") != "scored" for r in records):
            raise ValueError("reference scoring failures must be resolved explicitly")
        reference = ReferenceHull(records, model_hash)
        reference_ids = {r["id"] for r in records}
    start, success, failed = time.monotonic(), 0, 0
    with out.open("x") as handle:
        for c in islice(load_crystals(args.input), args.limit):
            row = {"id": c.id, "elements": c.elements, "model_hash": model_hash}
            try:
                row.update(evaluator.evaluate(c.elements, c.frac, c.lattice))
                row["status"] = "scored"
                if reference:
                    if c.id in reference_ids:
                        raise ValueError("candidate is a member of reference set")
                    row["hull"] = reference.evaluate(c.elements, row["energy_eV_atom"])
                success += 1
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                failed += 1
            handle.write(json.dumps(row, default=json_default, allow_nan=False)+"\n")
            handle.flush()
    manifest = {"configuration": vars(args), "input_sha256": sha256(args.input),
                "model": evaluator.provenance, "scored": success, "failed": failed,
                "elapsed_seconds": time.monotonic()-start, "potential_calls": evaluator.calls}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

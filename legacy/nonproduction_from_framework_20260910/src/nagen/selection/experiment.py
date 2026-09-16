"""Executable four-stage experiment using CHGNet; no HF account required.

Inputs are explicit training/candidate/reference files. Existing dataset energy
labels are ignored. Supply optional fixed conditioning for design compliance.
Without it, input candidate compositions are treated as their design specs.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .audit import load_crystals
from .pipeline import Conditioning, calibrate, hard_gates, robust_rank, diversity_select
from .energy import CHGNetEvaluator
from .hull import ReferenceHull
from .diversity import descriptor, distances, DESCRIPTOR_VERSION


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def target_crystals(path, split=None):
    rows = list(load_crystals(path, split))
    selected = [c for c in rows if {"Fe", "P", "O"}.issubset(c.elements)
                and set(c.elements).issubset({"Na", "Fe", "P", "O"})]
    if len({c.id for c in selected}) != len(selected):
        raise ValueError("duplicate IDs: resolve before running an experiment")
    return selected, len(rows)-len(selected)


def run(args):
    import torch
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists() or (output / "evaluations.jsonl").exists():
        raise FileExistsError("use a fresh output directory to preserve previous experiments")
    train, excluded = target_crystals(args.train, args.train_split)
    candidates, excluded_candidates = target_crystals(args.candidates)
    if not train or not candidates:
        raise ValueError("training and candidate Na/Fe/P/O subsets must be nonempty")
    if {c.id for c in train} & {c.id for c in candidates}:
        raise ValueError("candidate IDs overlap training IDs; supply held-out/generated candidates")
    profile = calibrate(train, args.min_structures)
    profile["training_source_sha256"] = sha256(args.train)
    (output / "profile.json").write_text(json.dumps(profile, indent=2))
    fixed = Conditioning(**json.loads(Path(args.condition).read_text())) if args.condition else None
    evaluator = CHGNetEvaluator(args.device)
    hull, reference_report = None, []
    if args.references:
        references = list(load_crystals(args.references))
        if {c.id for c in references} & {c.id for c in candidates}:
            raise ValueError("freeze reference set independently of candidates")
        for c in references:
            pred = evaluator.evaluate(c.elements, c.frac, c.lattice)
            reference_report.append({"id": c.id, "elements": c.elements,
                "energy_eV_atom": pred["energy_eV_atom"], "force_max_eV_A": pred["force_max_eV_A"],
                "model_hash": evaluator.provenance["state_sha256"]})
        hull = ReferenceHull(reference_report, evaluator.provenance["state_sha256"])
        (output / "reference_scores.json").write_text(json.dumps(reference_report, indent=2))
    if args.reference_threshold is not None and hull is None:
        raise ValueError("reference threshold requires reference structures")
    gradient = evaluator.gradient_check(train[0]) if args.check_gradient else None
    if gradient and not gradient["passed"]:
        (output / "gradient_check.json").write_text(json.dumps(gradient, indent=2))
        raise RuntimeError("CHGNet VJP finite-difference check failed; do not optimize")
    reports = []
    with (output / "evaluations.jsonl").open("w") as log:
        for i, c in enumerate(candidates):
            condition = fixed or Conditioning(dict(Counter(c.elements)), c.family)
            gate = hard_gates(c, condition, profile, motif_backend="native",
                              force_threshold=args.force_threshold, fe_valences=tuple(args.fe_valences))
            # Cheap gates first: don't send overlaps and invalid motifs to the potential.
            geometry_passed = all(v for k, v in gate["checks"].items() if k != "final_force")
            prediction = None
            if geometry_passed:
                try:
                    prediction = evaluator.evaluate(c.elements, c.frac, c.lattice)
                    gate = hard_gates(c, condition, profile, prediction["forces_eV_A"],
                        args.force_threshold, motif_backend="native", fe_valences=tuple(args.fe_valences))
                    gate["metrics"]["energy_eV_atom"] = prediction["energy_eV_atom"]
                    if hull:
                        gate["hull"] = hull.evaluate(c.elements, prediction["energy_eV_atom"])
                except Exception as error:
                    gate["passed"] = False
                    gate["evaluation_error"] = f"{type(error).__name__}: {error}"
            eligible = gate["passed"]
            if args.reference_threshold is not None:
                estimate = gate.get("hull", {}).get("e_above_reference_hull_eV_atom")
                eligible = eligible and estimate is not None and estimate <= args.reference_threshold
            gate.update(group=repr((c.family, c.composition)), geometry_passed=geometry_passed,
                        quality_screen_passed=bool(eligible), prediction=prediction)
            reports.append(gate)
            log.write(json.dumps(gate, default=json_default, allow_nan=False) + "\n")
            log.flush()
            print(f"[{i+1}/{len(candidates)}] {c.id}: hard_gates={gate['passed']}", flush=True)
    quality_rows = [{**r, "passed": r["quality_screen_passed"]} for r in reports]
    metrics = ["energy_eV_atom"] + [f"{e}_{m}" for e in ("P", "Fe")
        for m in ("off_center", "bond_distortion", "angle_distortion", "coordination_rarity")]
    groups = robust_rank(quality_rows, metrics)
    selected = []
    selection_report = {}
    crystal_map = {c.id: c for c in candidates}
    train_desc = np.stack([descriptor(c) for c in train]) if groups else None
    for group, ranked in groups.items():
        pool = ranked[:args.pool_factor*args.k]
        features = np.stack([descriptor(crystal_map[r["id"]]) for r in pool])
        nearest = distances(features, train_desc).min(axis=1)
        for row, d in zip(pool, nearest):
            row["nearest_training_distance"] = float(d)
        chosen = diversity_select(pool, distances(features, features), nearest,
                                  args.k, len(pool), args.duplicate_threshold)
        selected.extend(chosen)
        selection_report[group] = {"ranked_ids": [r["id"] for r in ranked],
                                   "selected_ids": [r["id"] for r in chosen]}
    summary = {
        "status": "completed", "backend": evaluator.provenance,
        "configuration": vars(args), "training_count": len(train), "candidate_count": len(candidates),
        "excluded_non_target_training": excluded, "excluded_non_target_candidates": excluded_candidates,
        "source_sha256": {"train": sha256(args.train), "candidates": sha256(args.candidates),
                          "references": sha256(args.references) if args.references else None},
        "geometry_passed": sum(r["geometry_passed"] for r in reports),
        "hard_gate_passed": sum(r["passed"] for r in reports),
        "selected_count": len(selected), "selected_ids": [r["id"] for r in selected],
        "selection": selection_report, "descriptor": DESCRIPTOR_VERSION,
        "hull_scope": "finite_reference_set" if hull else "not_evaluated",
        "gradient_check": gradient, "potential_calls": evaluator.calls,
        "elapsed_seconds": time.monotonic()-started,
        "limitations": ["No complete thermodynamic phase diagram is asserted.",
                        "Radial fingerprint may collide for different structures.",
                        "Native motif implementation; legacy external filter not invoked.",
                        "Caller must supply leakage-free structure-family splits."],
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, default=json_default, allow_nan=False))
    with (output / "selected.jsonl").open("w") as handle:
        for r in selected:
            c = crystal_map[r["id"]]
            handle.write(json.dumps({"material_id": c.id, "A": c.elements,
                "X_frac": c.frac, "L_matrix": c.lattice, "family": c.family}, default=json_default)+"\n")
    print(json.dumps({k: summary[k] for k in ("hard_gate_passed", "selected_count", "potential_calls", "elapsed_seconds")}))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--train-split", choices=["train"])
    p.add_argument("--candidates", required=True)
    p.add_argument("--references", help="Independent frozen reference structure JSONL; all energies recomputed")
    p.add_argument("--condition", help='JSON {"counts": {"Na":...,"Fe":...,"P":...,"O":...}, "family":"phosphate"}')
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--min-structures", type=int, default=20)
    p.add_argument("--force-threshold", type=float, default=.05)
    p.add_argument("--fe-valences", type=int, nargs="+", default=[2,3])
    p.add_argument("--reference-threshold", type=float)
    p.add_argument("--k", type=int, default=8, help="Selection budget per composition/family group")
    p.add_argument("--pool-factor", type=int, default=4)
    p.add_argument("--duplicate-threshold", type=float, default=.02)
    p.add_argument("--check-gradient", action="store_true")
    args = p.parse_args()
    if min(args.k, args.pool_factor, args.threads, args.min_structures) < 1:
        p.error("budgets must be positive")
    run(args)


if __name__ == "__main__":
    main()

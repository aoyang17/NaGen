"""Measure batch throughput and agreement with historical default UMA labels."""
import argparse
import json
from pathlib import Path
import sys
import time

from ase import Atoms
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nagen.selection.audit import load_crystals
from nagen.selection.uma import UMAEvaluator
from run_uma_campaign import inputs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch_fp32", "batch_tf32"), default="batch_fp32")
    args = parser.parse_args()
    torch.set_num_threads(2)
    samples = list(load_crystals(ROOT / "runs/framework_uma_dflow/relax/relaxed.jsonl"))[:8]
    labels = {r["id"]: r["after"] for r in map(json.loads, (ROOT / "runs/framework_uma_dflow/relax/pairs.jsonl").read_text().splitlines())}
    atoms = [Atoms(symbols=c.elements, scaled_positions=c.frac, cell=c.lattice, pbc=True) for c in samples]
    evaluator = UMAEvaluator(str(inputs()["uma"]), "cuda", "omat", inference_settings=args.mode)
    report = {"model": evaluator.provenance, "measurements": {}, "passed": True}
    for count in (1, 2, 4, 8):
        evaluator.evaluate_batch(atoms[:count])
        started = time.perf_counter()
        for _ in range(3):
            result = evaluator.evaluate_batch(atoms[:count])
        seconds = (time.perf_counter() - started) / 3
        errors = {key: max(float(np.max(np.abs(np.asarray(row[key]) - np.asarray(labels[c.id][key]))))
                          for c, row in zip(samples[:count], result))
                  for key in ("energy_eV_atom", "forces_eV_A", "stress_eV_A3")}
        # TF32 is only a proposal/initial-relaxation path; all finalists require
        # default FP32 predictions and strict refinement before acceptance.
        energy_tolerance, force_tolerance, stress_tolerance = (1e-3, 1e-2, 1e-3) if args.mode == "batch_tf32" else (1e-5, 1e-4, 1e-5)
        passed = errors["energy_eV_atom"] < energy_tolerance and errors["forces_eV_A"] < force_tolerance and errors["stress_eV_A3"] < stress_tolerance
        report["passed"] &= passed
        report["measurements"][str(count)] = {"seconds_per_batch": seconds, "seconds_per_structure": seconds / count,
                                             "maximum_absolute_differences": errors, "passed": passed}
        print(json.dumps({"batch": count, **report["measurements"][str(count)]}), flush=True)
    name = "uma_batch_tf32_benchmark.json" if args.mode == "batch_tf32" else "uma_batch_benchmark.json"
    write_json(ROOT / "runs" / name, report)
    assert report["passed"], report


if __name__ == "__main__":
    main()

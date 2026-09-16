"""Compare same-checkpoint full-precision merged inference to default E/F/S."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nagen.selection.audit import load_crystals
from nagen.selection.uma import UMAEvaluator
from run_uma_campaign import inputs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed_composition_fp32", "fixed_composition_fp32_compiled"), default="fixed_composition_fp32_compiled")
    args = parser.parse_args()
    torch.set_num_threads(2)
    samples = list(load_crystals(ROOT / "runs/framework_uma_dflow/relax/relaxed.jsonl"))[:4]
    samples.append(next(load_crystals(ROOT / "runs/final_refine_relax/relaxed.jsonl")))
    labels, reports = {}, {}
    for mode in ("default", args.mode):
        started = time.perf_counter()
        evaluator = UMAEvaluator(str(inputs()["uma"]), "cuda", "omat", inference_settings=mode)
        evaluator.evaluate(samples[0].elements, samples[0].frac, samples[0].lattice)
        startup = time.perf_counter() - started
        started = time.perf_counter()
        outputs = [evaluator.evaluate(c.elements, c.frac, c.lattice) for _ in range(3) for c in samples]
        elapsed = time.perf_counter() - started
        labels[mode] = outputs
        reports[mode] = {"startup_seconds": startup, "seconds_per_prediction": elapsed / len(outputs), "model": evaluator.provenance}
        print(json.dumps({"mode": mode, **reports[mode]}), flush=True)
        del evaluator
        torch.cuda.empty_cache()
    mode = args.mode
    delta = {key: max(float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key])))) for a, b in zip(labels["default"], labels[mode]))
             for key in ("energy_eV_atom", "forces_eV_A", "stress_eV_A3")}
    report = {"measurements": reports, "maximum_absolute_differences": delta,
              "speedup": reports["default"]["seconds_per_prediction"] / reports[mode]["seconds_per_prediction"],
              "passed": delta["energy_eV_atom"] < 1e-5 and delta["forces_eV_A"] < 1e-4 and delta["stress_eV_A3"] < 1e-5,
              "note": "Concurrent campaign GPU load affects timings. Finalists must still be verified using default inference."}
    name = "uma_compiled_inference_benchmark.json" if mode.endswith("_compiled") else "uma_inference_benchmark.json"
    write_json(ROOT / "runs" / name, report)
    print(json.dumps(report, indent=2), flush=True)
    assert report["passed"], report


if __name__ == "__main__":
    main()

"""Check optimized geometry/hull equivalence and current UMA runtime replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nagen.selection.audit import load_crystals
from nagen.selection.hull import ReferenceHull
from nagen.selection.pipeline import neighbors, hard_gates, Conditioning
from run_uma_campaign import inputs, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    files = inputs()
    final = next(load_crystals(ROOT / "runs/final_refine_relax/relaxed.jsonl"))
    old_source = subprocess.check_output(["git", "show", "HEAD:src/nagen/selection/pipeline.py"], cwd=ROOT, text=True)
    namespace = {"__name__": "baseline_pipeline"}
    exec(compile(old_source, "baseline_pipeline.py", "exec"), namespace)
    old_neighbors = namespace["neighbors"]
    timings = {}
    for name, fn in (("old", old_neighbors), ("optimized", neighbors)):
        tick = time.perf_counter()
        for _ in range(3):
            for index in range(len(final.elements)):
                fn(final, index, 5.)
        timings[name] = time.perf_counter() - tick
    for index in range(len(final.elements)):
        old, new = old_neighbors(final, index, 5.), neighbors(final, index, 5.)
        assert len(old) == len(new)
        assert [n[0] for n in old] == [n[0] for n in new]
        np.testing.assert_allclose([n[1] for n in old], [n[1] for n in new], atol=1e-12, rtol=0)
    report = {"neighbors": {"seconds": timings, "speedup": timings["old"] / timings["optimized"], "equivalent": True}}
    manifest = json.loads((ROOT / "runs/final_selected/manifest.json").read_text())
    labels = [json.loads(line) for line in files["references"].read_text().splitlines()]
    hull = ReferenceHull(labels, manifest["uma_model"]["state_sha256"])
    energy = manifest["strict_relaxation"]["energy_eV_atom"]
    first = hull.evaluate(final.elements, energy)
    np.testing.assert_allclose(first["e_above_reference_hull_eV_atom"], manifest["reference_hull"]["e_above_reference_hull_eV_atom"], atol=1e-12)
    tick = time.perf_counter()
    for index in range(100):
        result = hull.evaluate(final.elements, energy + index * .001)
        np.testing.assert_allclose(result["delta_e_reference_eV_atom"], first["delta_e_reference_eV_atom"] + index * .001, atol=1e-12)
    report["hull"] = {"equivalent": True, "cached_100_seconds": time.perf_counter() - tick, "cached_compositions": len(hull._solutions)}
    if args.gpu:
        import torch
        from nagen.selection.uma import UMAEvaluator
        torch.set_num_threads(2)
        tick = time.perf_counter()
        evaluator = UMAEvaluator(str(files["uma"]), "cuda", "omat")
        current = evaluator.evaluate(final.elements, final.frac, final.lattice)
        report["uma"] = {"model": evaluator.provenance, "startup_seconds": time.perf_counter() - tick,
                         "torch": torch.__version__, "energy_eV_atom": current["energy_eV_atom"],
                         "force_max_eV_A": current["force_max_eV_A"],
                         "energy_difference": current["energy_eV_atom"] - energy,
                         "force_difference": current["force_max_eV_A"] - manifest["strict_relaxation"]["force_max_eV_A"]}
        assert abs(report["uma"]["energy_difference"]) <= 1e-5, report["uma"]
        assert abs(report["uma"]["force_difference"]) <= 1e-4, report["uma"]
        profile = json.loads(files["profile"].read_text())
        gates = hard_gates(final, Conditioning(dict(Counter(final.elements))), profile,
                           forces=current["forces_eV_A"], force_threshold=.03, motif_backend="native")
        assert gates["passed"], gates
        assert evaluator.provenance["state_sha256"] == manifest["uma_model"]["state_sha256"]
        report["uma"]["strict_gates_passed"] = True
    write_json(args.out, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

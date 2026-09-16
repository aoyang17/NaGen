"""Relax and UMA-score one deterministic shard of a frozen reference JSONL.

All successful records carry the UMA checkpoint hash used by ReferenceHull.
Failures are retained in a sidecar file and are never silently substituted with
historical dataset energies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from nagen.selection.audit import load_crystals
from nagen.selection.experiment import json_default, sha256
from nagen.selection.uma import UMAEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--uma-checkpoint", required=True); parser.add_argument("--task-name", default="omat")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--fmax", type=float, default=.03); parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.steps < 1 or args.fmax <= 0:
        parser.error("invalid shard or relaxation protocol")
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    failures = out.with_suffix(out.suffix + ".failures.jsonl")
    evaluator = UMAEvaluator(args.uma_checkpoint, args.device, args.task_name)
    selected = 0; completed = 0; failed = 0
    with out.open("x") as handle, failures.open("x") as failure_handle:
        for index, crystal in enumerate(load_crystals(args.input)):
            if index % args.shard_count != args.shard_index: continue
            selected += 1
            try:
                atoms = Atoms(symbols=crystal.elements, scaled_positions=crystal.frac,
                              cell=crystal.lattice, pbc=True)
                atoms.calc = evaluator.calculator
                optimizer = FIRE(atoms, logfile=None, maxstep=.1)
                optimizer.run(fmax=args.fmax, steps=args.steps)
                frac = atoms.get_scaled_positions() % 1.0
                cell = np.asarray(atoms.cell)
                result = evaluator.evaluate(crystal.elements, frac, cell)
                record = {"id": crystal.id, "elements": crystal.elements,
                          "energy_eV_atom": result["energy_eV_atom"],
                          "model_hash": evaluator.provenance["state_sha256"],
                          "force_max_eV_A": result["force_max_eV_A"],
                          "converged": result["force_max_eV_A"] <= args.fmax,
                          "steps": optimizer.nsteps, "X_frac": frac, "L_matrix": cell}
                handle.write(json.dumps(record, default=json_default, allow_nan=False) + "\n"); handle.flush()
                completed += 1
            except Exception as error:
                failure_handle.write(json.dumps({"id": crystal.id, "error": f"{type(error).__name__}: {error}"}) + "\n"); failure_handle.flush()
                failed += 1
            if selected % 10 == 0:
                print(json.dumps({"shard": args.shard_index, "selected": selected, "completed": completed, "failed": failed}), flush=True)
    manifest = {"status": "completed" if not failed else "completed_with_failures", "configuration": vars(args),
                "input_sha256": sha256(args.input), "model": evaluator.provenance,
                "selected": selected, "completed": completed, "failed": failed,
                "protocol": "fixed_cell_FIRE; reference labels are valid only with candidates relaxed identically"}
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Fixed-cell UMA relaxation and post-relax hard-gate audit."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from nagen.selection.audit import load_crystals
from nagen.selection.experiment import json_default, sha256
from nagen.selection.pipeline import Crystal, Conditioning, hard_gates
from nagen.selection.uma import UMAEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--fmax", type=float, default=.03)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.fmax <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--steps and --fmax must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    evaluator = UMAEvaluator(args.uma_checkpoint, args.device, args.task_name)
    profile = json.loads(Path(args.profile).read_text())
    results = []
    start = time.monotonic()
    with (out / "pairs.jsonl").open("x") as pair_handle, (out / "relaxed.jsonl").open("x") as structure_handle:
        for input_index, crystal in enumerate(load_crystals(args.input)):
            if input_index % args.shard_count != args.shard_index:
                continue
            row = {"id": crystal.id, "status": "failed"}
            try:
                condition = Conditioning(dict(Counter(crystal.elements)), crystal.family)
                before = evaluator.evaluate(crystal.elements, crystal.frac, crystal.lattice)
                atoms = Atoms(symbols=crystal.elements, scaled_positions=crystal.frac,
                              cell=crystal.lattice, pbc=True)
                atoms.calc = evaluator.calculator
                optimizer = FIRE(atoms, logfile=None, maxstep=.1)
                optimizer.run(fmax=args.fmax, steps=args.steps)
                relaxed = Crystal(crystal.id, crystal.elements, atoms.get_scaled_positions(),
                                  np.asarray(atoms.cell), crystal.family)
                after = evaluator.evaluate(relaxed.elements, relaxed.frac, relaxed.lattice)
                gates = {
                    phase: hard_gates(value, condition, profile, forces=label["forces_eV_A"], motif_backend="native")
                    for phase, value, label in (("before", crystal, before), ("after", relaxed, after))
                }
                row.update(
                    status="completed", steps=optimizer.nsteps,
                    converged=after["force_max_eV_A"] <= args.fmax,
                    before=before, after=after, gates=gates,
                    delta_energy_eV_atom=after["energy_eV_atom"] - before["energy_eV_atom"],
                )
                structure_handle.write(json.dumps({"material_id": crystal.id, "A": crystal.elements,
                    "X_frac": relaxed.frac, "L_matrix": relaxed.lattice, "family": crystal.family},
                    default=json_default, allow_nan=False) + "\n")
                structure_handle.flush()
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            results.append(row)
            pair_handle.write(json.dumps(row, default=json_default, allow_nan=False) + "\n")
            pair_handle.flush()
            print(json.dumps({"id": crystal.id, "status": row["status"], "steps": row.get("steps")}), flush=True)
    complete = [row for row in results if row["status"] == "completed"]
    summary = {
        "status": "completed" if len(complete) == len(results) else "completed_with_failures",
        "configuration": vars(args), "input_sha256": sha256(args.input),
        "profile_sha256": sha256(args.profile), "model": evaluator.provenance,
        "completed": len(complete), "failed": len(results) - len(complete),
        "converged": sum(row["converged"] for row in complete),
        "elapsed_seconds": time.monotonic() - start,
        "protocol": "fixed_cell_FIRE; candidate and reference calculations must use this exact protocol",
        "all_gates_passed": {phase: sum(row["gates"][phase]["passed"] for row in complete)
                             for phase in ("before", "after")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

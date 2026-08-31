"""Energy-aware constrained inference using CHGNet plus exact-task barriers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.optimize import FIRE
from chgnet.model.model import CHGNet
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .campaign import analytical_feasible
from .constraints import evaluate_feasibility
from .energy_guidance import ConstraintBiasedCHGNetCalculator
from .relax_chgnet import wrapped_structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--fmax", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--distance-k", type=float, default=120.0)
    parser.add_argument("--coordination-k", type=float, default=80.0)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--fixed-topology", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.input) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    selected = [
        (index, record)
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard
    ]
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = output_dir / f"cif_shard_{args.shard:02d}"
    cif_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"constrained_shard_{args.shard:02d}.jsonl"
    summary_path = output_dir / f"summary_shard_{args.shard:02d}.json"

    model = CHGNet.load()
    calculator = ConstraintBiasedCHGNetCalculator(
        model,
        args.device,
        distance_k_eV_A2=args.distance_k,
        coordination_k_eV_A2=args.coordination_k,
        margin_A=args.margin,
        fixed_topology=args.fixed_topology,
    )
    completed = 0
    errors = 0
    converged_count = 0
    analytic_count = 0
    with output_path.open("w") as output:
        for source_index, record in selected:
            try:
                initial = wrapped_structure(Structure.from_dict(record["structure"]))
                initial_prediction = model.predict_structure(initial)
                atoms = AseAtomsAdaptor.get_atoms(initial)
                if args.fixed_topology:
                    calculator.reset_topology(atoms)
                atoms.calc = calculator
                optimizer = FIRE(atoms, logfile=None)
                converged = bool(
                    optimizer.run(fmax=args.fmax, steps=args.steps)
                )
                biased_forces = atoms.get_forces()
                final = wrapped_structure(AseAtomsAdaptor.get_structure(atoms))
                final_prediction = model.predict_structure(final)
                physical_max_force = float(
                    np.linalg.norm(final_prediction["f"], axis=1).max()
                )
                biased_max_force = float(
                    np.linalg.norm(biased_forces, axis=1).max()
                )
                feasibility = evaluate_feasibility(
                    [str(specie) for specie in final.species],
                    final.frac_coords,
                    final.lattice.matrix,
                    None,
                )
                analytic_ok = analytical_feasible(feasibility)
                initial_energy = float(initial_prediction["e"])
                final_energy = float(final_prediction["e"])
                record["pre_energy_guidance_structure"] = record["structure"]
                record["structure"] = final.as_dict()
                record["feasibility_after_energy_guidance"] = feasibility
                record["analytical_feasible_after_energy_guidance"] = analytic_ok
                record["thermodynamic_stability_score"] = {
                    "method": "CHGNet-0.3.0+analytic-constraint-barrier",
                    "meaning": "energy-aware constrained local optimization",
                    "initial_physical_energy_eV_atom": initial_energy,
                    "final_physical_energy_eV_atom": final_energy,
                    "physical_energy_change_eV_atom": final_energy
                    - initial_energy,
                    "final_physical_max_force_eV_A": physical_max_force,
                    "final_biased_max_force_eV_A": biased_max_force,
                    "constraint_barrier_energy_eV_cell": float(
                        calculator.results["constraint_barrier_energy"]
                    ),
                    "constrained_optimizer_converged": converged,
                    "delta_e_hull_eV_atom": None,
                    "hull_status": "not_evaluated_missing_consistent_competing_phase_energies",
                }
                record["energy_guidance_provenance"] = {
                    "optimizer": "ASE-FIRE",
                    "steps_taken": int(optimizer.nsteps),
                    "steps_allowed": args.steps,
                    "fmax_eV_A": args.fmax,
                    "cell_fixed": True,
                    "distance_k_eV_A2": args.distance_k,
                    "coordination_k_eV_A2": args.coordination_k,
                    "constraint_margin_A": args.margin,
                    "coordination_topology": (
                        "fixed_capacity_aware_graph_with_TM_CN_in_4_5_6"
                        if args.fixed_topology
                        else "dynamic_nearest_oxygen_P4_TM6"
                    ),
                    "source_index": source_index,
                }
                final.to(filename=str(cif_dir / f"candidate_{source_index + 1:04d}.cif"))
                completed += 1
                converged_count += int(converged)
                analytic_count += int(analytic_ok)
            except Exception as error:
                record["energy_guidance_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                errors += 1
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        "shard": args.shard,
                        "processed": completed + errors,
                        "completed": completed,
                        "converged": converged_count,
                        "analytic": analytic_count,
                        "errors": errors,
                    }
                ),
                flush=True,
            )

    summary = {
        "shard": args.shard,
        "assigned": len(selected),
        "completed": completed,
        "errors": errors,
        "constrained_optimizer_converged": converged_count,
        "analytically_feasible_after_energy_guidance": analytic_count,
        "output": str(output_path),
        "cif_directory": str(cif_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

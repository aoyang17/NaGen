"""Independent CHGNet relaxation audit for generated candidates."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from chgnet.model.dynamics import StructOptimizer
from chgnet.model.model import CHGNet
from ase.constraints import FixBondLengths
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .campaign import analytical_feasible
from .constraints import bond_cutoff, evaluate_feasibility, pbc_distance_matrix
from .spec import DEFAULT_SPEC

warnings.filterwarnings("ignore", message="logm result may be inaccurate")


def wrapped_structure(structure: Structure) -> Structure:
    return Structure(
        structure.lattice,
        structure.species,
        np.remainder(structure.frac_coords, 1.0),
        coords_are_cartesian=False,
        site_properties=structure.site_properties,
    )


def coordination_constrained_atoms(structure: Structure):
    """Legacy ablation: ASE atoms with generated P-O/Fe-O bonds fixed."""
    elements = [str(specie) for specie in structure.species]
    distances = pbc_distance_matrix(
        structure.frac_coords, structure.lattice.matrix
    )
    pairs: list[tuple[int, int]] = []
    for i, element in enumerate(elements):
        if element not in {"P", "Fe"}:
            continue
        for j, other in enumerate(elements):
            if other == "O" and distances[i, j] <= bond_cutoff(element, other):
                pairs.append((i, j))
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_constraint(FixBondLengths(pairs))
    return atoms, pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--fmax", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--relax-cell", action="store_true")
    parser.add_argument("--fix-coordination-bonds", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.input) as handle:
        all_records = [json.loads(line) for line in handle if line.strip()]
    selected = [
        (index, record)
        for index, record in enumerate(all_records)
        if index % args.num_shards == args.shard
    ]
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = output_dir / f"cif_shard_{args.shard:02d}"
    cif_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"relaxed_shard_{args.shard:02d}.jsonl"
    summary_path = output_dir / f"summary_shard_{args.shard:02d}.json"

    model = CHGNet.load()
    optimizer = StructOptimizer(model=model, use_device=args.device)
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    completed = 0
    converged_count = 0
    analytic_survivors = 0
    errors = 0
    with output_path.open("w") as output:
        for source_index, record in selected:
            initial = wrapped_structure(Structure.from_dict(record["structure"]))
            try:
                initial_prediction = model.predict_structure(initial)
                relaxation_input = initial
                fixed_bond_count = 0
                if args.fix_coordination_bonds:
                    relaxation_input, fixed_pairs = coordination_constrained_atoms(
                        initial
                    )
                    fixed_bond_count = len(fixed_pairs)
                relaxation = optimizer.relax(
                    relaxation_input,
                    fmax=args.fmax,
                    steps=args.steps,
                    relax_cell=args.relax_cell,
                    verbose=False,
                )
                final = wrapped_structure(relaxation["final_structure"])
                final_prediction = model.predict_structure(final)
                max_force = float(
                    np.linalg.norm(final_prediction["f"], axis=1).max()
                )
                converged = max_force <= args.fmax * 1.05
                feasibility = evaluate_feasibility(
                    [str(specie) for specie in final.species],
                    final.frac_coords,
                    final.lattice.matrix,
                    None,
                )
                analytic_ok = analytical_feasible(feasibility)
                try:
                    same_basin = bool(matcher.fit(initial, final))
                except Exception:
                    same_basin = False
                energy_initial = float(initial_prediction["e"])
                energy_final = float(final_prediction["e"])
                record["pre_relaxation_structure"] = record["structure"]
                if record.get("thermodynamic_stability_score") is not None:
                    record["pre_relaxation_thermodynamic_stability_score"] = record[
                        "thermodynamic_stability_score"
                    ]
                record["structure"] = final.as_dict()
                record["feasibility_after_relaxation"] = feasibility
                record["analytical_feasible_after_relaxation"] = analytic_ok
                record["thermodynamic_stability_score"] = {
                    "method": "CHGNet-0.3.0",
                    "meaning": "independent MLFF relaxation/single-point screening",
                    "initial_energy_eV_atom": energy_initial,
                    "relaxed_energy_eV_atom": energy_final,
                    "relaxation_delta_energy_eV_atom": energy_final
                    - energy_initial,
                    "final_max_force_eV_A": max_force,
                    "converged_at_fmax": converged,
                    "volume_ratio_final_over_initial": final.volume
                    / initial.volume,
                    "same_structure_match_basin": same_basin,
                    "delta_e_hull_eV_atom": None,
                    "hull_status": "not_evaluated_missing_consistent_competing_phase_energies",
                }
                record["relaxation_provenance"] = {
                    "steps_allowed": args.steps,
                    "trajectory_steps": len(relaxation["trajectory"].energies),
                    "fmax_eV_A": args.fmax,
                    "relax_cell": args.relax_cell,
                    "fixed_coordination_bonds": args.fix_coordination_bonds,
                    "fixed_bond_count": fixed_bond_count,
                    "device": args.device,
                    "source_index": source_index,
                }
                final.to(filename=str(cif_dir / f"candidate_{source_index + 1:04d}.cif"))
                completed += 1
                converged_count += int(converged)
                analytic_survivors += int(analytic_ok)
            except Exception as error:
                errors += 1
                record["relaxation_error"] = f"{type(error).__name__}: {error}"
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        "shard": args.shard,
                        "processed": completed + errors,
                        "completed": completed,
                        "converged": converged_count,
                        "analytic_survivors": analytic_survivors,
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
        "converged": converged_count,
        "analytically_feasible_after_relaxation": analytic_survivors,
        "output": str(output_path),
        "cif_directory": str(cif_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

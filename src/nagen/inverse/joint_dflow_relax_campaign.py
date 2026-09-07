"""Reference-protocol UMA relaxation and final MP2020 hull for D-Flow v2."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ._io import atomic_json, sha256_file, sha256_json
from ._uma import load_calculator
from .calibrated_mp_hull import CalibratedMPHullCache, build_calibrated_cache
from .constraints import evaluate_feasibility
from .relax_uma import RelaxationConfig, relax_structure
from .uma_guidance import checkpoint_sha256


TERMINAL_STATUSES = {
    "accepted", "not_converged", "energy_increased", "constraints_violated"
}


def final_acceptance(
    converged: bool,
    initial_energy_eV_atom: float,
    final_energy_eV_atom: float,
    feasibility: dict[str, Any],
    energy_increase_tolerance_eV_atom: float,
) -> dict[str, Any]:
    """Strict post-relaxation gate for a D-Flow candidate."""
    energy_nonincreasing = bool(
        final_energy_eV_atom
        <= initial_energy_eV_atom + energy_increase_tolerance_eV_atom
    )
    hull_satisfied = bool(
        feasibility.get("checks", {}).get("hull_threshold", False)
    )
    all_constraints_satisfied = bool(feasibility.get("feasible", False))
    accepted = bool(
        converged and energy_nonincreasing and all_constraints_satisfied
    )
    if accepted:
        status = "accepted"
    elif not converged:
        status = "not_converged"
    elif not energy_nonincreasing:
        status = "energy_increased"
    else:
        status = "constraints_violated"
    return {
        "accepted": accepted,
        "status": status,
        "relaxation_converged": bool(converged),
        "energy_nonincreasing": energy_nonincreasing,
        "energy_change_eV_atom": (
            final_energy_eV_atom - initial_energy_eV_atom
        ),
        "energy_increase_tolerance_eV_atom": (
            energy_increase_tolerance_eV_atom
        ),
        "final_hull_satisfied": hull_satisfied,
        "all_final_constraints_satisfied": all_constraints_satisfied,
    }


def _structure(atoms: Any) -> dict[str, Any]:
    return {
        "elements": atoms.get_chemical_symbols(),
        "frac_coords": np.asarray(
            atoms.get_scaled_positions(wrap=True), dtype=float
        ).tolist(),
        "lattice": np.asarray(atoms.cell.array, dtype=float).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
    }


def run(args: argparse.Namespace) -> None:
    source_root = Path(args.source_campaign)
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    if source_manifest.get("status") not in {"complete", "incomplete"}:
        raise RuntimeError("source D-Flow campaign has no terminal manifest")
    source_records = [
        json.loads(path.read_text())
        for path in sorted((source_root / "records").glob("sample_*.json"))
    ]
    source_records = [row for row in source_records if row.get("status") == "complete"]
    phase_cache = source_manifest["e_hull"]["phase_cache"]
    root = Path(args.out_directory)
    root.mkdir(parents=True, exist_ok=True)
    calibrated_path = root / "calibrated_mp2020_hull_cache.json"
    atomic_json(
        calibrated_path,
        build_calibrated_cache(
            phase_cache, [row["counts"] for row in source_records]
        ),
        sort_keys=True,
    )
    calibrated = CalibratedMPHullCache(calibrated_path)
    config = RelaxationConfig(
        fire_fmax_eV_A=args.fire_fmax,
        fire_steps=args.fire_steps,
        bfgs_fmax_eV_A=args.bfgs_fmax,
        bfgs_steps=args.bfgs_steps,
    )
    checkpoint_hash = checkpoint_sha256(args.uma_checkpoint)
    identity = {
        "version": "joint-dflow-reference-relax-campaign-v1",
        "source_campaign": str(source_root.resolve()),
        "source_campaign_status": source_manifest.get("status"),
        "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "calibrated_hull_cache_sha256": sha256_file(calibrated_path),
        "uma_checkpoint": str(Path(args.uma_checkpoint).resolve()),
        "uma_checkpoint_sha256": checkpoint_hash,
        "task_name": args.task_name,
        "protocol": {
            **config.__dict__,
            "energy_increase_tolerance_eV_atom": (
                args.energy_increase_tolerance
            ),
            "fire_variables": "cell_and_positions_via_ExpCellFilter",
            "bfgs_variables": "positions_only",
            "final_hull": (
                "MP2020-corrected_UMA_candidate_against_cached_MP_phase_diagram"
            ),
        },
    }
    identity_hash = sha256_json(identity)
    records_root = root / "records"
    records_root.mkdir(parents=True, exist_ok=True)
    manifest = identity | {
        "identity_sha256": identity_hash,
        "status": "running",
        "source_record_count": len(source_records),
    }
    atomic_json(root / "manifest.json", manifest)
    calculator = load_calculator(args.uma_checkpoint, args.device, args.task_name)
    counts = {
        "processed": 0, "accepted": 0, "converged": 0,
        "feasible": 0, "failed": 0,
    }
    for source in source_records:
        sample_id = int(source["sample_id"])
        target = records_root / f"sample_{sample_id:06d}.json"
        if target.exists() and args.resume:
            previous = json.loads(target.read_text())
            if previous.get("identity_sha256") != identity_hash:
                raise RuntimeError("relaxation resume identity mismatch")
            if previous.get("status") in TERMINAL_STATUSES:
                counts["processed"] += 1
                counts["accepted"] += int(previous.get("accepted", False))
                counts["converged"] += int(previous.get("converged", False))
                counts["feasible"] += int(
                    previous.get("final_feasibility", {}).get("feasible", False)
                )
                continue
        elif target.exists():
            raise FileExistsError(target)
        structure = source["optimized"]["exact"]["structure"]
        record: dict[str, Any] = {
            "version": "joint-dflow-reference-relax-record-v1",
            "identity_sha256": identity_hash,
            "sample_id": sample_id,
            "counts": source["counts"],
            "initial_structure": structure,
            "initial_values": source["optimized"]["values"],
            "status": "running",
        }
        atomic_json(target, record)
        durable_log = root / "logs" / f"sample_{sample_id:06d}"
        scratch_parent = Path(args.scratch_directory)
        scratch_parent.mkdir(parents=True, exist_ok=True)
        working_log = Path(tempfile.mkdtemp(
            prefix=f"nagen_joint_relax_{sample_id:06d}_",
            dir=scratch_parent,
        ))
        try:
            from ase import Atoms

            atoms = Atoms(
                symbols=structure["elements"],
                scaled_positions=np.asarray(structure["frac_coords"], dtype=float),
                cell=np.asarray(structure["lattice"], dtype=float),
                pbc=True,
            )
            atoms.calc = calculator
            initial_energy = float(atoms.get_potential_energy()) / len(atoms)
            result = relax_structure(
                atoms,
                calculator,
                str(working_log),
                config,
            )
            durable_log.mkdir(parents=True, exist_ok=True)
            shutil.copytree(working_log, durable_log, dirs_exist_ok=True)
            final_energy = float(result.atoms.get_potential_energy()) / len(atoms)
            final_e_hull = calibrated.e_hull(source["counts"], final_energy)
            relaxed = _structure(result.atoms)
            feasibility = evaluate_feasibility(
                relaxed["elements"], relaxed["frac_coords"], relaxed["lattice"],
                final_e_hull,
            )
            acceptance = final_acceptance(
                result.converged,
                initial_energy,
                final_energy,
                feasibility,
                args.energy_increase_tolerance,
            )
            record.update({
                "status": acceptance["status"],
                "accepted": acceptance["accepted"],
                "acceptance": acceptance,
                "converged": result.converged,
                "relaxation_status": result.status,
                "fire_converged": result.fire_converged,
                "bfgs_converged": result.bfgs_converged,
                "final_fmax_eV_A": result.final_fmax_eV_A,
                "initial_E_UMA_eV_atom": initial_energy,
                "final_E_UMA_eV_atom": final_energy,
                "final_E_hull_eV_atom": final_e_hull,
                "final_structure": relaxed,
                "final_feasibility": feasibility,
                "log_directory": str(durable_log),
            })
            counts["processed"] += 1
            counts["accepted"] += int(acceptance["accepted"])
            counts["converged"] += int(result.converged)
            counts["feasible"] += int(result.converged and feasibility["feasible"])
        except Exception as error:
            durable_log.mkdir(parents=True, exist_ok=True)
            shutil.copytree(working_log, durable_log, dirs_exist_ok=True)
            record.update({
                "status": "error", "error": repr(error),
                "traceback": traceback.format_exc(),
            })
            counts["failed"] += 1
        finally:
            shutil.rmtree(working_log, ignore_errors=True)
        atomic_json(target, record)
        print(json.dumps({
            "sample_id": sample_id,
            "status": record["status"],
            "accepted": record.get("accepted", False),
            "converged": record.get("converged", False),
            "E_hull": record.get("final_E_hull_eV_atom"),
            "feasible": record.get("final_feasibility", {}).get("feasible", False),
        }), flush=True)
    manifest.update({
        "status": (
            "complete" if counts["processed"] == len(source_records)
            else "incomplete"
        ),
        "counts": counts,
    })
    atomic_json(root / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-campaign", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--fire-fmax", type=float, default=0.05)
    parser.add_argument("--fire-steps", type=int, default=1500)
    parser.add_argument("--bfgs-fmax", type=float, default=0.01)
    parser.add_argument("--bfgs-steps", type=int, default=1500)
    parser.add_argument(
        "--energy-increase-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scratch-directory", default="/tmp")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

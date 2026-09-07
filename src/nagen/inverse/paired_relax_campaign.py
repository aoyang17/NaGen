"""Resumable post-generation UMA relaxation for paired ShootingFlow M0 output."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ._io import atomic_json, sha256_file, sha256_json
from ._uma import load_calculator
from .constraints import evaluate_feasibility
from .relax_uma import RelaxationConfig, relax_structure
from .uma_guidance import checkpoint_sha256
from .uma_hull import UMAHullCache
from .uma_hull_campaign import sync_relaxation_logs


PREFILTER_CHECKS = (
    "element_support_exact",
    "atom_count_range",
    "finite_structure",
    "fractional_coordinate_domain",
    "positive_lattice_determinant",
    "volume_per_atom_range",
    "composition_domain",
    "fe_average_valence",
    "specific_capacity",
    "minimum_pbc_distances",
    "p_coordination_eq_4",
)


def passes_relaxation_prefilter(feasibility: dict[str, Any]) -> bool:
    """Allow Fe coordination trade-off, but never collisions/P/composition failures."""
    checks = feasibility["checks"]
    return all(bool(checks[name]) for name in PREFILTER_CHECKS)


def _structure(atoms: Any) -> dict[str, Any]:
    return {
        "elements": atoms.get_chemical_symbols(),
        "frac_coords": np.asarray(
            atoms.get_scaled_positions(wrap=True), dtype=float
        ).tolist(),
        "lattice": np.asarray(atoms.cell.array, dtype=float).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
    }


def _load_source_records(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("source paired campaign is not complete")
    records = [
        json.loads(path.read_text())
        for path in sorted((root / "records").glob("sample_*.json"))
    ]
    expected = int(manifest["count"])
    if len(records) != expected:
        raise RuntimeError(
            f"source paired campaign has {len(records)} records, expected {expected}"
        )
    if any(record.get("status") != "complete" for record in records):
        raise RuntimeError("source paired campaign contains incomplete samples")
    return manifest, sorted(records, key=lambda record: int(record["sample_id"]))


def run(args: argparse.Namespace) -> None:
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("shard index must be in [0, shard_count)")
    source_root = Path(args.source_campaign)
    source_manifest, all_records = _load_source_records(source_root)
    selected = [
        record for index, record in enumerate(all_records)
        if index % args.shard_count == args.shard_index
    ]
    checkpoint_hash = checkpoint_sha256(args.uma_checkpoint)
    hull_cache = UMAHullCache(
        args.uma_hull_cache,
        expected_model_sha256=checkpoint_hash,
        expected_task_name=args.task_name,
    )
    config = RelaxationConfig(
        fire_fmax_eV_A=args.fire_fmax,
        fire_steps=args.fire_steps,
        bfgs_fmax_eV_A=args.bfgs_fmax,
        bfgs_steps=args.bfgs_steps,
    )
    protocol = {
        **config.__dict__,
        "fire_variables": "cell_and_positions_via_ExpCellFilter",
        "bfgs_variables": "positions_only",
        "energy_normalization": "eV_per_atom",
    }
    identity = {
        "version": "paired-post-uma-relax-shard-v1",
        "source_campaign": str(source_root.resolve()),
        "source_campaign_identity_sha256": source_manifest[
            "campaign_identity_sha256"
        ],
        "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "uma_checkpoint": str(Path(args.uma_checkpoint).resolve()),
        "uma_model_sha256": checkpoint_hash,
        "task_name": args.task_name,
        "uma_hull_cache": str(Path(args.uma_hull_cache).resolve()),
        "uma_hull_cache_sha256": sha256_file(args.uma_hull_cache),
        "uma_hull_content_sha256": hull_cache.data["content_sha256"],
        "relaxation_protocol": protocol,
        "relaxation_protocol_sha256": sha256_json(protocol),
        "prefilter_checks": list(PREFILTER_CHECKS),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    identity_hash = sha256_json(identity)
    output_root = Path(args.out_directory)
    records_root = output_root / "records"
    durable_logs = output_root / "logs"
    scratch_root = Path(args.scratch_directory) if args.scratch_directory else None
    calculator = None
    counts = {
        "selected": len(selected), "prefilter_rejected": 0,
        "relaxation_attempted": 0, "relaxation_converged": 0,
        "post_relaxation_all_constraints_pass": 0, "failed": 0, "skipped": 0,
    }
    for source in selected:
        sample_id = int(source["sample_id"])
        target = records_root / f"sample_{sample_id:06d}.json"
        if target.exists() and not args.resume:
            raise FileExistsError(f"post-relaxation record exists: {target}")
        if target.exists() and args.resume:
            previous = json.loads(target.read_text())
            if previous.get("identity_sha256") != identity_hash:
                raise RuntimeError(f"post-relaxation resume mismatch: {sample_id}")
            if previous.get("status") in {
                "complete", "prefilter_rejected", "relaxation_not_converged",
            }:
                counts["skipped"] += 1
                if previous["status"] == "prefilter_rejected":
                    counts["prefilter_rejected"] += 1
                else:
                    counts["relaxation_attempted"] += 1
                    counts["failed"] += int(
                        previous["status"] == "relaxation_not_converged"
                    )
                counts["relaxation_converged"] += int(
                    previous.get("relaxation_converged", False)
                )
                counts["post_relaxation_all_constraints_pass"] += int(
                    previous.get("final_feasibility", {}).get("feasible", False)
                )
                continue
            attempt = int(previous.get("attempt", 1)) + 1
        else:
            attempt = 1
        m0 = source["M0"]
        initial_feasibility = m0["exact"]["feasibility"]
        record: dict[str, Any] = {
            "version": "paired-post-uma-relax-record-v1",
            "identity_sha256": identity_hash,
            "sample_id": sample_id,
            "source_record_campaign_identity_sha256": source[
                "campaign_identity_sha256"
            ],
            "counts": source["counts"],
            "attempt": attempt,
            "initial_structure": m0["exact"]["structure"],
            "initial_proxy_values": m0["values"],
            "initial_feasibility": initial_feasibility,
            "prefilter_pass": passes_relaxation_prefilter(initial_feasibility),
            "uma_model_sha256": checkpoint_hash,
            "task_name": args.task_name,
            "relaxation_protocol": protocol,
            "relaxation_protocol_sha256": identity[
                "relaxation_protocol_sha256"
            ],
        }
        if not record["prefilter_pass"]:
            record["status"] = "prefilter_rejected"
            counts["prefilter_rejected"] += 1
            atomic_json(target, record)
            continue
        counts["relaxation_attempted"] += 1
        record.update({"status": "running", "relaxation_converged": False})
        atomic_json(target, record)
        try:
            if calculator is None:
                calculator = load_calculator(
                    args.uma_checkpoint, args.device, args.task_name
                )
            from ase import Atoms
            structure = record["initial_structure"]
            atoms = Atoms(
                symbols=structure["elements"],
                scaled_positions=np.asarray(structure["frac_coords"], dtype=float),
                cell=np.asarray(structure["lattice"], dtype=float),
                pbc=True,
            )
            atoms.calc = calculator
            initial_energy = float(atoms.get_potential_energy()) / len(atoms)
            durable_log_directory = (
                durable_logs / f"sample_{sample_id:06d}" / f"attempt_{attempt:03d}"
            )
            working_log_directory = (
                scratch_root
                / f"shard_{args.shard_index:03d}_pid_{os.getpid()}"
                / f"sample_{sample_id:06d}"
                / f"attempt_{attempt:03d}"
                if scratch_root is not None else durable_log_directory
            )
            result = relax_structure(
                atoms, calculator, str(working_log_directory), config
            )
            if scratch_root is not None:
                sync_relaxation_logs(
                    working_log_directory, durable_log_directory
                )
                shutil.rmtree(working_log_directory, ignore_errors=True)
            final_energy = float(result.atoms.get_potential_energy()) / len(atoms)
            reference_energy = hull_cache.get(source["counts"])
            e_hull = final_energy - reference_energy
            relaxed = _structure(result.atoms)
            final_feasibility = evaluate_feasibility(
                relaxed["elements"], relaxed["frac_coords"], relaxed["lattice"],
                e_hull,
            )
            final_stress = np.asarray(
                result.atoms.get_stress(voigt=False), dtype=float
            )
            record.update({
                "status": (
                    "complete" if result.converged
                    else "relaxation_not_converged"
                ),
                "relaxation_status": result.status,
                "relaxation_converged": result.converged,
                "fire_converged": result.fire_converged,
                "bfgs_converged": result.bfgs_converged,
                "final_fmax_eV_A": result.final_fmax_eV_A,
                "final_stress_eV_A3": final_stress.tolist(),
                "final_max_abs_stress_eV_A3": float(np.abs(final_stress).max()),
                "initial_E_UMA_eV_atom": initial_energy,
                "final_E_UMA_eV_atom": final_energy,
                "E_ref_UMA_eV_atom": reference_energy,
                "E_hull_eV_atom": e_hull,
                "final_structure": relaxed,
                "final_feasibility": final_feasibility,
                "log_directory": str(durable_log_directory),
            })
            counts["relaxation_converged"] += int(result.converged)
            counts["failed"] += int(not result.converged)
            counts["post_relaxation_all_constraints_pass"] += int(
                result.converged and final_feasibility["feasible"]
            )
        except Exception as error:
            record.update({
                "status": "error", "error": repr(error),
                "traceback": traceback.format_exc(),
            })
            counts["failed"] += 1
        atomic_json(target, record)
        print(json.dumps({
            "sample_id": sample_id, "status": record["status"],
            "converged": record.get("relaxation_converged", False),
            "final_feasible": record.get("final_feasibility", {}).get(
                "feasible", False
            ),
        }), flush=True)
    manifest = identity | {"identity_sha256": identity_hash, "counts": counts}
    atomic_json(
        output_root / f"manifest_shard_{args.shard_index:03d}.json", manifest
    )
    print(json.dumps(counts, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-campaign", required=True)
    parser.add_argument("--uma-hull-cache", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--fire-fmax", type=float, default=0.05)
    parser.add_argument("--fire-steps", type=int, default=1500)
    parser.add_argument("--bfgs-fmax", type=float, default=0.01)
    parser.add_argument("--bfgs-steps", type=int, default=1500)
    parser.add_argument("--scratch-directory", default="/tmp/nagen-paired-relax")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

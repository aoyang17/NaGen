"""Build resumable Flow-generated UMA relaxation pairs for relax-aware D-Flow."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._io import atomic_json, sha256_file, sha256_json
from ._uma import load_calculator
from .calibrated_mp_hull import CalibratedMPHullCache, build_calibrated_cache
from .constraints import evaluate_feasibility
from .generate import (
    generate_batch,
    load_model,
)
from .joint_dflow import model_sha256
from .relax_uma import RelaxationConfig, maximum_force, relax_structure
from .uma_guidance import checkpoint_sha256


def _counts(elements: list[str]) -> dict[str, int]:
    counter = Counter(elements)
    return {element: int(counter[element]) for element in ("Na", "Fe", "P", "O")}


def _structure(elements, frac, lattice) -> dict[str, Any]:
    return {
        "elements": list(elements),
        "frac_coords": np.asarray(frac, dtype=float).tolist(),
        "lattice": np.asarray(lattice, dtype=float).tolist(),
        "pbc": [True, True, True],
    }


def generate_sources(args: argparse.Namespace) -> None:
    if not 200 <= args.count <= 500:
        raise ValueError("surrogate source campaign count must be in [200, 500]")
    root = Path(args.out_directory)
    records_root = root / "sources"
    records_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    joint, joint_checkpoint = load_model(args.joint_checkpoint, device)
    geometry, geometry_checkpoint = load_model(args.geometry_checkpoint, device)
    if joint_checkpoint["element_to_index"] != geometry_checkpoint["element_to_index"]:
        raise ValueError("joint and geometry checkpoints use different vocabularies")
    inverse = {
        int(index): element
        for element, index in geometry_checkpoint["element_to_index"].items()
    }
    identity = {
        "version": "relaxation-surrogate-sources-v1",
        "count": args.count,
        "seed": args.seed,
        "ode_steps": args.ode_steps,
        "distribution": "unconditional_p(N,A)_then_p(X,L|N,A)",
        "joint_checkpoint": str(Path(args.joint_checkpoint).resolve()),
        "joint_checkpoint_sha256": sha256_file(args.joint_checkpoint),
        "joint_model_sha256": model_sha256(joint),
        "geometry_checkpoint": str(Path(args.geometry_checkpoint).resolve()),
        "geometry_checkpoint_sha256": sha256_file(args.geometry_checkpoint),
        "geometry_model_sha256": model_sha256(geometry),
        "phase_cache": str(Path(args.phase_cache).resolve()),
        "phase_cache_sha256": sha256_file(args.phase_cache),
    }
    identity_hash = sha256_json(identity)
    atomic_json(root / "manifest.json", identity | {
        "identity_sha256": identity_hash, "status": "running", "completed": 0,
    })
    generated = 0
    batch_id = 0
    all_counts: list[dict[str, int]] = []
    while generated < args.count:
        batch = min(args.batch_size, args.count - generated)
        types, frac, lattice, mask = generate_batch(
            joint, joint_checkpoint, batch, args.seed + batch_id,
            args.ode_steps, "midpoint", geometry_model=geometry,
            composition_filter=False,
        )
        for row in range(batch):
            n_atoms = int(mask[row].sum())
            elements = [inverse[int(value)] for value in types[row, :n_atoms].cpu()]
            structure = _structure(
                elements, frac[row, :n_atoms].cpu(), lattice[row].cpu()
            )
            counts = _counts(elements)
            record = {
                "version": "relaxation-surrogate-source-v1",
                "identity_sha256": identity_hash,
                "sample_id": generated,
            "seed": args.seed + batch_id,
                "counts": counts,
                "structure": structure,
                "geometry_before": evaluate_feasibility(
                    elements, structure["frac_coords"], structure["lattice"], None
                ),
                "status": "complete",
            }
            atomic_json(records_root / f"sample_{generated:06d}.json", record)
            all_counts.append(counts)
            generated += 1
        batch_id += 1
        print(json.dumps({"generated": generated, "target": args.count}), flush=True)
    calibrated = build_calibrated_cache(args.phase_cache, all_counts)
    atomic_json(root / "calibrated_mp2020_hull_cache.json", calibrated, sort_keys=True)
    atomic_json(root / "manifest.json", identity | {
        "identity_sha256": identity_hash,
        "status": "complete",
        "completed": generated,
        "calibrated_cache_sha256": sha256_file(
            root / "calibrated_mp2020_hull_cache.json"
        ),
    })


def relax_shard(args: argparse.Namespace) -> None:
    source_root = Path(args.source_campaign)
    manifest = json.loads((source_root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("source campaign is not complete")
    calibrated = CalibratedMPHullCache(
        source_root / "calibrated_mp2020_hull_cache.json"
    )
    source_paths = sorted((source_root / "sources").glob("sample_*.json"))
    selected = [
        path for path in source_paths
        if int(path.stem.split("_")[-1]) % args.num_shards == args.shard_index
    ]
    root = Path(args.out_directory) / f"shard_{args.shard_index:03d}"
    records_root = root / "records"
    records_root.mkdir(parents=True, exist_ok=True)
    config = RelaxationConfig(
        fire_fmax_eV_A=args.fire_fmax, fire_steps=args.fire_steps,
        bfgs_fmax_eV_A=args.bfgs_fmax, bfgs_steps=args.bfgs_steps,
    )
    identity = {
        "version": "relaxation-surrogate-label-shard-v1",
        "source_identity_sha256": manifest["identity_sha256"],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "uma_checkpoint": str(Path(args.uma_checkpoint).resolve()),
        "uma_checkpoint_sha256": checkpoint_sha256(args.uma_checkpoint),
        "task_name": args.task_name,
        "protocol": config.__dict__,
    }
    identity_hash = sha256_json(identity)
    atomic_json(root / "manifest.json", identity | {
        "identity_sha256": identity_hash, "status": "running",
        "selected": len(selected), "completed": 0,
    })
    calculator = load_calculator(args.uma_checkpoint, args.device, args.task_name)
    completed = 0
    for path in selected:
        source = json.loads(path.read_text())
        sample_id = int(source["sample_id"])
        target = records_root / f"sample_{sample_id:06d}.json"
        if target.exists() and args.resume:
            previous = json.loads(target.read_text())
            if previous.get("status") in {"complete", "not_converged"}:
                completed += 1
                continue
        structure = source["structure"]
        record = {
            "version": "relaxation-surrogate-pair-v1",
            "identity_sha256": identity_hash,
            "sample_id": sample_id,
            "counts": source["counts"],
            "initial_structure": structure,
            "status": "running",
        }
        atomic_json(target, record)
        work = Path(tempfile.mkdtemp(
            prefix=f"nagen_relax_surrogate_{sample_id:06d}_", dir=args.scratch_directory
        ))
        durable_log = root / "logs" / f"sample_{sample_id:06d}"
        try:
            from ase import Atoms
            atoms = Atoms(
                symbols=structure["elements"],
                scaled_positions=np.asarray(structure["frac_coords"], dtype=float),
                cell=np.asarray(structure["lattice"], dtype=float), pbc=True,
            )
            atoms.calc = calculator
            initial_energy = float(atoms.get_potential_energy()) / len(atoms)
            initial_fmax = maximum_force(atoms)
            initial_hull = calibrated.e_hull(source["counts"], initial_energy)
            initial_feasibility = evaluate_feasibility(
                structure["elements"], structure["frac_coords"],
                structure["lattice"], initial_hull,
            )
            result = relax_structure(atoms, calculator, str(work), config)
            final_energy = float(result.atoms.get_potential_energy()) / len(atoms)
            final_hull = calibrated.e_hull(source["counts"], final_energy)
            final_structure = _structure(
                result.atoms.get_chemical_symbols(),
                result.atoms.get_scaled_positions(wrap=True), result.atoms.cell.array,
            )
            final_feasibility = evaluate_feasibility(
                final_structure["elements"], final_structure["frac_coords"],
                final_structure["lattice"], final_hull,
            )
            record.update({
                "status": "complete" if result.converged else "not_converged",
                "converged": result.converged,
                "relaxation_status": result.status,
                "initial_E_UMA_eV_atom": initial_energy,
                "final_E_UMA_eV_atom": final_energy,
                "delta_E_relax_eV_atom": final_energy - initial_energy,
                "initial_E_hull_eV_atom": initial_hull,
                "final_E_hull_eV_atom": final_hull,
                "initial_fmax_eV_A": initial_fmax,
                "final_fmax_eV_A": result.final_fmax_eV_A,
                "initial_feasibility": initial_feasibility,
                "final_feasibility": final_feasibility,
                "final_structure": final_structure,
            })
            completed += 1
        except Exception as error:
            record.update({
                "status": "error", "error": repr(error),
                "traceback": traceback.format_exc(),
            })
        finally:
            durable_log.mkdir(parents=True, exist_ok=True)
            shutil.copytree(work, durable_log, dirs_exist_ok=True)
            shutil.rmtree(work, ignore_errors=True)
        atomic_json(target, record)
        print(json.dumps({
            "sample_id": sample_id, "status": record["status"],
            "delta_E_relax": record.get("delta_E_relax_eV_atom"),
            "final_E_hull": record.get("final_E_hull_eV_atom"),
        }), flush=True)
    atomic_json(root / "manifest.json", identity | {
        "identity_sha256": identity_hash,
        "status": "complete" if completed == len(selected) else "incomplete",
        "selected": len(selected), "completed": completed,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--joint-checkpoint", required=True)
    generate.add_argument("--geometry-checkpoint", required=True)
    generate.add_argument("--phase-cache", required=True)
    generate.add_argument("--out-directory", required=True)
    generate.add_argument("--count", type=int, default=200)
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--seed", type=int, default=20260902)
    generate.add_argument("--ode-steps", type=int, default=24)
    generate.add_argument("--max-atoms", type=int, default=120)
    generate.add_argument("--device", default="cuda")
    generate.set_defaults(func=generate_sources)
    relax = subparsers.add_parser("relax")
    relax.add_argument("--source-campaign", required=True)
    relax.add_argument("--uma-checkpoint", required=True)
    relax.add_argument("--out-directory", required=True)
    relax.add_argument("--shard-index", type=int, required=True)
    relax.add_argument("--num-shards", type=int, required=True)
    relax.add_argument("--fire-fmax", type=float, default=0.05)
    relax.add_argument("--fire-steps", type=int, default=1500)
    relax.add_argument("--bfgs-fmax", type=float, default=0.01)
    relax.add_argument("--bfgs-steps", type=int, default=1500)
    relax.add_argument("--task-name", default="omat")
    relax.add_argument("--device", default="cuda")
    relax.add_argument("--scratch-directory", default="/tmp")
    relax.add_argument("--resume", action="store_true")
    relax.set_defaults(func=relax_shard)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

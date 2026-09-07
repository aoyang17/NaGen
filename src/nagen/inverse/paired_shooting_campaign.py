"""Resumable paired B0/B1/B2/M0 ShootingFlow campaign.

Pilot runs may estimate objective scales from their complete baseline slice.
Main runs must receive the scales frozen by the accepted pilot.
"""

from __future__ import annotations

import argparse
import json
import statistics
import traceback
from pathlib import Path
from typing import Any

import torch

from ._campaign import fixed_composition_source as _source
from ._campaign import state_payload as _state
from ._io import atomic_json, sha256_file, sha256_json
from .constraints import evaluate_feasibility
from .generate import load_model
from .model import CrystalState
from .multiobjective_shooting import (
    CrystalDesignProblem,
    ShootingConfig,
    ShootingFlowRunner,
    model_sha256,
)
from .novelty import NoveltyIndex
from .sample import decode_terminal, integrate_flow
from .uma_guidance import load_uma_calculator_vjp_factory
from .uma_hull import UMAHullCache


ATOMIC_NUMBERS = {"Na": 11, "Fe": 26, "P": 15, "O": 8}
GEOMETRY_CHECKS = (
    "positive_lattice_determinant",
    "volume_per_atom_range",
    "minimum_pbc_distances",
    "p_coordination_eq_4",
    "fe_coordination_in_4_5_6",
)


def campaign_composition_index(sample_id: int, composition_count: int) -> int:
    if sample_id < 0 or composition_count <= 0:
        raise ValueError("sample ID and composition count must be positive")
    return sample_id % composition_count


def _values(problem: CrystalDesignProblem, terminal: CrystalState) -> dict[str, float]:
    with torch.no_grad():
        return {key: float(value) for key, value in problem.evaluate(terminal).items()}


def _exact(
    model, checkpoint: dict[str, Any], mask: torch.Tensor,
    terminal: CrystalState, expected_elements: list[str], e_hull: float,
) -> dict[str, Any]:
    with torch.no_grad():
        types, frac, lattice = decode_terminal(model, terminal, mask)
    inverse = {
        int(index): element for element, index in checkpoint["element_to_index"].items()
    }
    decoded = [inverse[int(index)] for index in types[0].detach().cpu().tolist()]
    feasibility = evaluate_feasibility(
        decoded,
        frac[0].detach().cpu().numpy(),
        lattice[0].detach().cpu().numpy(),
        e_hull,
    )
    return {
        "fixed_A_preserved": decoded == expected_elements,
        "geometry_pass": all(
            feasibility["checks"][name] for name in GEOMETRY_CHECKS
        ),
        "feasibility": feasibility,
        "structure": {
            "elements": decoded,
            "frac_coords": frac[0].detach().cpu().tolist(),
            "lattice": lattice[0].detach().cpu().tolist(),
        },
    }


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "restoration_steps": args.restoration_steps,
        "guidance_steps": args.guidance_steps,
        "learning_rate": args.learning_rate,
        "gradient_clip": args.gradient_clip,
        "source_prior": args.source_prior,
        "augmented_rho": args.augmented_rho,
        "distance_margin_A": args.distance_margin_A,
        "distance_weight": args.distance_weight,
        "p_coordination_weight": args.p_coordination_weight,
        "fe_coordination_weight": args.fe_coordination_weight,
        "hull_threshold_eV_atom": args.hull_threshold,
        "hull_penalty_weight": args.hull_penalty_weight,
        "m0_only": bool(args.m0_only),
        "fe_coordination_prior": list(args.fe_coordination_prior),
        "ode_steps": args.ode_steps,
        "integrator": "midpoint",
    }


def run(args: argparse.Namespace) -> None:
    if args.count <= 0 or args.start_index < 0:
        raise ValueError("campaign slice must be nonempty and nonnegative")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("shard index must be in [0, shard_count)")
    if args.shard_count > args.count:
        raise ValueError("shard count cannot exceed campaign sample count")
    if (args.hull_scale is None) != (args.novelty_scale is None):
        raise ValueError("objective scales must be both supplied or both estimated")
    if args.main_run and args.hull_scale is None:
        raise ValueError("main run requires objective scales frozen by the pilot")
    if args.shard_count > 1 and args.hull_scale is None:
        raise ValueError(
            "sharded campaigns require objective scales calibrated on the full pilot"
        )
    if args.scale_only_out and args.shard_count != 1:
        raise ValueError("objective-scale calibration must use the complete unsharded slice")
    if args.scale_only_out and args.hull_scale is not None:
        raise ValueError("objective-scale calibration cannot receive frozen scales")
    device = torch.device(args.device)
    geometry_path = Path(args.geometry_checkpoint)
    novelty_path = Path(args.novelty_index)
    hull_path = Path(args.uma_hull_cache)
    model, checkpoint = load_model(str(geometry_path), device, use_ema=True)
    flow_hash = model_sha256(model)
    pool = json.loads(Path(args.composition_pool).read_text())
    compositions = pool["unique_compositions"]
    if not compositions:
        raise ValueError("composition pool is empty")
    novelty = NoveltyIndex.load(str(novelty_path), device)
    factory, uma_metadata = load_uma_calculator_vjp_factory(
        args.uma_checkpoint, device=args.device, task_name=args.task_name
    )
    hull_cache = UMAHullCache(
        str(hull_path),
        expected_model_sha256=uma_metadata["checkpoint_sha256"],
        expected_task_name=uma_metadata["task_name"],
    )
    root = Path(args.out_directory)
    records_dir = root / "records"
    configuration = _configuration(args)
    identity = {
        "version": "paired-shooting-campaign-v1",
        "role": "main_256" if args.main_run else "pilot",
        "start_index": args.start_index,
        "count": args.count,
        "execution_shard_count": args.shard_count,
        "seed": args.seed,
        "configuration": configuration,
        "geometry_checkpoint": str(geometry_path.resolve()),
        "geometry_checkpoint_sha256": sha256_file(geometry_path),
        "flow_model_sha256": flow_hash,
        "composition_pool": str(Path(args.composition_pool).resolve()),
        "composition_pool_sha256": sha256_file(Path(args.composition_pool)),
        "novelty_index": str(novelty_path.resolve()),
        "novelty_index_sha256": sha256_file(novelty_path),
        "uma": uma_metadata,
        "uma_hull_cache": str(hull_path.resolve()),
        "uma_hull_cache_sha256": sha256_file(hull_path),
        "uma_hull_content_sha256": hull_cache.data["content_sha256"],
        "uma_hull_support_entry_count": hull_cache.data[
            "relaxed_support_entry_count"
        ],
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key, value in identity.items():
            if previous.get(key) != value:
                raise RuntimeError(f"campaign resume identity mismatch: {key}")
    elif root.exists():
        # A launcher may create a logs directory before concurrent shard
        # processes race to write the shared manifest.  Other pre-existing
        # content is still rejected to protect resumability and provenance.
        unexpected = [
            path for path in root.iterdir() if path.name not in {"logs"}
        ]
        if unexpected:
            raise RuntimeError("nonempty campaign directory has no matching manifest")

    prepared: list[dict[str, Any]] = []
    sample_ids = [
        sample_id
        for offset, sample_id in enumerate(
            range(args.start_index, args.start_index + args.count)
        )
        if offset % args.shard_count == args.shard_index
    ]
    for sample_id in sample_ids:
        composition_index = campaign_composition_index(sample_id, len(compositions))
        composition = compositions[composition_index]
        sample_seed = args.seed + sample_id
        source, mask = _source(model, composition, sample_seed, device)
        elements = list(composition["elements"])
        numbers = torch.tensor(
            [ATOMIC_NUMBERS[element] for element in elements], device=device
        )
        problem = CrystalDesignProblem(
            model,
            mask,
            factory(numbers),
            hull_cache.get(composition["counts"]),
            novelty,
            fe_coordination_prior=tuple(args.fe_coordination_prior),
            constraint_margin_A=args.distance_margin_A,
        )
        with torch.no_grad():
            terminal = integrate_flow(
                model, source, mask, steps=args.ode_steps,
                method="midpoint", freeze_atom=True,
            )
        values = _values(problem, terminal)
        prepared.append({
            "sample_id": sample_id,
            "sample_seed": sample_seed,
            "composition_index": composition_index,
            "composition": composition,
            "elements": elements,
            "source": source,
            "mask": mask,
            "problem": problem,
            "baseline_terminal": terminal,
            "baseline_values": values,
        })
    scales = (
        (float(args.hull_scale), float(args.novelty_scale))
        if args.hull_scale is not None else
        (
            max(statistics.pstdev(
                item["baseline_values"]["E_hull"] for item in prepared
            ), 1e-3),
            max(statistics.pstdev(
                item["baseline_values"]["novelty"] for item in prepared
            ), 1e-3),
        )
    )
    campaign_identity_sha256 = sha256_json(
        identity | {"objective_scales": list(scales)}
    )
    if args.scale_only_out:
        atomic_json(Path(args.scale_only_out), {
            "version": "paired-shooting-objective-scales-v1",
            "campaign_identity_sha256": campaign_identity_sha256,
            "baseline_records": len(prepared),
            "objective_scales": list(scales),
            "baseline_values": [
                {
                    "sample_id": item["sample_id"],
                    "sample_seed": item["sample_seed"],
                    "composition_index": item["composition_index"],
                    "counts": item["composition"]["counts"],
                    "values": item["baseline_values"],
                }
                for item in prepared
            ],
            "identity": identity,
        })
        print(json.dumps({
            "scale_only_out": str(Path(args.scale_only_out)),
            "baseline_records": len(prepared),
            "objective_scales": list(scales),
        }, indent=2))
        return
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("campaign_identity_sha256") != campaign_identity_sha256:
            raise RuntimeError("campaign resume objective scales mismatch")
    manifest = identity | {
        "status": "running",
        "objective_scales": list(scales),
        "campaign_identity_sha256": campaign_identity_sha256,
        "variants": ["B0", "B1", "B2", "M0"],
    }
    atomic_json(manifest_path, manifest)

    for item in prepared:
        sample_id = item["sample_id"]
        target = records_dir / f"sample_{sample_id:06d}.json"
        if target.exists() and args.resume:
            previous = json.loads(target.read_text())
            if (
                previous.get("campaign_identity_sha256")
                != campaign_identity_sha256
            ):
                raise RuntimeError(f"sample resume identity mismatch: {sample_id}")
            if previous.get("status") == "complete":
                continue
        elif target.exists():
            raise FileExistsError(f"sample record exists: {target}")
        baseline_exact = _exact(
            model, checkpoint, item["mask"], item["baseline_terminal"],
            item["elements"], item["baseline_values"]["E_hull"],
        )
        record: dict[str, Any] = {
            "version": "paired-shooting-sample-v1",
            "campaign_identity_sha256": campaign_identity_sha256,
            "sample_id": sample_id,
            "seed": item["sample_seed"],
            "composition_index": item["composition_index"],
            "N": item["composition"]["N"],
            "counts": item["composition"]["counts"],
            "z0": _state(item["source"]),
            "B0": {
                "status": "complete",
                "values": item["baseline_values"],
                "exact": baseline_exact,
                "terminal": _state(item["baseline_terminal"]),
            },
            "status": "running",
        }
        atomic_json(target, record)
        variants = (
            {"M0": {"use_hull": True, "use_novelty": True}}
            if args.m0_only else
            {
                "B1": {"use_hull": False, "use_novelty": False},
                "B2": {"use_hull": False, "use_novelty": True},
                "M0": {"use_hull": True, "use_novelty": True},
            }
        )
        for name, mode in variants.items():
            try:
                result = ShootingFlowRunner(model, item["problem"]).run(
                    item["composition"]["N"], item["source"].atom,
                    item["source"],
                    ShootingConfig(
                        restoration_steps=args.restoration_steps,
                        guidance_steps=args.guidance_steps,
                        learning_rate=args.learning_rate,
                        gradient_clip=args.gradient_clip,
                        source_prior=args.source_prior,
                        augmented_rho=args.augmented_rho,
                        ode_steps=args.ode_steps,
                        objective_scales=scales,
                        use_hull=mode["use_hull"],
                        use_novelty=mode["use_novelty"],
                        use_constraints=True,
                        distance_weight=args.distance_weight,
                        p_coordination_weight=args.p_coordination_weight,
                        fe_coordination_weight=args.fe_coordination_weight,
                        hull_threshold_eV_atom=args.hull_threshold,
                        hull_penalty_weight=args.hull_penalty_weight,
                    ),
                )
                values = _values(item["problem"], result.terminal)
                record[name] = {
                    "status": result.status,
                    "values": values,
                    "history": result.history,
                    "z0_star": _state(result.z0_star),
                    "terminal": _state(result.terminal),
                    "exact": _exact(
                        model, checkpoint, item["mask"], result.terminal,
                        item["elements"], values["E_hull"],
                    ),
                }
            except Exception as error:
                record[name] = {
                    "status": "error",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            atomic_json(target, record)
        record["status"] = (
            "complete" if all(
                record[name].get("status") == "complete" for name in variants
            ) else "variant_failure"
        )
        atomic_json(target, record)
        print(json.dumps({
            "sample_id": sample_id,
            "status": record["status"],
            "variants": {name: record[name]["status"] for name in variants},
        }), flush=True)
    manifest["completed_records"] = len(list(records_dir.glob("sample_*.json")))
    manifest["status"] = (
        "complete" if manifest["completed_records"] == args.count else "incomplete"
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "output": str(root), "status": manifest["status"],
        "completed_records": manifest["completed_records"],
        "shard_index": args.shard_index,
        "shard_records": len(sample_ids),
        "objective_scales": manifest["objective_scales"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--composition-pool", required=True)
    parser.add_argument("--uma-hull-cache", required=True)
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument(
        "--restoration-steps", type=int, default=0,
        help="legacy objective-free phase; default 0 for full joint M0 guidance",
    )
    parser.add_argument("--guidance-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--source-prior", type=float, default=0.02)
    parser.add_argument("--augmented-rho", type=float, default=1.0)
    parser.add_argument("--distance-margin-A", type=float, default=0.10)
    parser.add_argument("--distance-weight", type=float, default=40.0)
    parser.add_argument("--p-coordination-weight", type=float, default=12.0)
    parser.add_argument("--fe-coordination-weight", type=float, default=2.0)
    parser.add_argument(
        "--hull-threshold", type=float, default=0.150,
        help="strict E_hull target enforced by the guidance hinge",
    )
    parser.add_argument(
        "--hull-penalty-weight", type=float, default=100.0,
        help="augmented-Lagrangian weight for E_hull threshold violation",
    )
    parser.add_argument(
        "--fe-coordination-prior", type=float, nargs=3,
        default=(0.084, 0.1025, 0.8135),
    )
    parser.add_argument("--hull-scale", type=float)
    parser.add_argument("--novelty-scale", type=float)
    parser.add_argument("--scale-only-out")
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--main-run", action="store_true")
    parser.add_argument(
        "--m0-only", action="store_true",
        help="run only the full joint M0 optimization branch",
    )
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

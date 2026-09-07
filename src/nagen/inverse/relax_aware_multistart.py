"""Unconditional multi-start relax-aware z_A,z_X,z_L IPOPT campaign."""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ._io import atomic_json, sha256_file, sha256_json
from .calibrated_mp_hull import build_calibrated_cache
from .constraints import evaluate_feasibility
from .differentiable_mp_hull import DifferentiableMP2020Hull
from .generate import load_model, sample_atom_counts, variable_random_source
from .joint_dflow import model_sha256
from .novelty import NoveltyIndex
from .relax_aware_dflow import (
    RelaxAwareConfig,
    RelaxAwareIPOPTSolver,
    RelaxAwareProblem,
)
from .relaxation_surrogate import RelaxationSurrogateEnsemble
from .sample import decode_terminal, integrate_flow


def _state(state) -> dict[str, Any]:
    return {
        "atom": state.atom.detach().cpu().tolist(),
        "frac": state.frac.detach().cpu().tolist(),
        "lattice": state.lattice.detach().cpu().tolist(),
    }


def _hard_structure(model, terminal, mask, checkpoint):
    with torch.no_grad():
        types, frac, lattice = decode_terminal(model, terminal, mask)
    inverse = {int(index): element for element, index in checkpoint["element_to_index"].items()}
    elements = [inverse[int(value)] for value in types[0, mask[0]].cpu()]
    return {
        "elements": elements,
        "frac_coords": frac[0, mask[0]].cpu().tolist(),
        "lattice": lattice[0].cpu().tolist(),
        "pbc": [True, True, True],
    }


def run(args: argparse.Namespace) -> None:
    if not 16 <= args.starts <= 32:
        raise ValueError("multi-start count must be in [16, 32]")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.joint_checkpoint, device)
    surrogate, surrogate_manifest = RelaxationSurrogateEnsemble.load(
        args.relaxation_surrogate, device
    )
    hull = DifferentiableMP2020Hull.from_phase_cache(
        args.phase_cache, temperature_eV_atom=args.hull_reference_temperature
    ).to(device)
    novelty = NoveltyIndex.load(args.novelty_index, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    n_atoms = sample_atom_counts(checkpoint["n_histogram"], args.starts, generator).to(device)
    config = RelaxAwareConfig(
        steps=args.optimization_steps, ode_steps=args.ode_steps,
        atom_temperature=args.atom_temperature,
        hull_temperature=args.hull_temperature,
        hull_uncertainty_kappa=args.hull_uncertainty_kappa,
        hull_scale=args.hull_scale,
        hull_weight=args.hull_weight,
        novelty_weight=args.novelty_weight,
        distance_weight=args.distance_weight,
        p_coordination_weight=args.p_coordination_weight,
        fe_coordination_weight=args.fe_coordination_weight,
        volume_weight=args.volume_weight,
        composition_weight=args.composition_weight,
        geometry_survival_weight=args.geometry_survival_weight,
        source_prior_weight=args.source_prior_weight,
    )
    root = Path(args.out_directory)
    records_root = root / "records"
    records_root.mkdir(parents=True, exist_ok=True)
    identity = {
        "version": "relax-aware-unconditional-multistart-v1",
        "starts": args.starts, "seed": args.seed,
        "distribution": "fully_unconditional_p(N,A,X,L)",
        "decision_variables": ["z_A", "z_X", "z_L"],
        "fixed_within_start": ["N"],
        "joint_checkpoint": str(Path(args.joint_checkpoint).resolve()),
        "joint_checkpoint_sha256": sha256_file(args.joint_checkpoint),
        "joint_model_sha256": model_sha256(model),
        "relaxation_surrogate": str(Path(args.relaxation_surrogate).resolve()),
        "relaxation_surrogate_sha256": sha256_file(args.relaxation_surrogate),
        "relaxation_surrogate_metrics": surrogate_manifest["metrics"],
        "phase_cache": str(Path(args.phase_cache).resolve()),
        "phase_cache_sha256": sha256_file(args.phase_cache),
        "novelty_index": str(Path(args.novelty_index).resolve()),
        "novelty_index_sha256": sha256_file(args.novelty_index),
        "config": asdict(config),
    }
    identity_hash = sha256_json(identity)
    atomic_json(root / "manifest.json", identity | {
        "identity_sha256": identity_hash, "status": "running", "completed": 0,
    })
    completed = 0
    summaries = []
    for sample_id in range(args.starts):
        target = records_root / f"sample_{sample_id:06d}.json"
        if target.exists() and args.resume:
            previous = json.loads(target.read_text())
            if previous.get("status") == "complete":
                completed += 1
                summaries.append(previous["selection"])
                continue
        source, mask = variable_random_source(
            n_atoms[sample_id:sample_id + 1], model.config.vocab_size,
            device, generator,
        )
        record = {
            "version": "relax-aware-unconditional-start-v1",
            "identity_sha256": identity_hash, "sample_id": sample_id,
            "N": int(n_atoms[sample_id]), "unconditional_z0": _state(source),
            "status": "running",
        }
        atomic_json(target, record)
        try:
            problem = RelaxAwareProblem(model, mask, surrogate, hull, novelty)
            with torch.no_grad():
                baseline_terminal = integrate_flow(
                    model, source, mask, steps=args.ode_steps,
                    method="midpoint", freeze_atom=False,
                )
                baseline_components = problem.components(baseline_terminal, config)
            result = RelaxAwareIPOPTSolver(model, problem).solve(source, config)
            with torch.no_grad():
                optimized_components = problem.components(result.terminal, config)
            structure = _hard_structure(model, result.terminal, mask, checkpoint)
            counts = Counter(structure["elements"])
            counts = {element: int(counts[element]) for element in ("Na", "Fe", "P", "O")}
            predicted_hull = float(optimized_components["E_hull_proxy"])
            exact = evaluate_feasibility(
                structure["elements"], structure["frac_coords"],
                structure["lattice"], predicted_hull,
            )
            selection = {
                "sample_id": sample_id,
                "merit": min(row["merit"] for row in result.history),
                "predicted_E_hull_eV_atom": predicted_hull,
                "predicted_final_E_UMA_eV_atom": float(
                    optimized_components["final_E_UMA_prediction"]
                ),
                "surrogate_uncertainty_eV_atom": float(
                    optimized_components["final_E_UMA_uncertainty"]
                ),
                "exact_composition_and_geometry": bool(all(
                    value for name, value in exact["checks"].items()
                    if name != "hull_threshold"
                )),
            }
            record.update({
                "status": "complete", "counts": counts,
                "unconditional": {
                    "terminal": _state(baseline_terminal),
                    "components": {name: float(value) for name, value in baseline_components.items()},
                },
                "optimized": {
                    "solver_status": result.status,
                    "solver_stats": result.solver_stats,
                    "adaptive_weights": result.adaptive_weights,
                    "z0_star": _state(result.z0_star),
                    "terminal": _state(result.terminal),
                    "history": result.history,
                    "components": {name: float(value) for name, value in optimized_components.items()},
                    "exact": {"structure": structure, "feasibility": exact},
                    # Compatibility keys used by the final reference relaxer.
                    "values": {
                        "E_hull": predicted_hull,
                        "E_UMA": float(optimized_components["final_E_UMA_prediction"]),
                    },
                },
                "selection": selection,
            })
            summaries.append(selection)
            completed += 1
        except Exception as error:
            record.update({
                "status": "error", "error": repr(error),
                "traceback": traceback.format_exc(),
            })
        atomic_json(target, record)
        print(json.dumps({
            "sample_id": sample_id, "status": record["status"],
            "selection": record.get("selection"),
        }), flush=True)
    ranked = sorted(
        summaries,
        key=lambda row: (
            not row["exact_composition_and_geometry"],
            row["predicted_E_hull_eV_atom"] + row["surrogate_uncertainty_eV_atom"],
            row["merit"],
        ),
    )
    selected_ids = [row["sample_id"] for row in ranked[:args.final_relax_count]]
    selected_root = root / "selected_for_relaxation"
    selected_records = selected_root / "records"
    selected_records.mkdir(parents=True, exist_ok=True)
    selected_counts = []
    for new_id, original_id in enumerate(selected_ids):
        record = json.loads((records_root / f"sample_{original_id:06d}.json").read_text())
        record["original_sample_id"] = original_id
        record["sample_id"] = new_id
        atomic_json(selected_records / f"sample_{new_id:06d}.json", record)
        selected_counts.append(record["counts"])
    calibrated = build_calibrated_cache(args.phase_cache, selected_counts)
    atomic_json(selected_root / "calibrated_mp2020_hull_cache.json", calibrated, sort_keys=True)
    atomic_json(selected_root / "manifest.json", {
        "version": "relax-aware-selected-for-reference-relax-v1",
        "status": "complete", "selected_ids": selected_ids,
        "count": len(selected_ids), "source_identity_sha256": identity_hash,
    })
    atomic_json(root / "manifest.json", identity | {
        "identity_sha256": identity_hash,
        "status": "complete" if completed == args.starts else "incomplete",
        "completed": completed, "selected_ids": selected_ids,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--relaxation-surrogate", required=True)
    parser.add_argument("--phase-cache", required=True)
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--final-relax-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--optimization-steps", type=int, default=160)
    parser.add_argument("--atom-temperature", type=float, default=0.35)
    parser.add_argument("--hull-temperature", type=float, default=0.03)
    parser.add_argument("--hull-reference-temperature", type=float, default=0.002)
    parser.add_argument("--hull-uncertainty-kappa", type=float, default=1.0)
    parser.add_argument("--hull-scale", type=float, default=1.0)
    parser.add_argument("--hull-weight", type=float, default=4.0)
    parser.add_argument("--novelty-weight", type=float, default=0.10)
    parser.add_argument("--distance-weight", type=float, default=40.0)
    parser.add_argument("--p-coordination-weight", type=float, default=12.0)
    parser.add_argument("--fe-coordination-weight", type=float, default=4.0)
    parser.add_argument("--volume-weight", type=float, default=2.0)
    parser.add_argument("--composition-weight", type=float, default=20.0)
    parser.add_argument("--geometry-survival-weight", type=float, default=2.0)
    parser.add_argument("--source-prior-weight", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

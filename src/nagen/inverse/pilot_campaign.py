"""Run one eight-source paired ShootingFlow pilot configuration."""

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
from ._io import atomic_json as _write
from .constraints import evaluate_feasibility
from .generate import load_model
from .model import CrystalState
from .uma_hull import UMAHullCache
from .multiobjective_shooting import (
    CrystalDesignProblem,
    ShootingConfig,
    ShootingFlowRunner,
    model_sha256,
)
from .novelty import NoveltyIndex
from .sample import decode_terminal, integrate_flow
from .uma_guidance import load_uma_calculator_vjp_factory


ATOMIC_NUMBERS = {"Na": 11, "Fe": 26, "P": 15, "O": 8}
CONFIGS = (
    {"restoration_steps": 40, "guidance_steps": 20, "learning_rate": 0.01, "source_prior": 0.02},
    {"restoration_steps": 40, "guidance_steps": 40, "learning_rate": 0.01, "source_prior": 0.02},
    {"restoration_steps": 60, "guidance_steps": 40, "learning_rate": 0.005, "source_prior": 0.02},
    {"restoration_steps": 60, "guidance_steps": 40, "learning_rate": 0.01, "source_prior": 0.20},
)


def _values(problem: CrystalDesignProblem, terminal: CrystalState) -> dict[str, float]:
    with torch.no_grad():
        return {key: float(value) for key, value in problem.evaluate(terminal).items()}


def _exact(model, checkpoint, mask, terminal, elements, e_hull):
    with torch.no_grad():
        types, frac, lattice = decode_terminal(model, terminal, mask)
    inverse = {
        int(index): element for element, index in checkpoint["element_to_index"].items()
    }
    decoded = [inverse[int(index)] for index in types[0].detach().cpu().tolist()]
    return {
        "fixed_A_preserved": decoded == elements,
        "feasibility": evaluate_feasibility(
            decoded, frac[0].detach().cpu().numpy(),
            lattice[0].detach().cpu().numpy(), e_hull,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-index", type=int, choices=range(4), required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--composition-pool", required=True)
    parser.add_argument("--uma-hull-cache", required=True)
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(args.geometry_checkpoint, device, use_ema=True)
    pool = json.loads(Path(args.composition_pool).read_text())
    selected = pool["unique_compositions"][args.config_index * 8:(args.config_index + 1) * 8]
    if len(selected) != 8:
        raise RuntimeError("pilot requires exactly eight compositions per configuration")
    novelty = NoveltyIndex.load(args.novelty_index, device)
    factory, uma_metadata = load_uma_calculator_vjp_factory(
        args.uma_checkpoint, device=args.device, task_name="omat"
    )
    uma_hull_cache = UMAHullCache(
        args.uma_hull_cache,
        expected_model_sha256=uma_metadata["checkpoint_sha256"],
        expected_task_name=uma_metadata["task_name"],
    )
    prepared = []
    baselines = []
    for local_index, composition in enumerate(selected):
        sample_id = args.config_index * 8 + local_index
        sample_seed = args.seed + sample_id
        source, mask = _source(model, composition, sample_seed, device)
        elements = list(composition["elements"])
        numbers = torch.tensor(
            [ATOMIC_NUMBERS[element] for element in elements], device=device
        )
        problem = CrystalDesignProblem(
            model, mask, factory(numbers),
            uma_hull_cache.get(composition["counts"]), novelty,
        )
        with torch.no_grad():
            terminal = integrate_flow(
                model, source, mask, steps=args.ode_steps,
                method="midpoint", freeze_atom=True,
            )
        values = _values(problem, terminal)
        exact = _exact(
            model, checkpoint, mask, terminal, elements, values["E_hull"]
        )
        prepared.append((sample_id, sample_seed, composition, elements, source, mask, problem))
        baselines.append({
            "sample_id": sample_id, "seed": sample_seed,
            "values": values, "exact": exact,
            "terminal": _state(terminal),
        })
    hull_scale = max(statistics.pstdev(x["values"]["E_hull"] for x in baselines), 1e-3)
    novelty_scale = max(statistics.pstdev(x["values"]["novelty"] for x in baselines), 1e-3)
    config = CONFIGS[args.config_index]
    payload: dict[str, Any] = {
        "version": "shootingflow-pilot-v1",
        "config_index": args.config_index,
        "config": config,
        "ode_steps": args.ode_steps,
        "seed": args.seed,
        "objective_scales": [hull_scale, novelty_scale],
        "model_hash": model_sha256(model),
        "uma": uma_metadata,
        "samples": [],
        "status": "running",
    }
    target = Path(args.out)
    for baseline, prepared_sample in zip(baselines, prepared, strict=True):
        sample_id, sample_seed, composition, elements, source, mask, problem = prepared_sample
        record: dict[str, Any] = {
            "sample_id": sample_id, "seed": sample_seed,
            "counts": composition["counts"], "N": composition["N"],
            "z0": _state(source), "B0": baseline,
        }
        variants = {
            "B1": {"use_hull": False, "use_novelty": False},
            "B2": {"use_hull": False, "use_novelty": True},
            "M0": {"use_hull": True, "use_novelty": True},
        }
        for name, mode in variants.items():
            try:
                result = ShootingFlowRunner(model, problem).run(
                    composition["N"], source.atom, source,
                    ShootingConfig(
                        restoration_steps=config["restoration_steps"],
                        guidance_steps=config["guidance_steps"],
                        learning_rate=config["learning_rate"],
                        source_prior=config["source_prior"],
                        ode_steps=args.ode_steps,
                        objective_scales=(hull_scale, novelty_scale),
                        use_hull=mode["use_hull"],
                        use_novelty=mode["use_novelty"],
                        use_constraints=True,
                    ),
                )
                values = _values(problem, result.terminal)
                record[name] = {
                    "status": result.status, "values": values,
                    "history": result.history,
                    "z0_star": _state(result.z0_star),
                    "terminal": _state(result.terminal),
                    "exact": _exact(
                        model, checkpoint, mask, result.terminal,
                        elements, values["E_hull"],
                    ),
                }
            except Exception as error:  # preserve paired failure evidence
                record[name] = {
                    "status": "error", "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            _write(target, payload | {"samples": payload["samples"] + [record]})
        payload["samples"].append(record)
        _write(target, payload)
        print(json.dumps({
            "config": args.config_index, "sample_id": sample_id,
            "statuses": {name: record[name]["status"] for name in variants},
        }), flush=True)
    payload["status"] = "complete"
    _write(target, payload)
    print(json.dumps({
        "output": str(target), "status": payload["status"],
        "samples": len(payload["samples"]),
    }, indent=2))


if __name__ == "__main__":
    main()

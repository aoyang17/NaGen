"""One-sample end-to-end UMA ShootingFlow acceptance spike."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .constraints import evaluate_feasibility
from .generate import load_model
from ._campaign import state_payload as _state_payload
from .uma_hull import UMAHullCache
from .multiobjective_shooting import (
    CrystalDesignProblem,
    ShootingConfig,
    ShootingFlowRunner,
)
from .novelty import NoveltyIndex
from .sample import decode_terminal, integrate_flow
from .spec import DEFAULT_SPEC
from .uma_guidance import load_uma_calculator_vjp_factory


ATOMIC_NUMBERS = {"Na": 11, "Fe": 26, "P": 15, "O": 8}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--composition-pool", required=True)
    parser.add_argument("--uma-hull-cache", required=True)
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--guidance-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(args.geometry_checkpoint, device, use_ema=True)
    pool = json.loads(Path(args.composition_pool).read_text())
    composition = min(pool["unique_compositions"], key=lambda item: item["N"])
    elements = list(composition["elements"])
    type_indices = torch.tensor(
        composition["type_indices"], dtype=torch.long, device=device
    )[None, :]
    mask = torch.ones_like(type_indices, dtype=torch.bool)
    atom = F.one_hot(
        type_indices - 1, num_classes=model.config.vocab_size
    ).to(dtype=next(model.parameters()).dtype) * 2.0
    generator = torch.Generator(device=device).manual_seed(args.seed)
    source = CrystalState(
        atom,
        torch.rand(1, len(elements), 3, device=device, generator=generator),
        torch.randn(1, 6, device=device, generator=generator),
    )
    novelty = NoveltyIndex.load(args.novelty_index, device)
    factory, uma_metadata = load_uma_calculator_vjp_factory(
        args.uma_checkpoint, device=args.device, task_name="omat"
    )
    uma_hull_cache = UMAHullCache(
        args.uma_hull_cache,
        expected_model_sha256=uma_metadata["checkpoint_sha256"],
        expected_task_name=uma_metadata["task_name"],
    )
    e_ref_uma = uma_hull_cache.get(composition["counts"])
    numbers = torch.tensor(
        [ATOMIC_NUMBERS[element] for element in elements], device=device
    )
    problem = CrystalDesignProblem(
        model, mask, factory(numbers), e_ref_uma, novelty
    )
    with torch.no_grad():
        baseline_terminal = integrate_flow(
            model, source, mask, steps=args.ode_steps,
            method="midpoint", freeze_atom=True,
        )
        baseline_values = {
            key: float(value) for key, value in problem.evaluate(baseline_terminal).items()
        }
    result = ShootingFlowRunner(model, problem).run(
        len(elements), atom, source,
        ShootingConfig(
            guidance_steps=args.guidance_steps,
            learning_rate=0.01,
            ode_steps=args.ode_steps,
        ),
    )
    with torch.no_grad():
        final_values = {
            key: float(value) for key, value in problem.evaluate(result.terminal).items()
        }
        types, frac, lattice = decode_terminal(model, result.terminal, mask)
    index_to_element = {
        int(index): element for element, index in checkpoint["element_to_index"].items()
    }
    decoded_elements = [
        index_to_element[int(index)] for index in types[0].detach().cpu().tolist()
    ]
    feasibility = evaluate_feasibility(
        decoded_elements,
        frac[0].detach().cpu().numpy(),
        lattice[0].detach().cpu().numpy(),
        final_values["E_hull"],
    )
    output = {
        "version": "shootingflow-uma-spike-v1",
        "seed": args.seed,
        "N": len(elements),
        "counts": composition["counts"],
        "fixed_A_preserved": decoded_elements == elements,
        "ode_steps": args.ode_steps,
        "guidance_steps": args.guidance_steps,
        "baseline": baseline_values,
        "optimized": final_values,
        "history": result.history,
        "status": result.status,
        "model_hash": result.model_hash,
        "uma": uma_metadata,
        "source": _state_payload(source),
        "z0_star": _state_payload(result.z0_star),
        "terminal": _state_payload(result.terminal),
        "exact_feasibility": feasibility,
        "policy": {
            "E_hull_role": "UMA_same_scale_convex_hull_objective",
            "final_threshold_eV_atom": DEFAULT_SPEC.hull_max_eV_atom,
        },
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)
    print(json.dumps({
        "output": str(target), "N": len(elements), "status": result.status,
        "fixed_A_preserved": output["fixed_A_preserved"],
        "baseline": baseline_values, "optimized": final_values,
        "exact_feasible": feasibility["feasible"],
    }, indent=2))


if __name__ == "__main__":
    main()

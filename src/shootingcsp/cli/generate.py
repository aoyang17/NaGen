"""Primary ShootingCSP task-oriented generation command.

Physical constraints, objectives, and optimization weights are read from a
versioned ShootingCSP constraint JSON. CLI arguments are reserved for model and
runtime resources such as checkpoints, profiles, devices, seeds, and output.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from shootingcsp.config import (
    conditioning_counts,
    constraint_config_sha256,
    default_constraint_path,
    load_constraint_config,
    load_constraint_mapping,
    optimization_weights,
    to_optimization_spec,
)
from shootingcsp.generation.optimize import optimize_source_with_surrogate
from shootingcsp.generation.source import catastrophic_reason, make_source
from shootingcsp.inverse._io import json_default, sha256_file as sha256
from shootingcsp.inverse.generate import load_model
from shootingcsp.inverse.spec import parse_fe_coordination_options
from shootingcsp.selection.pipeline import Crystal, Conditioning, hard_gates
from shootingcsp.surrogate import SurrogateSpec, load_surrogate


def _surrogate_option(value: str):
    key, separator, raw = value.partition("=")
    if not separator or not key:
        raise ValueError("surrogate option must use key=value")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constraints",
        default=str(default_constraint_path()),
        help="ShootingCSP constraint JSON. Defaults to the packaged Na-Fe-P-O profile.",
    )
    parser.add_argument("--flow", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--ode-steps", type=int, default=48)
    parser.add_argument("--dflow-steps", type=int, default=12)
    parser.add_argument("--dflow-lr", type=float, default=0.03)
    parser.add_argument("--source-mode", choices=("uniform", "polyhedral"), default="polyhedral")
    parser.add_argument("--surrogate", default="uma")
    parser.add_argument("--surrogate-checkpoint")
    parser.add_argument("--surrogate-factory", help="External surrogate factory: module:callable")
    parser.add_argument("--surrogate-option", action="append", default=[], type=_surrogate_option, metavar="KEY=VALUE")
    parser.add_argument("--uma-checkpoint", help="Deprecated alias for --surrogate-checkpoint")
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--device", default="cuda")
    # Explicit overrides remain available for legacy scripts, but JSON is canonical.
    parser.add_argument("--condition", help="Deprecated condition JSON override")
    parser.add_argument("--fe-coordination", type=parse_fe_coordination_options)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.count < 1 or args.dflow_steps < 0 or args.ode_steps < 1:
        raise ValueError("count, dflow-steps, and ode-steps must be valid")
    if args.surrogate.lower() == "uma" and not (args.surrogate_checkpoint or args.uma_checkpoint):
        raise ValueError("UMA surrogate requires --surrogate-checkpoint or --uma-checkpoint")

    constraint_path = None if args.constraints == str(default_constraint_path()) else args.constraints
    input_constraint = load_constraint_mapping(constraint_path)
    config = load_constraint_config(constraint_path)
    spec = to_optimization_spec(config)
    if args.fe_coordination is not None:
        spec = spec.__class__(
            **{**spec.__dict__, "fe_coordination_options": args.fe_coordination}
        )
    weights = optimization_weights(config)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "constraints.input.json").write_text(
        json.dumps(input_constraint, indent=2, ensure_ascii=False) + "\n"
    )
    resolved_constraints = config.as_dict()
    (out / "constraints.resolved.json").write_text(
        json.dumps(resolved_constraints, indent=2, ensure_ascii=False) + "\n"
    )
    config_hash = constraint_config_sha256(config)
    (out / "constraints.sha256").write_text(config_hash + "\n")

    if args.condition:
        condition_doc = json.loads(Path(args.condition).read_text())
        condition = Conditioning(**condition_doc)
    else:
        condition = Conditioning(conditioning_counts(config))
    elements = [
        element
        for element in ("Na", "Fe", "P", "O")
        for _ in range(int(condition.counts.get(element, 0)))
    ]
    if not elements:
        raise ValueError("constraint composition produced an empty cell")

    device = torch.device(args.device)
    model, checkpoint = load_model(args.flow, device, True)
    if checkpoint.get("training_mode") not in ("geometry", "fixed_NA_geometry"):
        raise ValueError("flow checkpoint must be geometry-only")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        type_indices = torch.tensor(
            [checkpoint["element_to_index"][element] for element in elements],
            device=device,
        )
    except KeyError as error:
        raise ValueError(f"flow vocabulary lacks condition element {error.args[0]}") from error
    if any(int(index) > 4 for index in type_indices):
        raise ValueError("current phosphate D-Flow terms require core Na/Fe/P/O indices")
    core_probabilities = F.one_hot(type_indices[None] - 1, num_classes=4).to(
        device=device, dtype=torch.float32
    )

    surrogate = load_surrogate(
        SurrogateSpec(
            backend=args.surrogate,
            checkpoint=args.surrogate_checkpoint or args.uma_checkpoint,
            device=args.device,
            task_name=args.task_name,
            options=dict(args.surrogate_option),
            factory=args.surrogate_factory,
        )
    )
    atomic_numbers = torch.tensor(
        [{"Na": 11, "Fe": 26, "P": 15, "O": 8}[element] for element in elements],
        device=device,
    )
    energy_fn = surrogate.guidance_energy(atomic_numbers)
    profile = json.loads(Path(args.profile).read_text())

    counts = Counter()
    records = []
    with (out / "generated_all.jsonl").open("x") as all_handle, (
        out / "generated_safe.jsonl"
    ).open("x") as safe_handle:
        for index in range(args.count):
            source_seed = args.seed + index
            source, mask = make_source(model, type_indices, source_seed, args.source_mode)
            final_source, frac, lattice, history = optimize_source_with_surrogate(
                model,
                source,
                mask,
                energy_fn,
                core_probabilities,
                args.dflow_steps,
                args.dflow_lr,
                args.ode_steps,
                fe_coordination_options=spec.fe_coordination_options,
                p_coordination_options=spec.p_coordination_options,
                spec=spec,
                weights=weights,
            )
            crystal = Crystal(
                f"shootingcsp_seed{args.seed}_{index:04d}",
                elements,
                frac[0].detach().cpu().numpy(),
                lattice[0].detach().cpu().numpy(),
                condition.family,
            )
            reason = catastrophic_reason(crystal, config.source_precheck)
            gates = hard_gates(
                crystal,
                condition,
                profile,
                forces=None,
                motif_backend="native",
                constraint_config=config,
                enforce_force=False,
                enforce_energy=False,
            )
            row = {
                "material_id": crystal.id,
                "A": elements,
                "X_frac": crystal.frac,
                "L_matrix": crystal.lattice,
                "family": condition.family,
                "constraint_config_sha256": config_hash,
                "generation": {
                    "method": "fixed_NA_geometry_flow_surrogate_DFlow_source_optimization",
                    "surrogate_backend": args.surrogate,
                    "source_seed": source_seed,
                    "ode_steps": args.ode_steps,
                    "dflow_steps": args.dflow_steps,
                    "dflow_lr": args.dflow_lr,
                    "source_mode": args.source_mode,
                    "p_coordination_options": list(spec.p_coordination_options),
                    "fe_coordination_options": list(spec.fe_coordination_options),
                    "generated_variables": "all Na/Fe/P/O coordinates and lattice",
                    "copied_parent_coordinates": False,
                    "source_final_frac": final_source.frac.cpu().numpy(),
                    "source_final_lattice": final_source.lattice.cpu().numpy(),
                    "optimization_history": history,
                },
                "pre_relax": {
                    "catastrophic_reason": reason,
                    "gates": gates,
                    "safe_for_relaxation": reason is None and gates["passed"],
                },
            }
            all_handle.write(json.dumps(row, default=json_default, allow_nan=False) + "\n")
            all_handle.flush()
            records.append(row)
            counts["attempted"] += 1
            if row["pre_relax"]["safe_for_relaxation"]:
                safe_handle.write(json.dumps(row, default=json_default, allow_nan=False) + "\n")
                safe_handle.flush()
                counts["safe_for_relaxation"] += 1
            elif reason is not None:
                counts[reason] += 1
            else:
                counts["failed_pre_relaxation_gate"] += 1
            print(json.dumps({"generated": index + 1, "counts": counts}), flush=True)

    manifest = {
        "status": "completed",
        "configuration": vars(args),
        "constraints": {
            "path": str(constraint_path) if constraint_path else None,
            "sha256": config_hash,
            "name": config.name,
            "schema_version": config.schema_version,
            "resolved": resolved_constraints,
        },
        "counts": counts,
        "conditioning": {
            "N": len(elements),
            "counts": condition.counts,
            "family": condition.family,
        },
        "flow_sha256": sha256(args.flow),
        "flow_training": checkpoint.get("training_objective"),
        "surrogate": surrogate.provenance,
        "profile_sha256": sha256(args.profile),
        "note": "Hull and finalized force gates remain post-relaxation checks in the full campaign.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default, ensure_ascii=False) + "\n"
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()

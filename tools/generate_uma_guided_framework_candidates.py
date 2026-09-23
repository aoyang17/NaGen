"""Generate fixed-condition Flow candidates with UMA-guided source optimization.

The optimizer changes only the random X/L source of a frozen geometry Flow.
It never runs an ASE optimizer inside an inference step; complete UMA
relaxation is a separate post-generation stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from nagen.inverse.generate import load_model
from nagen.inverse.guidance import _source_prior, soft_constraint_terms
from nagen.inverse.model import CrystalState
from nagen.inverse.sample import decode_terminal, integrate_flow
from nagen.inverse.uma_guidance import load_uma_calculator_vjp_factory
from nagen.inverse._io import json_default, sha256_file as sha256
from nagen.inverse.spec import (
    DEFAULT_FE_COORDINATION_OPTIONS,
    parse_fe_coordination_options,
)
from nagen.selection.pipeline import Crystal, Conditioning, hard_gates, neighbors


def catastrophic_reason(crystal: Crystal) -> str | None:
    volume_per_atom = abs(np.linalg.det(crystal.lattice)) / len(crystal.elements)
    if not np.isfinite(crystal.frac).all() or not np.isfinite(crystal.lattice).all():
        return "nonfinite"
    if not 2 <= volume_per_atom <= 60 or np.linalg.cond(crystal.lattice) > 40:
        return "unsafe_cell"
    if any(distance < .4 for site in range(len(crystal.elements))
           for _, _, distance in neighbors(crystal, site, .4)):
        return "overlap_below_0.4A"
    return None


def make_source(model, type_indices, seed: int, source_mode: str):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    count = len(type_indices)
    mask = torch.ones(1, count, dtype=torch.bool, device=device)
    fixed_atom = F.one_hot(type_indices[None] - 1, num_classes=model.config.vocab_size).to(
        device=device, dtype=next(model.parameters()).dtype
    ) * 2.0
    lattice = torch.randn(1, 6, device=device, generator=generator)
    if source_mode == "uniform":
        frac = torch.rand(1, count, 3, device=device, generator=generator)
    elif source_mode == "polyhedral":
        from nagen.inverse.polyhedral_prior import polyhedral_source_frac
        source_matrix = model._lattice_matrix(lattice, torch.tensor([count], device=device))
        frac = polyhedral_source_frac(type_indices[None], mask, source_matrix, generator=generator)
    else:
        raise ValueError(source_mode)
    return CrystalState(fixed_atom, frac, lattice), mask


def optimize_source_with_uma(
    model,
    source,
    mask,
    energy_fn,
    core_probabilities,
    steps,
    lr,
    ode_steps,
    fe_coordination_options=DEFAULT_FE_COORDINATION_OPTIONS,
):
    """First-order D-Flow optimization with UMA VJP and smooth geometry terms."""
    frac = source.frac.detach().clone().requires_grad_(True)
    lattice = source.lattice.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam((frac, lattice), lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state = CrystalState(source.atom, frac, lattice)
        terminal = integrate_flow(model, state, mask, steps=ode_steps, method="midpoint", freeze_atom=True)
        _, final_frac, final_lattice = decode_terminal(model, terminal, mask)
        energy = energy_fn(final_frac, final_lattice).mean()
        soft = soft_constraint_terms(
            model,
            terminal,
            mask,
            probabilities_override=core_probabilities,
            fe_coordination_options=fe_coordination_options,
        )
        prior = _source_prior(state, mask)
        loss = (.10 * energy + 20.0 * soft["distance"] + 3.0 * soft["p_coordination"]
                + 3.0 * soft["fe_coordination"] + 2.0 * soft["volume"] + .02 * prior)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((frac, lattice), 5.0)
        optimizer.step()
        with torch.no_grad():
            frac.remainder_(1.0)
        history.append({"step": step + 1, "loss": float(loss.detach()),
                        "uma_energy_eV_atom": float(energy.detach()),
                        **{f"soft_{name}": float(value.detach()) for name, value in soft.items()},
                        "source_prior": float(prior.detach())})
    final_source = CrystalState(source.atom, frac.detach(), lattice.detach())
    with torch.no_grad():
        terminal = integrate_flow(model, final_source, mask, steps=ode_steps, method="midpoint", freeze_atom=True)
        _, final_frac, final_lattice = decode_terminal(model, terminal, mask)
    return final_source, final_frac, final_lattice, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", required=True); parser.add_argument("--condition", required=True)
    parser.add_argument("--profile", required=True); parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=271828); parser.add_argument("--ode-steps", type=int, default=48)
    parser.add_argument("--dflow-steps", type=int, default=12); parser.add_argument("--dflow-lr", type=float, default=.03)
    parser.add_argument("--source-mode", choices=("uniform", "polyhedral"), default="polyhedral")
    parser.add_argument(
        "--fe-coordination",
        type=parse_fe_coordination_options,
        default=DEFAULT_FE_COORDINATION_OPTIONS,
        help="Allowed Fe-O coordination numbers, e.g. 4, 4,5, or 4,5,6.",
    )
    parser.add_argument("--task-name", default="omat"); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.count < 1 or args.dflow_steps < 0 or args.ode_steps < 1:
        parser.error("invalid positive generation budget")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.flow, device, True)
    if checkpoint.get("training_mode") not in ("geometry", "fixed_NA_geometry"):
        raise ValueError("flow checkpoint must be geometry-only")
    for parameter in model.parameters(): parameter.requires_grad_(False)
    condition = Conditioning(**json.loads(Path(args.condition).read_text()))
    elements = [element for element in ("Na", "Fe", "P", "O") for _ in range(condition.counts.get(element, 0))]
    if not elements: raise ValueError("condition has no supported atoms")
    try: type_indices = torch.tensor([checkpoint["element_to_index"][element] for element in elements], device=device)
    except KeyError as error: raise ValueError(f"flow vocabulary lacks condition element {error.args[0]}") from error
    # Existing soft phosphate terms deliberately operate only on the fixed core
    # vocabulary, whose indices are required to remain Na/Fe/P/O = 1/2/3/4.
    if any(int(index) > 4 for index in type_indices):
        raise ValueError("current phosphate D-Flow terms require core Na/Fe/P/O indices")
    core_probabilities = F.one_hot(type_indices[None] - 1, num_classes=4).to(device=device, dtype=torch.float32)
    factory, uma_provenance = load_uma_calculator_vjp_factory(args.uma_checkpoint, args.device, args.task_name)
    atomic_numbers = torch.tensor([{ "Na": 11, "Fe": 26, "P": 15, "O": 8 }[element] for element in elements], device=device)
    energy_fn = factory(atomic_numbers)
    profile = json.loads(Path(args.profile).read_text()); counts = Counter(); records = []
    with (out / "generated_all.jsonl").open("x") as all_handle, (out / "generated_safe.jsonl").open("x") as safe_handle:
        for index in range(args.count):
            source_seed = args.seed + index
            source, mask = make_source(model, type_indices, source_seed, args.source_mode)
            final_source, frac, lattice, history = optimize_source_with_uma(
                model,
                source,
                mask,
                energy_fn,
                core_probabilities,
                args.dflow_steps,
                args.dflow_lr,
                args.ode_steps,
                fe_coordination_options=args.fe_coordination,
            )
            crystal = Crystal(f"uma_dflow_seed{args.seed}_{index:04d}", elements,
                              frac[0].detach().cpu().numpy(), lattice[0].detach().cpu().numpy(), condition.family)
            reason = catastrophic_reason(crystal)
            gates = hard_gates(
                crystal,
                condition,
                profile,
                forces=None,
                motif_backend="native",
                fe_coordination_options=args.fe_coordination,
            )
            row = {"material_id": crystal.id, "A": elements, "X_frac": crystal.frac, "L_matrix": crystal.lattice,
                   "family": condition.family,
                   "generation": {"method": "fixed_NA_geometry_flow_UMA_DFlow_source_optimization",
                     "source_seed": source_seed, "ode_steps": args.ode_steps, "dflow_steps": args.dflow_steps,
                     "dflow_lr": args.dflow_lr, "source_mode": args.source_mode,
                     "fe_coordination_options": list(args.fe_coordination),
                     "generated_variables": "all Na/Fe/P/O coordinates and lattice", "copied_parent_coordinates": False,
                     "source_final_frac": final_source.frac.cpu().numpy(), "source_final_lattice": final_source.lattice.cpu().numpy(),
                     "optimization_history": history},
                   "pre_relax": {"catastrophic_reason": reason, "gates": gates}}
            all_handle.write(json.dumps(row, default=json_default, allow_nan=False) + "\n"); all_handle.flush()
            records.append(row); counts["attempted"] += 1
            if reason is None:
                safe_handle.write(json.dumps(row, default=json_default, allow_nan=False) + "\n"); safe_handle.flush(); counts["safe_for_UMA_relax"] += 1
            else: counts[reason] += 1
            print(json.dumps({"generated": index + 1, "counts": counts}), flush=True)
    manifest = {"status": "completed", "configuration": vars(args), "counts": counts,
                "fe_coordination_options": list(args.fe_coordination),
                "conditioning": {"N": len(elements), "counts": condition.counts, "family": condition.family},
                "flow_sha256": sha256(args.flow), "flow_training": checkpoint.get("training_objective"),
                "uma": uma_provenance, "profile_sha256": sha256(args.profile),
                "note": "UMA enters only through first-order source-space guidance. Final UMA relaxation and all hard gates remain independent stages."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default) + "\n")


if __name__ == "__main__":
    main()

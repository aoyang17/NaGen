"""Unconditional sampling and auditable export for the complete crystal flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Structure
from torch.nn import functional as F

from .constraints import composition_metrics, evaluate_feasibility
from .guidance import optimize_source, optimize_terminal
from .model import CrystalState, CrystalVectorField
from .novelty import NoveltyIndex, crystal_descriptor
from .sample import decode_terminal, integrate_flow
from .spec import (
    DEFAULT_FE_COORDINATION_OPTIONS,
    DEFAULT_SPEC,
    OptimizationSpec,
    parse_fe_coordination_options,
)


def load_model(path: str, device: torch.device, use_ema: bool = True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CrystalVectorField.from_checkpoint(checkpoint)
    if use_ema and "ema_model" in checkpoint:
        model.load_state_dict(checkpoint["ema_model"])
    model.to(device).eval()
    return model, checkpoint


def sample_atom_counts(
    histogram: torch.Tensor, count: int, generator: torch.Generator
) -> torch.Tensor:
    weights = histogram.float().to(generator.device).clone()
    weights[:1] = 0.0
    return torch.multinomial(weights, count, replacement=True, generator=generator)


def variable_random_source(
    n_atoms: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[CrystalState, torch.Tensor]:
    batch = len(n_atoms)
    n_max = int(n_atoms.max())
    indices = torch.arange(n_max, device=device)
    mask = indices[None, :] < n_atoms.to(device)[:, None]
    atom = torch.randn(
        batch, n_max, vocab_size, device=device, generator=generator
    ) * mask.unsqueeze(-1)
    frac = torch.rand(batch, n_max, 3, device=device, generator=generator)
    lattice = torch.randn(batch, 6, device=device, generator=generator)
    return CrystalState(atom, frac, lattice), mask


def _composition_passes(
    generated_types: torch.Tensor,
    mask: torch.Tensor,
    index_to_element: dict[int, str],
) -> bool:
    elements = [
        index_to_element[int(index)]
        for index in generated_types[mask].detach().cpu().tolist()
    ]
    composition = composition_metrics(elements)
    if not (
        composition["domain_ok"]
        and composition["charge_ok"]
        and composition["capacity_ok"]
    ):
        return False
    # Exact CN=4 cannot be met when a generated composition contains a P/TM
    # center but fewer than four oxygen atoms.  This is a mathematical
    # feasibility precheck, not a family or formula template.
    has_p = "P" in elements
    has_fe = "Fe" in elements
    required_oxygen = 6 if has_fe else (4 if has_p else 0)
    return elements.count("O") >= required_oxygen


def sample_feasible_compositions(
    model: CrystalVectorField,
    checkpoint: dict[str, Any],
    count: int,
    generator: torch.Generator,
    ode_steps: int,
    integrator: str,
    pool_factor: int = 6,
    max_rounds: int = 20,
    max_atoms: int | None = None,
    proposal_batch_limit: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection-control p(A,N) using only exact composition constraints."""
    device = next(model.parameters()).device
    index_to_element = {
        int(index): element for element, index in checkpoint["element_to_index"].items()
    }
    accepted: list[torch.Tensor] = []
    for _ in range(max_rounds):
        remaining = count - len(accepted)
        if remaining <= 0:
            break
        proposal_count = min(
            proposal_batch_limit, max(32, remaining * pool_factor)
        )
        n_atoms = sample_atom_counts(
            checkpoint["n_histogram"], proposal_count, generator
        ).to(device)
        source, proposal_mask = variable_random_source(
            n_atoms, model.config.vocab_size, device, generator
        )
        with torch.no_grad():
            terminal = integrate_flow(
                model, source, proposal_mask, steps=ode_steps, method=integrator
            )
            proposal_types = terminal.atom.argmax(dim=-1) + 1
            proposal_types = proposal_types.masked_fill(~proposal_mask, 0)
        for row in range(proposal_count):
            if max_atoms is not None and int(proposal_mask[row].sum()) > max_atoms:
                continue
            if _composition_passes(
                proposal_types[row], proposal_mask[row], index_to_element
            ):
                accepted.append(proposal_types[row, proposal_mask[row]].detach().clone())
                if len(accepted) == count:
                    break
    if len(accepted) < count:
        raise RuntimeError(
            f"only {len(accepted)} of {count} requested compositions passed exact "
            f"charge/capacity prechecks after {max_rounds} rounds"
        )
    n_atoms = torch.tensor([len(types) for types in accepted], device=device)
    n_max = int(n_atoms.max())
    mask = torch.arange(n_max, device=device)[None, :] < n_atoms[:, None]
    generated_types = torch.zeros(count, n_max, dtype=torch.long, device=device)
    for row, types in enumerate(accepted):
        generated_types[row, : len(types)] = types
    return generated_types, mask


def sample_unconditional_compositions(
    model: CrystalVectorField,
    checkpoint: dict[str, Any],
    count: int,
    generator: torch.Generator,
    ode_steps: int,
    integrator: str,
    max_atoms: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``(N, A)`` from the learned flow without constraint rejection."""
    device = next(model.parameters()).device
    histogram = checkpoint["n_histogram"].clone()
    if max_atoms is not None and max_atoms + 1 < histogram.numel():
        histogram[max_atoms + 1 :] = 0
    n_atoms = sample_atom_counts(histogram, count, generator).to(device)
    source, mask = variable_random_source(
        n_atoms, model.config.vocab_size, device, generator
    )
    with torch.no_grad():
        terminal = integrate_flow(
            model, source, mask, steps=ode_steps, method=integrator
        )
        generated_types = terminal.atom.argmax(dim=-1) + 1
        generated_types = generated_types.masked_fill(~mask, 0)
    return generated_types, mask


def generate_batch(
    model: CrystalVectorField,
    checkpoint: dict[str, Any],
    count: int,
    seed: int,
    ode_steps: int,
    integrator: str,
    guidance_steps: int = 0,
    guidance_lr: float = 0.03,
    geometry_model: CrystalVectorField | None = None,
    terminal_steps: int = 0,
    terminal_lr: float = 0.025,
    composition_filter: bool = False,
    composition_pool_factor: int = 6,
    composition_max_rounds: int = 20,
    max_atoms: int | None = None,
    fe_coordination_options: tuple[int, ...] = DEFAULT_FE_COORDINATION_OPTIONS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    if composition_filter:
        generated_types, mask = sample_feasible_compositions(
            model,
            checkpoint,
            count,
            generator,
            ode_steps,
            integrator,
            composition_pool_factor,
            composition_max_rounds,
            max_atoms=max_atoms,
        )
        n_atoms = mask.sum(dim=1)
        source = None
    else:
        n_atoms = sample_atom_counts(checkpoint["n_histogram"], count, generator).to(device)
        source, mask = variable_random_source(
            n_atoms, model.config.vocab_size, device, generator
        )
        # Stage 1 samples A from p(A | N).
        with torch.no_grad():
            composition_terminal = integrate_flow(
                model, source, mask, steps=ode_steps, method=integrator
            )
            generated_types = composition_terminal.atom.argmax(dim=-1) + 1
            generated_types = generated_types.masked_fill(~mask, 0)

    active_model = model
    freeze_atom = False
    if geometry_model is not None:
        # Stage 2 samples (X, L) from p(X, L | N, A). The composition is
        # generated, not copied from a family/parent template.
        active_model = geometry_model
        conditioned_atom = F.one_hot(
            (generated_types - 1).clamp_min(0),
            num_classes=geometry_model.config.vocab_size,
        ).to(
            device=device,
            dtype=next(geometry_model.parameters()).dtype,
        ) * 2.0
        conditioned_atom = conditioned_atom * mask.unsqueeze(-1)
        source = CrystalState(
            conditioned_atom,
            torch.rand(
                mask.shape[0], mask.shape[1], 3, device=device, generator=generator
            ),
            torch.randn(mask.shape[0], 6, device=device, generator=generator),
        )
        freeze_atom = True
    elif source is None:
        raise ValueError("composition filtering requires --geometry-checkpoint")
    if guidance_steps > 0:
        terminal, _ = optimize_source(
            active_model,
            source,
            mask,
            guidance_steps=guidance_steps,
            learning_rate=guidance_lr,
            ode_steps=ode_steps,
            integrator=integrator,
            freeze_atom=freeze_atom,
            fe_coordination_options=fe_coordination_options,
        )
    else:
        with torch.no_grad():
            terminal = integrate_flow(
                active_model,
                source,
                mask,
                steps=ode_steps,
                method=integrator,
                freeze_atom=freeze_atom,
            )
    if terminal_steps > 0:
        terminal, _ = optimize_terminal(
            active_model,
            terminal,
            mask,
            optimization_steps=terminal_steps,
            learning_rate=terminal_lr,
            fe_coordination_options=fe_coordination_options,
        )
    with torch.no_grad():
        types, frac, lattice = decode_terminal(active_model, terminal, mask)
    return types, frac, lattice, mask


def generate_conditioned_geometry_batch(
    geometry_model: CrystalVectorField,
    type_sequences: list[torch.Tensor],
    seed: int,
    ode_steps: int = 8,
    integrator: str = "midpoint",
    terminal_steps: int = 400,
    terminal_lr: float = 0.02,
    source_mode: str = "uniform",
    end_time: float = 1.0,
    fe_coordination_options: tuple[int, ...] = DEFAULT_FE_COORDINATION_OPTIONS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample X,L from p(X,L|N,A) for previously DFlow-generated N,A.

    This is controlled inference in the same N,A,X,L design space. No parent
    structure, family label, coordinates, or lattice are copied; only selected
    generated compositions are held fixed while fresh X and L are sampled.
    """
    if not type_sequences:
        raise ValueError("at least one type sequence is required")
    device = next(geometry_model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    lengths = torch.tensor(
        [len(sequence) for sequence in type_sequences], device=device
    )
    n_max = int(lengths.max())
    mask = torch.arange(n_max, device=device)[None, :] < lengths[:, None]
    generated_types = torch.zeros(
        len(type_sequences), n_max, dtype=torch.long, device=device
    )
    for row, sequence in enumerate(type_sequences):
        generated_types[row, : len(sequence)] = sequence.to(device)
    conditioned_atom = F.one_hot(
        (generated_types - 1).clamp_min(0),
        num_classes=geometry_model.config.vocab_size,
    ).to(dtype=next(geometry_model.parameters()).dtype)
    conditioned_atom = conditioned_atom * 2.0 * mask.unsqueeze(-1)
    source_lattice=torch.randn(
            len(type_sequences), 6, device=device, generator=generator
        )
    if source_mode == 'uniform':
        source_frac=torch.rand(
            len(type_sequences), n_max, 3, device=device, generator=generator
        )
    elif source_mode == 'polyhedral':
        from .polyhedral_prior import polyhedral_source_frac
        source_matrix=geometry_model._lattice_matrix(source_lattice,lengths)
        # The prior uses the canonical Na/Fe/P/O index convention recorded by
        # NaGen packed datasets.
        source_frac=polyhedral_source_frac(generated_types,mask,source_matrix,generator=generator)
    else:
        raise ValueError(f'unsupported source_mode: {source_mode}')
    source = CrystalState(
        conditioned_atom,
        source_frac,
        source_lattice,
    )
    with torch.no_grad():
        terminal = integrate_flow(
            geometry_model,
            source,
            mask,
            steps=ode_steps,
            method=integrator,
            freeze_atom=True,
            end_time=end_time,
        )
    if terminal_steps > 0:
        terminal, _ = optimize_terminal(
            geometry_model,
            terminal,
            mask,
            optimization_steps=terminal_steps,
            learning_rate=terminal_lr,
            fe_coordination_options=fe_coordination_options,
        )
    with torch.no_grad():
        types, frac, lattice = decode_terminal(
            geometry_model, terminal, mask
        )
    return types, frac, lattice, mask


def export_samples(
    model: CrystalVectorField,
    checkpoint: dict[str, Any],
    novelty: NoveltyIndex,
    count: int,
    seed: int,
    ode_steps: int,
    integrator: str,
    output_path: str,
    guidance_steps: int = 0,
    guidance_lr: float = 0.03,
    geometry_model: CrystalVectorField | None = None,
    terminal_steps: int = 0,
    terminal_lr: float = 0.025,
    composition_filter: bool = False,
    composition_pool_factor: int = 6,
    max_atoms: int | None = None,
    fe_coordination_options: tuple[int, ...] = DEFAULT_FE_COORDINATION_OPTIONS,
) -> list[dict[str, Any]]:
    types, frac, lattice, mask = generate_batch(
        model,
        checkpoint,
        count,
        seed,
        ode_steps,
        integrator,
        guidance_steps,
        guidance_lr,
        geometry_model,
        terminal_steps,
        terminal_lr,
        composition_filter,
        composition_pool_factor,
        max_atoms=max_atoms,
        fe_coordination_options=fe_coordination_options,
    )
    descriptor = crystal_descriptor(types, frac, lattice, mask, novelty.config)
    novelty_score, nearest_index = novelty.score(descriptor)
    index_to_element = {
        int(index): element for element, index in checkpoint["element_to_index"].items()
    }
    records: list[dict[str, Any]] = []
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cif_directory = output.with_suffix("")
    cif_directory.mkdir(parents=True, exist_ok=True)
    for row in range(count):
        n_atoms = int(mask[row].sum())
        element_indices = types[row, :n_atoms].detach().cpu().tolist()
        elements = [index_to_element[index] for index in element_indices]
        coordinates = frac[row, :n_atoms].detach().cpu().numpy()
        cell = lattice[row].detach().cpu().numpy()
        structure = Structure(cell, elements, coordinates, coords_are_cartesian=False)
        feasibility = evaluate_feasibility(
            elements,
            coordinates,
            cell,
            None,
            spec=OptimizationSpec(fe_coordination_options=fe_coordination_options),
        )
        nearest = int(nearest_index[row])
        record = {
            "candidate_id": f"NaGen-flow-{seed}-{row:05d}",
            "seed": seed,
            "N": n_atoms,
            "formula": structure.composition.formula,
            "reduced_formula": structure.composition.reduced_formula,
            "structure": structure.as_dict(),
            "novelty_score": float(novelty_score[row]),
            "nearest_reference_index": nearest,
            "nearest_reference_id": novelty.ids[nearest],
            "nearest_reference_formula": novelty.formulas[nearest],
            "thermodynamic_stability_score": None,
            "feasibility": feasibility,
            "provenance": {
                "generator": "unconditional-naxl-dflow",
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "dataset_source_hash": checkpoint["dataset_source_hash"],
                "ode_steps": ode_steps,
                "integrator": integrator,
                "ema": "ema_model" in checkpoint,
                "guidance_steps": guidance_steps,
                "guidance_lr": guidance_lr,
                "geometry_model": geometry_model is not None,
                "terminal_constraint_steps": terminal_steps,
                "terminal_constraint_lr": terminal_lr,
                "composition_exact_filter": composition_filter,
                "composition_pool_factor": composition_pool_factor,
                "inference_max_atoms": max_atoms,
                "fe_coordination_options": list(fe_coordination_options),
            },
        }
        records.append(record)
        structure.to(filename=str(cif_directory / f"candidate_{row:05d}.cif"))
    with open(output, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--geometry-checkpoint")
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--ode-steps", type=int, default=24)
    parser.add_argument("--integrator", choices=("euler", "midpoint"), default="midpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--guidance-steps", type=int, default=0)
    parser.add_argument("--guidance-lr", type=float, default=0.03)
    parser.add_argument("--terminal-steps", type=int, default=0)
    parser.add_argument("--terminal-lr", type=float, default=0.025)
    parser.add_argument("--composition-filter", action="store_true")
    parser.add_argument("--composition-pool-factor", type=int, default=6)
    parser.add_argument(
        "--fe-coordination",
        type=parse_fe_coordination_options,
        default=DEFAULT_FE_COORDINATION_OPTIONS,
        help="Allowed Fe-O coordination numbers, e.g. 4, 4,5, or 4,5,6.",
    )
    parser.add_argument("--max-atoms", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device, not args.no_ema)
    geometry_model = None
    if args.geometry_checkpoint:
        geometry_model, _ = load_model(
            args.geometry_checkpoint, device, not args.no_ema
        )
    novelty = NoveltyIndex.load(args.novelty_index, device)
    records = export_samples(
        model,
        checkpoint,
        novelty,
        args.num,
        args.seed,
        args.ode_steps,
        args.integrator,
        args.out,
        args.guidance_steps,
        args.guidance_lr,
        geometry_model,
        args.terminal_steps,
        args.terminal_lr,
        args.composition_filter,
        args.composition_pool_factor,
        args.max_atoms,
        fe_coordination_options=args.fe_coordination,
    )
    print(
        json.dumps(
            {
                "generated": len(records),
                "fully_feasible": sum(r["feasibility"]["feasible"] for r in records),
                "mean_novelty": float(np.mean([r["novelty_score"] for r in records])),
                "output": args.out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Resumable constrained-generation campaign for analytically feasible crystals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .constraints import evaluate_feasibility
from .generate import generate_batch, load_model
from .novelty import NoveltyIndex, crystal_descriptor


def analytical_feasible(feasibility: dict[str, Any]) -> bool:
    """All immutable hard constraints except the separately evaluated hull."""
    return all(
        passed
        for name, passed in feasibility["checks"].items()
        if name != "thermodynamic_stability"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--max-proposals", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--terminal-steps", type=int, default=250)
    parser.add_argument("--terminal-lr", type=float, default=0.025)
    parser.add_argument("--composition-pool-factor", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=80)
    parser.add_argument("--min-novelty", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    geometry_model, geometry_checkpoint = load_model(
        args.geometry_checkpoint, device
    )
    novelty = NoveltyIndex.load(args.novelty_index, device)
    index_to_element = {
        int(index): element for element, index in checkpoint["element_to_index"].items()
    }

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = output_dir / f"cif_shard_{args.shard:02d}"
    cif_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / f"accepted_shard_{args.shard:02d}.jsonl"
    rejected_path = output_dir / f"rejected_shard_{args.shard:02d}.jsonl"
    summary_path = output_dir / f"summary_shard_{args.shard:02d}.json"

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    accepted_by_formula: dict[str, list[Structure]] = defaultdict(list)
    accepted_count = 0
    proposal_count = 0
    duplicate_count = 0
    rejection_counts: dict[str, int] = defaultdict(int)

    with accepted_path.open("w") as accepted_handle, rejected_path.open(
        "w"
    ) as rejected_handle:
        batch_index = 0
        while accepted_count < args.target and proposal_count < args.max_proposals:
            current_batch = min(
                args.batch_size, args.max_proposals - proposal_count
            )
            batch_seed = args.seed + args.shard * 1_000_000 + batch_index
            types, frac, lattice, mask = generate_batch(
                model,
                checkpoint,
                current_batch,
                batch_seed,
                args.ode_steps,
                "midpoint",
                geometry_model=geometry_model,
                terminal_steps=args.terminal_steps,
                terminal_lr=args.terminal_lr,
                composition_filter=True,
                composition_pool_factor=args.composition_pool_factor,
                max_atoms=args.max_atoms,
            )
            descriptors = crystal_descriptor(
                types, frac, lattice, mask, novelty.config
            )
            novelty_scores, nearest_indices = novelty.score(descriptors)
            for row in range(current_batch):
                if accepted_count >= args.target:
                    break
                proposal_count += 1
                n_atoms = int(mask[row].sum())
                elements = [
                    index_to_element[int(index)]
                    for index in types[row, :n_atoms].detach().cpu().tolist()
                ]
                coordinates = frac[row, :n_atoms].detach().cpu().numpy()
                cell = lattice[row].detach().cpu().numpy()
                structure = Structure(
                    cell, elements, coordinates, coords_are_cartesian=False
                )
                feasibility = evaluate_feasibility(elements, coordinates, cell, None)
                analytic_ok = analytical_feasible(feasibility)
                novelty_score = float(novelty_scores[row])
                nearest = int(nearest_indices[row])
                formula = structure.composition.reduced_formula
                duplicate = False
                if analytic_ok and novelty_score >= args.min_novelty:
                    duplicate = any(
                        matcher.fit(structure, previous)
                        for previous in accepted_by_formula[formula]
                    )
                if duplicate:
                    duplicate_count += 1
                accepted = analytic_ok and novelty_score >= args.min_novelty and not duplicate
                record = {
                    "candidate_id": (
                        f"NaGen-DFlow-g{args.seed}-s{args.shard:02d}"
                        f"-p{proposal_count:05d}"
                    ),
                    "seed": batch_seed,
                    "N": n_atoms,
                    "formula": structure.composition.formula,
                    "reduced_formula": formula,
                    "structure": structure.as_dict(),
                    "novelty_score": novelty_score,
                    "nearest_reference_index": nearest,
                    "nearest_reference_id": novelty.ids[nearest],
                    "nearest_reference_formula": novelty.formulas[nearest],
                    "thermodynamic_stability_score": None,
                    "thermodynamic_status": "pending_independent_evaluation",
                    "analytical_feasible": analytic_ok,
                    "feasibility": feasibility,
                    "provenance": {
                        "generator": "unconditional-naxl-dflow+terminal-shooting",
                        "composition_checkpoint_epoch": int(checkpoint["epoch"]),
                        "geometry_checkpoint_epoch": int(
                            geometry_checkpoint["epoch"]
                        ),
                        "dataset_source_hash": checkpoint["dataset_source_hash"],
                        "ode_steps": args.ode_steps,
                        "integrator": "midpoint",
                        "terminal_constraint_steps": args.terminal_steps,
                        "terminal_constraint_lr": args.terminal_lr,
                        "terminal_polyhedron_guidance": {
                            "P": "O4_tetrahedral",
                            "Fe": "O6_octahedral_ablation",
                            "note": "legacy endpoint-only ablation; excluded from the main experiment",
                        },
                        "composition_exact_filter": True,
                        "inference_max_atoms": args.max_atoms,
                        "family_or_parent_conditioning": False,
                    },
                }
                if accepted:
                    accepted_by_formula[formula].append(structure)
                    accepted_count += 1
                    accepted_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    accepted_handle.flush()
                    structure.to(
                        filename=str(cif_dir / f"{record['candidate_id']}.cif")
                    )
                else:
                    failed = [
                        name
                        for name, passed in feasibility["checks"].items()
                        if not passed and name != "thermodynamic_stability"
                    ]
                    if novelty_score < args.min_novelty:
                        failed.append("minimum_novelty")
                    if duplicate:
                        failed.append("within_shard_duplicate")
                    for name in failed:
                        rejection_counts[name] += 1
                    rejected_handle.write(
                        json.dumps(
                            {
                                "candidate_id": record["candidate_id"],
                                "N": n_atoms,
                                "reduced_formula": formula,
                                "novelty_score": novelty_score,
                                "failed_checks": failed,
                                "minimum_distance_A": feasibility["values"][
                                    "minimum_distance_A"
                                ],
                                "distance_violation_count": len(
                                    feasibility["details"]["distance_violations"]
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            batch_index += 1
            print(
                json.dumps(
                    {
                        "shard": args.shard,
                        "proposed": proposal_count,
                        "accepted": accepted_count,
                    }
                ),
                flush=True,
            )

    summary = {
        "shard": args.shard,
        "target": args.target,
        "proposed": proposal_count,
        "accepted": accepted_count,
        "target_reached": accepted_count >= args.target,
        "acceptance_rate": accepted_count / max(1, proposal_count),
        "duplicates_rejected": duplicate_count,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "accepted_jsonl": str(accepted_path),
        "rejected_jsonl": str(rejected_path),
        "cif_directory": str(cif_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if accepted_count < args.target:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

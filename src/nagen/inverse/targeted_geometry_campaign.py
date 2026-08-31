"""Fresh X,L sampling for promising N,A selected from unconditional DFlow."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .campaign import analytical_feasible
from .constraints import evaluate_feasibility
from .generate import generate_conditioned_geometry_batch, load_model
from .novelty import NoveltyIndex, crystal_descriptor


def select_compositions(records: list[dict], limit: int) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    for record in sorted(
        records,
        key=lambda item: item.get("local_hull", {}).get(
            "delta_e_hull_eV_atom", float("inf")
        ),
    ):
        formula = record["reduced_formula"]
        if formula in seen:
            continue
        selected.append(record)
        seen.add(formula)
        if len(selected) == limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--seed-records", required=True)
    parser.add_argument("--novelty-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--max-proposals", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-compositions", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--terminal-steps", type=int, default=400)
    parser.add_argument("--terminal-lr", type=float, default=0.02)
    parser.add_argument("--min-novelty", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_model(args.geometry_checkpoint, device)
    novelty = NoveltyIndex.load(args.novelty_index, device)
    with open(args.seed_records) as handle:
        seed_records = [json.loads(line) for line in handle if line.strip()]
    compositions = select_compositions(seed_records, args.num_compositions)
    if len(compositions) < args.num_compositions:
        raise ValueError(
            f"only {len(compositions)} unique compositions available"
        )
    element_to_index = {
        element: int(index)
        for element, index in checkpoint["element_to_index"].items()
    }
    index_to_element = {index: element for element, index in element_to_index.items()}
    composition_types: list[torch.Tensor] = []
    composition_meta: list[dict] = []
    for record in compositions:
        structure = Structure.from_dict(record["structure"])
        composition_types.append(
            torch.tensor(
                [element_to_index[str(specie)] for specie in structure.species],
                dtype=torch.long,
                device=device,
            )
        )
        composition_meta.append(
            {
                "source_candidate_id": record["candidate_id"],
                "reduced_formula": record["reduced_formula"],
                "source_data_relative_delta_e_hull_eV_atom": record[
                    "local_hull"
                ]["delta_e_hull_eV_atom"],
            }
        )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = output_dir / f"cif_shard_{args.shard:02d}"
    cif_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / f"accepted_shard_{args.shard:02d}.jsonl"
    rejected_path = output_dir / f"rejected_shard_{args.shard:02d}.jsonl"
    summary_path = output_dir / f"summary_shard_{args.shard:02d}.json"
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    accepted_by_formula: dict[str, list[Structure]] = defaultdict(list)
    generator = torch.Generator(device=device).manual_seed(
        args.seed + args.shard * 1_000_000
    )
    accepted = 0
    proposed = 0
    batch_index = 0
    rejected_counts: dict[str, int] = defaultdict(int)
    with accepted_path.open("w") as accepted_handle, rejected_path.open(
        "w"
    ) as rejected_handle:
        while accepted < args.target and proposed < args.max_proposals:
            batch = min(args.batch_size, args.max_proposals - proposed)
            selected_offsets = [
                (args.shard + proposed + row) % len(composition_types)
                for row in range(batch)
            ]
            sequences = []
            for offset in selected_offsets:
                original = composition_types[offset]
                permutation = torch.randperm(
                    len(original), device=device, generator=generator
                )
                sequences.append(original[permutation])
            batch_seed = args.seed + args.shard * 1_000_000 + batch_index
            types, frac, lattice, mask = generate_conditioned_geometry_batch(
                model,
                sequences,
                batch_seed,
                args.ode_steps,
                "midpoint",
                args.terminal_steps,
                args.terminal_lr,
            )
            descriptors = crystal_descriptor(
                types, frac, lattice, mask, novelty.config
            )
            novelty_scores, nearest_indices = novelty.score(descriptors)
            for row, composition_offset in enumerate(selected_offsets):
                proposed += 1
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
                analytic = analytical_feasible(feasibility)
                formula = structure.composition.reduced_formula
                score = float(novelty_scores[row])
                duplicate = analytic and any(
                    matcher.fit(structure, previous)
                    for previous in accepted_by_formula[formula]
                )
                keep = analytic and score >= args.min_novelty and not duplicate
                candidate_id = (
                    f"NaGen-DFlow-target-g{args.seed}-s{args.shard:02d}"
                    f"-p{proposed:05d}"
                )
                if keep:
                    nearest = int(nearest_indices[row])
                    meta = composition_meta[composition_offset]
                    record = {
                        "candidate_id": candidate_id,
                        "seed": batch_seed,
                        "N": n_atoms,
                        "formula": structure.composition.formula,
                        "reduced_formula": formula,
                        "structure": structure.as_dict(),
                        "novelty_score": score,
                        "nearest_reference_index": nearest,
                        "nearest_reference_id": novelty.ids[nearest],
                        "nearest_reference_formula": novelty.formulas[nearest],
                        "thermodynamic_stability_score": None,
                        "analytical_feasible": True,
                        "feasibility": feasibility,
                        "provenance": {
                            "generator": "naxl-dflow-conditioned-geometry",
                            "geometry_checkpoint_epoch": int(checkpoint["epoch"]),
                            "dataset_source_hash": checkpoint["dataset_source_hash"],
                            "N_A_source": "selected_prior_unconditional_dflow_candidate",
                            "N_A_source_candidate_id": meta["source_candidate_id"],
                            "N_A_source_formula": meta["reduced_formula"],
                            "N_A_source_data_relative_delta_e_hull_eV_atom": meta[
                                "source_data_relative_delta_e_hull_eV_atom"
                            ],
                            "fresh_X_L_sample": True,
                            "source_coordinates_or_lattice_copied": False,
                            "family_or_parent_conditioning": False,
                            "ode_steps": args.ode_steps,
                            "terminal_constraint_steps": args.terminal_steps,
                            "terminal_constraint_lr": args.terminal_lr,
                        },
                    }
                    accepted_by_formula[formula].append(structure)
                    accepted += 1
                    accepted_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    accepted_handle.flush()
                    structure.to(filename=str(cif_dir / f"{candidate_id}.cif"))
                else:
                    failures = [
                        name
                        for name, passed in feasibility["checks"].items()
                        if not passed and name != "thermodynamic_stability"
                    ]
                    if score < args.min_novelty:
                        failures.append("minimum_novelty")
                    if duplicate:
                        failures.append("within_shard_duplicate")
                    for failure in failures:
                        rejected_counts[failure] += 1
                    rejected_handle.write(
                        json.dumps(
                            {
                                "candidate_id": candidate_id,
                                "reduced_formula": formula,
                                "failed_checks": failures,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            batch_index += 1
            print(
                json.dumps(
                    {"shard": args.shard, "proposed": proposed, "accepted": accepted}
                ),
                flush=True,
            )
    summary = {
        "shard": args.shard,
        "proposed": proposed,
        "accepted": accepted,
        "target": args.target,
        "target_reached": accepted >= args.target,
        "selected_compositions": composition_meta,
        "rejection_counts": dict(rejected_counts),
        "accepted_jsonl": str(accepted_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if accepted < args.target:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Persistent GPU generation/relaxation workers and a resumable audited collector.

One calculator per worker is shared by the force/stress VJP and fixed-cell FIRE.
Each candidate is an atomic record; only the coordinator writes selected CIFs.
Run with the existing ShootingCSP Python. No alternate environment is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shootingcsp.selection.audit import load_crystals
from shootingcsp.inverse._io import json_default, sha256_file as sha256
from shootingcsp.inverse.spec import (
    DEFAULT_FE_COORDINATION_OPTIONS,
    parse_fe_coordination_options,
)
from shootingcsp.naming import build_output_stem_from_records, parse_output_index
from shootingcsp.selection.hull import ReferenceHull
from shootingcsp.selection.pipeline import Crystal, Conditioning, hard_gates


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, default=json_default, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def record(crystal):
    return {"material_id": crystal.id, "A": crystal.elements,
            "X_frac": crystal.frac, "L_matrix": crystal.lattice, "family": crystal.family}


def crystal(row):
    return Crystal(row["material_id"], row["A"], np.asarray(row["X_frac"], float),
                   np.asarray(row["L_matrix"], float), row.get("family", "phosphate"))


def inputs():
    return {"flow": ROOT / "runs/nacathode_broad_flow.best.pt",
            "profile": ROOT / "runs/nafe_po_profile.json",
            "condition": ROOT / "configs/framework_condition_orthophosphate.json",
            "references": ROOT / "runs/uma_reference_labels.jsonl",
            "uma": Path("/mnt/data2/aobo/UMA/uma-m-1p1.pt"),
            "known": ROOT / "runs/nafe_po_known"}


def load_hull(model_hash):
    rows = [json.loads(line) for line in inputs()["references"].read_text().splitlines()]
    return ReferenceHull(rows, model_hash)


def relax(
    evaluator,
    raw,
    condition,
    profile,
    steps=400,
    fe_coordination_options=DEFAULT_FE_COORDINATION_OPTIONS,
):
    from ase import Atoms
    from ase.optimize import FIRE
    atoms = Atoms(symbols=raw.elements, scaled_positions=raw.frac, cell=raw.lattice, pbc=True)
    atoms.calc = evaluator.calculator
    before = evaluator.evaluate_atoms(atoms)
    optimizer = FIRE(atoms, logfile=None, maxstep=.1)
    optimizer.run(fmax=.03, steps=steps)
    after = evaluator.evaluate_atoms(atoms)
    # Do not wrap terminal positions before measuring forces: avoid invalidating
    # the calculator cache by a mathematically equivalent periodic translation.
    final = Crystal(raw.id, raw.elements, atoms.get_scaled_positions(),
                    np.asarray(atoms.cell), raw.family)
    gates = hard_gates(
        final,
        condition,
        profile,
        forces=after["forces_eV_A"],
        force_threshold=.03,
        motif_backend="native",
        fe_coordination_options=fe_coordination_options,
    )
    return final, {"before": before, "after": after, "steps": optimizer.nsteps,
                   "converged": after["force_max_eV_A"] <= .03, "gates": gates}


def finish_candidate(
    row,
    final,
    audit,
    condition,
    profile,
    hull,
    get_validation,
    fe_coordination_options=DEFAULT_FE_COORDINATION_OPTIONS,
):
    """Common strict endpoint acceptance for scalar and batched proposals."""
    row.update(final=record(final), relaxation=audit)
    hr = hull.evaluate(final.elements, audit["after"]["energy_eV_atom"])
    row["reference_hull"] = hr
    geometry_ok = all(v for k, v in audit["gates"]["checks"].items() if k != "final_force")
    eh = hr.get("e_above_reference_hull_eV_atom")
    if geometry_ok and audit["after"]["force_max_eV_A"] <= .05 and eh is not None and eh <= .15:
        tick = time.monotonic()
        evaluator = get_validation()
        final, audit = relax(
            evaluator,
            final,
            condition,
            profile,
            steps=1000,
            fe_coordination_options=fe_coordination_options,
        )
        row.update(final=record(final), refinement=audit, validation_model=evaluator.provenance)
        row["timing"]["refinement_seconds"] = time.monotonic() - tick
        row["reference_hull"] = hull.evaluate(final.elements, audit["after"]["energy_eV_atom"])
    row["hard_gates"] = audit["gates"]
    eh = row["reference_hull"].get("e_above_reference_hull_eV_atom")
    row["status"] = "eligible" if audit["converged"] and audit["gates"]["passed"] and eh is not None and eh <= .15 else "rejected"
    return row


def batch_worker(args, model, indices, probabilities, energy_fn, evaluator, hull, condition, profile, config, get_validation):
    from ase import Atoms
    from shootingcsp.selection.batched_relax import fire_batch
    from generate_uma_guided_framework_candidates import make_source, optimize_source_with_uma, catastrophic_reason
    fe_coordination_options = tuple(
        config.get("protocol", {}).get(
            "fe_coordination_options", DEFAULT_FE_COORDINATION_OPTIONS
        )
    )
    out = Path(args.out)
    index = args.worker
    while not (out / "STOP").exists():
        pending = []
        while len(pending) < config["relax_batch_size"]:
            if args.max_samples and index >= args.max_samples:
                break
            path = out / "records" / f"{index:08d}.json"
            current = index
            index += args.workers
            if path.exists():
                continue
            seed = config["seed"] + current
            ident = f"uma_campaign_seed{seed}"
            started = time.monotonic()
            row = {"id": ident, "index": current, "source_seed": seed, "status": "started", "timing": {},
                   "generation_and_relaxation_inference": config["inference_settings"]}
            try:
                source, mask = make_source(model, indices, seed, "polyhedral")
                final_source, frac, cell, history = optimize_source_with_uma(
                    model,
                    source,
                    mask,
                    energy_fn,
                    probabilities,
                    12,
                    .03,
                    48,
                    fe_coordination_options=fe_coordination_options,
                )
                raw = Crystal(ident, [e for e in ("Na", "Fe", "P", "O") for _ in range(condition.counts.get(e, 0))],
                              frac[0].cpu().numpy(), cell[0].cpu().numpy(), condition.family)
                row.update(raw=record(raw), generation={"method": "fixed_NA_geometry_flow_UMA_DFlow_source_optimization",
                    "source_seed": seed, "dflow_steps": 12, "dflow_lr": .03, "ode_steps": 48,
                    "source_mode": "polyhedral", "generated_variables": "all Na/Fe/P/O coordinates and lattice",
                    "copied_parent_coordinates": False, "optimization_history": history,
                    "fe_coordination_options": list(fe_coordination_options),
                    "source_final_frac": final_source.frac.cpu().numpy(), "source_final_lattice": final_source.lattice.cpu().numpy()})
                row["timing"]["generation_seconds"] = time.monotonic() - started
                reason = catastrophic_reason(raw)
                if reason is not None:
                    row.update(status="unsafe", reason=reason)
                    write_json(path, row)
                else:
                    pending.append((path, row, raw))
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                write_json(path, row)
                raise
            if (out / "STOP").exists():
                break
        if not pending:
            break
        atoms = [Atoms(symbols=raw.elements, scaled_positions=raw.frac, cell=raw.lattice, pbc=True) for _, _, raw in pending]
        tick = time.monotonic()
        try:
            audits = fire_batch(evaluator.evaluate_batch, atoms, fmax=.03, steps=400, maxstep=.1)
        except Exception as error:
            for path, row, _ in pending:
                row.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                write_json(path, row)
            raise
        elapsed = time.monotonic() - tick
        for (path, row, raw), a, audit in zip(pending, atoms, audits):
            try:
                final = Crystal(raw.id, raw.elements, a.get_scaled_positions(), np.asarray(a.cell), raw.family)
                audit["gates"] = hard_gates(
                    final,
                    condition,
                    profile,
                    forces=audit["after"]["forces_eV_A"],
                    force_threshold=.03,
                    motif_backend="native",
                    fe_coordination_options=fe_coordination_options,
                )
                row["timing"].update(relaxation_seconds=elapsed / len(pending), batch_relaxation_seconds=elapsed, batch_size=len(pending))
                finish_candidate(row, final, audit, condition, profile, hull, get_validation)
                row["timing"]["total_seconds"] = row["timing"]["generation_seconds"] + elapsed / len(pending) + row["timing"].get("refinement_seconds", 0.)
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            write_json(path, row)
            print(json.dumps({"index": row["index"], "id": row["id"], "status": row["status"], "timing": row["timing"]}), flush=True)
            if row["status"] == "failed":
                raise RuntimeError(row["error"])
    write_json(out / f"worker_{args.worker}.json", {"status": "stopped", "pid": os.getpid()})


def worker(args):
    import torch
    from torch.nn import functional as F
    from shootingcsp.inverse.generate import load_model
    from shootingcsp.inverse.uma_guidance import load_uma_calculator_vjp_factory
    from shootingcsp.selection.uma import UMAEvaluator
    from generate_uma_guided_framework_candidates import make_source, optimize_source_with_uma, catastrophic_reason

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    files = inputs()
    config = json.loads((out / "configuration.json").read_text())
    fe_coordination_options = tuple(
        config.get("protocol", {}).get(
            "fe_coordination_options", DEFAULT_FE_COORDINATION_OPTIONS
        )
    )
    profile = json.loads(files["profile"].read_text())
    condition = Conditioning(**json.loads(files["condition"].read_text()))
    model, checkpoint = load_model(str(files["flow"]), torch.device("cuda"), True)
    if checkpoint.get("training_mode") not in ("geometry", "fixed_NA_geometry"):
        raise ValueError("geometry-only checkpoint required")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    elements = [e for e in ("Na", "Fe", "P", "O") for _ in range(condition.counts.get(e, 0))]
    indices = torch.tensor([checkpoint["element_to_index"][e] for e in elements], device="cuda")
    if any(checkpoint["element_to_index"][e] != i for i, e in enumerate(("Na", "Fe", "P", "O"), 1)):
        raise ValueError("soft gates require core Na/Fe/P/O vocabulary")
    probabilities = F.one_hot(indices[None] - 1, num_classes=4).float()
    mode = config.get("inference_settings", "default")
    evaluator = UMAEvaluator(str(files["uma"]), "cuda", "omat", inference_settings=mode)
    validation_evaluator = evaluator if mode == "default" else None
    def get_validation():
        nonlocal validation_evaluator
        if validation_evaluator is None:
            validation_evaluator = UMAEvaluator(str(files["uma"]), "cuda", "omat")
        return validation_evaluator
    if evaluator.provenance["state_sha256"] != config["hashes"]["uma"]:
        raise ValueError("UMA checkpoint changed")
    factory, _ = load_uma_calculator_vjp_factory(str(files["uma"]), calculator=evaluator.calculator)
    energy_fn = factory(torch.tensor([{"Na": 11, "Fe": 26, "P": 15, "O": 8}[e] for e in elements]))
    hull = load_hull(evaluator.provenance["state_sha256"])
    write_json(out / f"worker_{args.worker}.json", {"status": "ready", "pid": os.getpid(), "model": evaluator.provenance})
    if mode.startswith("batch_"):
        return batch_worker(args, model, indices, probabilities, energy_fn, evaluator, hull, condition, profile, config, get_validation)
    index = args.worker
    while not (out / "STOP").exists():
        if args.max_samples and index >= args.max_samples:
            break
        path = out / "records" / f"{index:08d}.json"
        if path.exists():
            index += args.workers
            continue
        seed = config["seed"] + index
        ident = f"uma_campaign_seed{seed}"
        row = {"id": ident, "index": index, "source_seed": seed, "status": "started", "timing": {},
               "generation_and_relaxation_inference": mode}
        started = time.monotonic()
        try:
            source, mask = make_source(model, indices, seed, "polyhedral")
            final_source, frac, cell, history = optimize_source_with_uma(
                model,
                source,
                mask,
                energy_fn,
                probabilities,
                12,
                .03,
                48,
                fe_coordination_options=fe_coordination_options,
            )
            raw = Crystal(ident, elements, frac[0].cpu().numpy(), cell[0].cpu().numpy(), condition.family)
            row.update(raw=record(raw), generation={"method": "fixed_NA_geometry_flow_UMA_DFlow_source_optimization",
                "source_seed": seed, "dflow_steps": 12, "dflow_lr": .03, "ode_steps": 48,
                "source_mode": "polyhedral", "generated_variables": "all Na/Fe/P/O coordinates and lattice",
                "copied_parent_coordinates": False, "optimization_history": history,
                "fe_coordination_options": list(fe_coordination_options),
                "source_final_frac": final_source.frac.cpu().numpy(),
                "source_final_lattice": final_source.lattice.cpu().numpy()})
            row["timing"]["generation_seconds"] = time.monotonic() - started
            reason = catastrophic_reason(raw)
            if reason is not None:
                row.update(status="unsafe", reason=reason)
            else:
                tick = time.monotonic()
                final, audit = relax(
                    evaluator,
                    raw,
                    condition,
                    profile,
                    fe_coordination_options=fe_coordination_options,
                )
                row["timing"]["relaxation_seconds"] = time.monotonic() - tick
                finish_candidate(
                    row,
                    final,
                    audit,
                    condition,
                    profile,
                    hull,
                    get_validation,
                    fe_coordination_options=fe_coordination_options,
                )
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        row["timing"]["total_seconds"] = time.monotonic() - started
        write_json(path, row)
        print(json.dumps({"index": index, "id": ident, "status": row["status"], "timing": row["timing"]}), flush=True)
        if row["status"] == "failed":
            # A software/runtime failure must surface instead of burning an
            # unbounded series of samples with a broken worker.
            raise RuntimeError(row["error"])
        index += args.workers
    write_json(out / f"worker_{args.worker}.json", {"status": "stopped", "pid": os.getpid()})


class Collector:
    def __init__(self, out):
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from select_framework_candidates import strip_na, structure, coordination_graph
        from shootingcsp.selection.diversity import descriptor
        self.out = Path(out)
        self.strip, self.structure, self.graph, self.descriptor = strip_na, structure, coordination_graph, descriptor
        self.matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=True)
        condition = Conditioning(**json.loads(inputs()["condition"].read_text()))
        self.condition = condition
        self.profile = json.loads(inputs()["profile"].read_text())
        config = json.loads((self.out / "configuration.json").read_text())
        self.fe_coordination_options = tuple(
            config.get("protocol", {}).get(
                "fe_coordination_options", DEFAULT_FE_COORDINATION_OPTIONS
            )
        )
        self.hull = load_hull(config["hashes"]["uma"])
        target = Counter({e: n for e, n in condition.counts.items() if e != "Na"})
        known = [c for split in ("train", "val", "test") for c in load_crystals(inputs()["known"] / f"{split}.jsonl")]
        self.refs = [self.strip(c) for c in known if Counter(e for e in c.elements if e != "Na") == target]
        if not self.refs:
            raise ValueError("no known framework references; novelty cannot be asserted")
        self.ref_structures = [self.structure(c) for c in self.refs]
        self.ref_graphs = [self.graph(c) for c in self.refs]
        self.ref_features = np.stack([descriptor(c) for c in self.refs])
        self.selected = []
        self.selected_structures, self.selected_features = [], []
        for path in sorted((self.out / "selected").glob("*.audit.json")):
            entry = json.loads(path.read_text())
            self._register(entry)
        self._next_output_index = 1 + max(
            (
                index
                for entry in self.selected
                for index in [parse_output_index(Path(entry["file"]).stem)]
                if index is not None
            ),
            default=0,
        )

    def _register(self, entry):
        if sha256(self.out / "selected" / entry["file"]) != entry["sha256"]:
            raise ValueError(f"selected CIF hash mismatch: {entry['id']}")
        c = self.strip(crystal(entry["final"]))
        self.selected.append(entry)
        self.selected_structures.append(self.structure(c))
        self.selected_features.append(self.descriptor(c))

    def consider(self, row):
        import networkx as nx
        from networkx.algorithms.isomorphism import categorical_node_match
        from shootingcsp.selection.diversity import distances
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
        from pymatgen.analysis.structure_matcher import StructureMatcher
        if any(e["id"] == row["id"] for e in self.selected):
            return {"status": "already_selected"}
        if (row.get("generation_and_relaxation_inference", "default") != "default"
                and row.get("validation_model", {}).get("inference_settings") != "default"):
            raise ValueError("accelerated candidate lacks default-inference final validation")
        final = crystal(row["final"])
        label = row.get("refinement", row.get("relaxation", {})).get("after")
        if label is None:
            raise ValueError("final forces and energy are required for independent collection audit")
        gates = hard_gates(
            final,
            self.condition,
            self.profile,
            forces=label["forces_eV_A"],
            force_threshold=.03,
            motif_backend="native",
            fe_coordination_options=self.fe_coordination_options,
        )
        hr = self.hull.evaluate(final.elements, label["energy_eV_atom"])
        eh = hr.get("e_above_reference_hull_eV_atom")
        if not gates["passed"] or eh is None or eh > .15:
            return {"status": "strict_collection_rejected", "hard_gates": gates, "reference_hull": hr}
        provenance = row.get("generation", {})
        if provenance.get("copied_parent_coordinates") is not False or provenance.get("generated_variables") != "all Na/Fe/P/O coordinates and lattice":
            raise ValueError("generated framework provenance missing")
        row = {**row, "hard_gates": gates, "reference_hull": hr}
        before, after = self.strip(crystal(row["raw"])), self.strip(crystal(row["final"]))
        bs, fs, fg = self.structure(before), self.structure(after), self.graph(after)
        pre = any(self.matcher.fit(bs, ref) for ref in self.ref_structures)
        post = any(self.matcher.fit(fs, ref) for ref in self.ref_structures)
        topology = any(nx.is_isomorphic(fg, graph, node_match=categorical_node_match("element", "")) for graph in self.ref_graphs)
        feature = self.descriptor(after)
        nearest = float(distances(feature[None], self.ref_features).min())
        duplicates = [e["id"] for e, s in zip(self.selected, self.selected_structures) if self.matcher.fit(fs, s)]
        selected_distance = float(distances(feature[None], np.stack(self.selected_features)).min()) if self.selected_features else None
        audit = {"pre_relax_training_match": pre, "post_refine_training_match": post,
                 "post_refine_topology_match": topology, "nearest_known_framework_distance": nearest,
                 "pre_edges": self.graph(before).number_of_edges(), "post_refine_edges": fg.number_of_edges(),
                 "selected_structure_matches": duplicates, "nearest_selected_framework_distance": selected_distance,
                 "known_framework_references": len(self.refs)}
        accepted = not (pre or post or topology or duplicates) and nearest >= 1e-4 and (selected_distance is None or selected_distance >= 1e-4)
        if not accepted:
            return {"status": "novelty_or_duplicate_rejected", "framework_novelty": audit}
        output_index = self._next_output_index
        output_name = build_output_stem_from_records(
            output_index,
            gates,
            hr,
            audit,
        )
        path = self.out / "selected" / f"{output_name}.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"selected output already exists: {path}")
        CifWriter(self.structure(crystal(row["final"])), significant_figures=10).write_file(path)
        roundtrip = StructureMatcher(ltol=1e-6, stol=1e-5, angle_tol=1e-5, primitive_cell=False, scale=False).fit(self.structure(crystal(row["final"])), Structure.from_file(path))
        if not roundtrip:
            raise ValueError(f"CIF round-trip failed: {row['id']}")
        self._next_output_index += 1
        entry = {**row, "status": "selected_refined", "framework_novelty": audit,
                 "output_index": output_index, "output_name": output_name,
                 "file": path.name, "vesta_png": f"{output_name}.png",
                 "sha256": sha256(path), "cif_roundtrip": True}
        write_json(path.with_suffix(".audit.json"), entry)
        self._register(entry)
        return {"status": "selected", "framework_novelty": audit}


def bootstrap(out, config):
    """Retain the audited original structure without counting it twice."""
    manifest = json.loads((ROOT / "runs/final_selected/manifest.json").read_text())
    if sha256(ROOT / "runs/final_selected" / manifest["file"]) != manifest["sha256"]:
        raise ValueError("original CIF hash mismatch")
    ident = manifest["id"]
    raw = next(c for c in load_crystals(ROOT / "runs/framework_uma_dflow/generation/generated_all.jsonl") if c.id == ident)
    final = next(c for c in load_crystals(ROOT / "runs/final_refine_relax/relaxed.jsonl") if c.id == ident)
    row = {"id": ident, "raw": record(raw), "final": record(final), "status": "eligible",
           "hard_gates": manifest["hard_gates"], "reference_hull": manifest["reference_hull"],
           "strict_relaxation": manifest["strict_relaxation"], "origin": "previous_128_sample_run",
           "uma_model": manifest["uma_model"]}
    row["refinement"] = next(json.loads(line) for line in (ROOT / "runs/final_refine_relax/pairs.jsonl").read_text().splitlines() if json.loads(line)["id"] == ident)
    row["generation"] = next(json.loads(line)["generation"] for line in (ROOT / "runs/framework_uma_dflow/generation/generated_all.jsonl").read_text().splitlines() if json.loads(line)["material_id"] == ident)
    if manifest["uma_model"]["state_sha256"] != config["hashes"]["uma"]:
        raise ValueError("original model mismatch")
    return row


def coordinator(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = inputs()
    config_path = out / "configuration.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if (config["seed"], config["workers"], config["target"]) != (args.seed, args.workers, args.target):
            raise ValueError("resume requires identical seed, workers, and target")
        stored_fe_coordination = tuple(
            config.get("protocol", {}).get(
                "fe_coordination_options", DEFAULT_FE_COORDINATION_OPTIONS
            )
        )
        if stored_fe_coordination != args.fe_coordination:
            raise ValueError(
                "resume requires identical Fe coordination options: "
                f"{stored_fe_coordination} != {args.fe_coordination}"
            )
    else:
        config = {"seed": args.seed, "workers": args.workers, "target": args.target,
                  "python": sys.executable, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "hashes": {k: sha256(v) for k, v in files.items() if k != "known"},
                  "known_hashes": {s: sha256(files["known"] / f"{s}.jsonl") for s in ("train", "val", "test")},
                  "protocol": {"dflow_steps": 12, "ode_steps": 48, "dflow_lr": .03, "fmax": .03,
                               "relax_steps": 400, "refine_steps": 1000, "hull_threshold": .15,
                               "duplicate_threshold": 1e-4, "task_name": "omat", "fixed_cell": True,
                               "fe_coordination_options": list(args.fe_coordination)}}
        write_json(config_path, config)
    for k, path in files.items():
        if k != "known" and sha256(path) != config["hashes"][k]:
            raise ValueError(f"frozen input changed: {k}")
    for s, digest in config["known_hashes"].items():
        if sha256(files["known"] / f"{s}.jsonl") != digest:
            raise ValueError(f"known reference split changed: {s}")
    if args.inference_settings != config.get("inference_settings", "default"):
        if args.inference_settings.startswith("fixed_composition_fp32"):
            benchmark_name = "uma_compiled_inference_benchmark.json" if args.inference_settings.endswith("_compiled") else "uma_inference_benchmark.json"
            benchmark = json.loads((ROOT / "runs" / benchmark_name).read_text())
            if not benchmark["passed"]:
                raise ValueError("accelerated inference benchmark did not pass")
        elif args.inference_settings.startswith("batch_"):
            name = "uma_batch_tf32_benchmark.json" if args.inference_settings == "batch_tf32" else "uma_batch_benchmark.json"
            benchmark = json.loads((ROOT / "runs" / name).read_text())
            if not benchmark["passed"]:
                raise ValueError("batched inference benchmark did not pass")
        config.setdefault("runtime_phases", []).append({
            "previous": config.get("inference_settings", "default"), "current": args.inference_settings,
            "changed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "final_validation_inference": "default"})
        config["inference_settings"] = args.inference_settings
        write_json(config_path, config)
    config["relax_batch_size"] = args.relax_batch_size
    config["source_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in (
        Path(__file__).resolve(), ROOT / "tools/generate_uma_guided_framework_candidates.py",
        ROOT / "src/shootingcsp/inverse/uma_guidance.py", ROOT / "src/shootingcsp/inverse/guidance.py",
        ROOT / "src/shootingcsp/inverse/constraints.py", ROOT / "src/shootingcsp/inverse/spec.py",
        ROOT / "src/shootingcsp/selection/uma.py", ROOT / "src/shootingcsp/selection/pipeline.py",
        ROOT / "src/shootingcsp/selection/hull.py",
        ROOT / "src/shootingcsp/selection/batched_relax.py",
        ROOT / "src/shootingcsp/selection/diversity.py", ROOT / "tools/select_framework_candidates.py")}
    write_json(config_path, config)
    collector = Collector(out)
    collector.consider(bootstrap(out, config))
    if len(collector.selected) >= args.target:
        print("Target already met", flush=True)
        return
    if (out / "STOP").exists():
        raise ValueError("STOP marker present; inspect campaign before resuming")
    processes, handles = [], []
    try:
        for i in range(args.workers):
            log = (out / f"worker_{i}.log").open("a")
            handles.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
            env["PYTHONPATH"] = str(ROOT / "src")
            # Share the runtime's existing compilation cache across workers and
            # the validated benchmark, instead of compiling eight identical
            # models in isolated cache directories.
            env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
            cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--out", str(out),
                   "--worker", str(i), "--workers", str(args.workers), "--threads", str(args.threads),
                   "--max-samples", str(args.max_samples)]
            processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True))
        write_json(out / "processes.json", {"coordinator": os.getpid(), "workers": [p.pid for p in processes]})
        while True:
            counts = Counter()
            for path in sorted((out / "records").glob("*.json")):
                row = json.loads(path.read_text())
                counts[row["status"]] += 1
                decision = out / "decisions" / path.name
                if not decision.exists():
                    result = collector.consider(row) if row["status"] == "eligible" and len(collector.selected) < args.target else {"status": row["status"]}
                    write_json(decision, {"id": row["id"], **result})
            done = len(collector.selected) >= args.target
            alive = sum(p.poll() is None for p in processes)
            status = "completed" if done else ("running" if alive else "stopped_before_target")
            summary = {"status": status, "target": args.target, "selected": len(collector.selected),
                       "new_samples": sum(counts.values()), "counts": dict(counts), "active_workers": alive,
                       "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "selected_ids": [e["id"] for e in collector.selected],
                       "note": "Frozen finite same-UMA-reference hull proxy; no DFT stability claim."}
            write_json(out / "summary.json", summary)
            write_json(out / "selected/manifest.json", {"configuration": config, "summary": summary,
                       "structures": [{k: e[k] for k in ("id", "file", "sha256", "reference_hull", "hard_gates", "framework_novelty", "cif_roundtrip")} for e in collector.selected]})
            print(json.dumps(summary), flush=True)
            if done:
                write_json(out / "STOP", {"reason": "target_met"})
                break
            if not alive:
                raise RuntimeError("all workers stopped before target; inspect worker logs")
            time.sleep(15)
    finally:
        # Workers are owned by this coordinator. Do not leave orphan jobs if
        # collection/auditing fails or the coordinator is interrupted.
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "worker"))
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, default=24)
    p.add_argument("--seed", type=int, default=2026091000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--inference-settings", choices=("default", "fixed_composition_fp32", "fixed_composition_fp32_compiled", "batch_fp32", "batch_tf32"), default="default")
    p.add_argument(
        "--fe-coordination",
        type=parse_fe_coordination_options,
        default=DEFAULT_FE_COORDINATION_OPTIONS,
        help="Allowed Fe-O coordination numbers, e.g. 4, 4,5, or 4,5,6.",
    )
    p.add_argument("--relax-batch-size", type=int, default=1)
    args = p.parse_args()
    if args.target < 1 or args.workers < 1 or args.threads < 1 or not 0 <= args.worker < args.workers or args.max_samples < 0 or args.relax_batch_size < 1:
        p.error("invalid target/worker/thread/sample budget")
    (worker if args.mode == "worker" else coordinator)(args)


if __name__ == "__main__":
    main()

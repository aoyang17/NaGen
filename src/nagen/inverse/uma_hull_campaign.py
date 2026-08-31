"""Build a same-scale UMA convex hull from Materials Project structures.

The workflow is intentionally split into three resumable commands:

``fetch``
    Download and cache every MP structure represented by the frozen phase-entry
    cache.  Credentials are read only in memory and are never serialized.
``relax``
    Relax a deterministic shard with one UMA checkpoint and FIRE/BFGS protocol.
``build``
    Require complete, converged support and build target-composition reference
    energies on the UMA scale.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ._io import atomic_json as _atomic_json
from ._io import sha256_file, sha256_json
from ._uma import load_calculator as _load_calculator
from .mp_uma_calibration import ase_atoms_from_structure_dict, material_id
from .relax_uma import RelaxationConfig, relax_structure
from .uma_guidance import checkpoint_sha256


STRUCTURE_CACHE_VERSION = "uma-hull-mp-structures-v1"
RELAX_RECORD_VERSION = "uma-hull-relax-record-v1"
HULL_CACHE_VERSION = "uma-hull-cache-v1"
SUPPORT_POLICY = "all_mp2020_compatible_entries"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    _atomic_json(path, payload, sort_keys=True)


def load_phase_support(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if payload.get("version") != "mp-phase-entries-v1":
        raise ValueError("unsupported MP phase cache")
    if payload.get("compatibility") != "MaterialsProject2020Compatibility":
        raise ValueError("phase cache is not MP2020 compatible")
    entries = list(payload.get("entries", []))
    if len(entries) != int(payload.get("n_compatible_entries", -1)):
        raise ValueError("phase cache entry count does not match its manifest")
    seen: set[str] = set()
    for record in entries:
        entry_id = str(record["entry_id"])
        if entry_id in seen:
            raise ValueError(f"duplicate phase entry {entry_id}")
        seen.add(entry_id)
        if not material_id(entry_id).startswith("mp-"):
            raise ValueError(f"phase entry has no Materials Project ID: {entry_id}")
    return payload, sorted(entries, key=lambda item: str(item["entry_id"]))


def read_api_key(source_script: Path | None = None) -> str:
    key = os.environ.get("MP_API_KEY")
    if key:
        return key
    if source_script is None:
        raise RuntimeError("MP_API_KEY is required for missing MP structures")
    text = source_script.read_text()
    patterns = (
        r"(?:MAPI_KEY|MP_API_KEY|api_key)\s*=\s*['\"]([^'\"]+)['\"]",
        r"MPRester\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    raise RuntimeError("no Materials Project API key was found in the source script")


def seed_structures(
    phase_entries: Iterable[dict[str, Any]], seed_cache: Path | None,
) -> dict[str, dict[str, Any]]:
    """Map an older material-ID-keyed cache onto exact phase-entry IDs."""
    if seed_cache is None or not seed_cache.exists():
        return {}
    payload = json.loads(seed_cache.read_text())
    source = payload.get("structures", {})
    seeded: dict[str, dict[str, Any]] = {}
    for entry in phase_entries:
        entry_id = str(entry["entry_id"])
        mp_id = material_id(entry_id)
        if mp_id in source:
            seeded[entry_id] = {
                "entry_id": entry_id,
                "requested_material_id": mp_id,
                "returned_material_id": payload.get("returned_aliases", {}).get(
                    mp_id, mp_id
                ),
                "composition": entry["composition"],
                "structure": source[mp_id],
                "source": "seed_cache",
            }
    return seeded


def structure_cache_payload(
    phase_path: Path, phase_payload: dict[str, Any], entries: list[dict[str, Any]],
    structures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = [str(entry["entry_id"]) for entry in entries]
    missing = sorted(set(expected) - set(structures))
    return {
        "version": STRUCTURE_CACHE_VERSION,
        "status": "complete" if not missing else "incomplete",
        "support_policy": SUPPORT_POLICY,
        "phase_cache": str(phase_path.resolve()),
        "phase_cache_sha256": sha256_file(phase_path),
        "phase_cache_queried_at": phase_payload.get("queried_at"),
        "mp_compatibility": phase_payload.get("compatibility"),
        "expected_entry_count": len(expected),
        "cached_entry_count": len(structures),
        "missing_entry_ids": missing,
        "updated_at": utc_now(),
        "structures": structures,
    }


def fetch_command(args: argparse.Namespace) -> None:
    phase_path = Path(args.phase_cache)
    phase_payload, entries = load_phase_support(phase_path)
    target = Path(args.out)
    if target.exists():
        cached = json.loads(target.read_text())
        if cached.get("version") != STRUCTURE_CACHE_VERSION:
            raise ValueError("cannot resume an incompatible structure cache")
        if cached.get("phase_cache_sha256") != sha256_file(phase_path):
            raise ValueError("cannot resume with a different phase cache")
        structures = dict(cached.get("structures", {}))
    else:
        structures = seed_structures(
            entries, Path(args.seed_cache) if args.seed_cache else None
        )
        atomic_json(
            target,
            structure_cache_payload(
                phase_path, phase_payload, entries, structures
            ),
        )
    missing = [entry for entry in entries if str(entry["entry_id"]) not in structures]
    if not missing:
        print(json.dumps({"status": "complete", "cached_entry_count": len(structures)}))
        return
    if args.prepare_only:
        print(json.dumps({
            "status": "incomplete",
            "cached_entry_count": len(structures),
            "expected_entry_count": len(entries),
            "missing_entry_count": len(missing),
        }, indent=2))
        return
    api_key = read_api_key(
        Path(args.api_key_source_script) if args.api_key_source_script else None
    )
    from mp_api.client import MPRester

    with MPRester(api_key, use_document_model=False) as rester:
        for offset in range(0, len(missing), args.batch_size):
            batch = missing[offset:offset + args.batch_size]
            requested = {material_id(str(entry["entry_id"])): entry for entry in batch}
            documents = rester.materials.summary.search(
                material_ids=sorted(requested),
                fields=["material_id", "structure"],
                all_fields=False,
            )
            returned = {str(document["material_id"]): document for document in documents}
            unresolved: list[dict[str, Any]] = []
            for requested_id, entry in requested.items():
                document = returned.get(requested_id)
                if document is None:
                    unresolved.append(entry)
                    continue
                structure = document["structure"]
                entry_id = str(entry["entry_id"])
                structures[entry_id] = {
                    "entry_id": entry_id,
                    "requested_material_id": requested_id,
                    "returned_material_id": str(document["material_id"]),
                    "composition": entry["composition"],
                    "structure": (
                        structure.as_dict() if hasattr(structure, "as_dict") else structure
                    ),
                    "source": "materials_project_api",
                }
            # Old MP IDs can resolve to a current alias. Querying those IDs alone
            # preserves the requested-ID association even if the returned ID differs.
            for entry in unresolved:
                requested_id = material_id(str(entry["entry_id"]))
                documents = rester.materials.summary.search(
                    material_ids=[requested_id],
                    fields=["material_id", "structure"],
                    all_fields=False,
                )
                if len(documents) != 1:
                    continue
                document = documents[0]
                structure = document["structure"]
                entry_id = str(entry["entry_id"])
                structures[entry_id] = {
                    "entry_id": entry_id,
                    "requested_material_id": requested_id,
                    "returned_material_id": str(document["material_id"]),
                    "composition": entry["composition"],
                    "structure": (
                        structure.as_dict() if hasattr(structure, "as_dict") else structure
                    ),
                    "source": "materials_project_api_alias",
                }
            atomic_json(
                target,
                structure_cache_payload(
                    phase_path, phase_payload, entries, structures
                ),
            )
            print(json.dumps({
                "cached_entry_count": len(structures),
                "expected_entry_count": len(entries),
            }), flush=True)
    final = structure_cache_payload(phase_path, phase_payload, entries, structures)
    atomic_json(target, final)
    if final["status"] != "complete":
        raise RuntimeError(
            f"MP structure cache is incomplete: {len(final['missing_entry_ids'])} missing"
        )


def _safe_id(entry_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", entry_id)


def _structure_payload(atoms: Any) -> dict[str, Any]:
    return {
        "elements": atoms.get_chemical_symbols(),
        "frac_coords": np.asarray(
            atoms.get_scaled_positions(wrap=True), dtype=float
        ).tolist(),
        "lattice": np.asarray(atoms.cell.array, dtype=float).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
    }


def relax_command(args: argparse.Namespace) -> None:
    phase_path = Path(args.phase_cache)
    _, entries = load_phase_support(phase_path)
    structure_path = Path(args.structure_cache)
    structure_cache = json.loads(structure_path.read_text())
    if structure_cache.get("version") != STRUCTURE_CACHE_VERSION:
        raise ValueError("unsupported UMA hull structure cache")
    if structure_cache.get("status") != "complete":
        raise RuntimeError("UMA hull structure cache is incomplete")
    if structure_cache.get("phase_cache_sha256") != sha256_file(phase_path):
        raise ValueError("phase and structure cache hashes do not match")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("shard index must be in [0, shard_count)")
    selected = [
        entry for index, entry in enumerate(entries)
        if index % args.shard_count == args.shard_index
    ]
    checkpoint_hash = checkpoint_sha256(args.uma_checkpoint)
    config = RelaxationConfig(
        fire_fmax_eV_A=args.fire_fmax,
        fire_steps=args.fire_steps,
        bfgs_fmax_eV_A=args.bfgs_fmax,
        bfgs_steps=args.bfgs_steps,
    )
    protocol = {
        **config.__dict__,
        "fire_variables": "cell_and_positions_via_ExpCellFilter",
        "bfgs_variables": "positions_only",
        "energy_normalization": "eV_per_atom",
    }
    protocol_hash = sha256_json(protocol)
    phase_hash = sha256_file(phase_path)
    structure_hash = sha256_file(structure_path)
    root = Path(args.out_directory)
    record_directory = root / "records"
    log_directory = root / "logs"
    calculator = _load_calculator(args.uma_checkpoint, args.device, args.task_name)
    counts = {"selected": len(selected), "converged": 0, "failed": 0, "skipped": 0}
    for entry in selected:
        entry_id = str(entry["entry_id"])
        target = record_directory / f"{_safe_id(entry_id)}.json"
        if target.exists() and not args.resume:
            raise FileExistsError(
                f"relaxation record already exists for {entry_id}; use --resume"
            )
        if target.exists() and args.resume:
            previous = json.loads(target.read_text())
            identity_matches = (
                previous.get("version") == RELAX_RECORD_VERSION
                and previous.get("entry_id") == entry_id
                and previous.get("uma_model_sha256") == checkpoint_hash
                and previous.get("task_name") == args.task_name
                and previous.get("protocol_sha256") == protocol_hash
                and previous.get("phase_cache_sha256") == phase_hash
                and previous.get("structure_cache_sha256") == structure_hash
            )
            if not identity_matches:
                raise RuntimeError(f"resume metadata mismatch for {entry_id}")
            if previous.get("converged"):
                counts["converged"] += 1
                counts["skipped"] += 1
                continue
        source = structure_cache["structures"][entry_id]
        atoms = ase_atoms_from_structure_dict(source["structure"])
        from pymatgen.core import Composition
        expected_formula = Composition(entry["composition"]).reduced_formula
        structure_formula = Composition(
            dict(Counter(atoms.get_chemical_symbols()))
        ).reduced_formula
        if structure_formula != expected_formula:
            raise ValueError(
                f"structure composition mismatch for {entry_id}: "
                f"{structure_formula} != {expected_formula}"
            )
        record: dict[str, Any] = {
            "version": RELAX_RECORD_VERSION,
            "entry_id": entry_id,
            "requested_material_id": source["requested_material_id"],
            "returned_material_id": source["returned_material_id"],
            "composition": entry["composition"],
            "n_atoms": len(atoms),
            "phase_cache_sha256": phase_hash,
            "structure_cache_sha256": structure_hash,
            "source_structure_sha256": sha256_json(source["structure"]),
            "uma_checkpoint": str(Path(args.uma_checkpoint).resolve()),
            "uma_model_sha256": checkpoint_hash,
            "task_name": args.task_name,
            "protocol": protocol,
            "protocol_sha256": protocol_hash,
            "device": args.device,
            "started_at": utc_now(),
            "status": "running",
            "converged": False,
        }
        atomic_json(target, record)
        try:
            atoms.calc = calculator
            initial_energy = float(atoms.get_potential_energy()) / len(atoms)
            result = relax_structure(
                atoms, calculator, str(log_directory / _safe_id(entry_id)), config
            )
            final_energy = float(result.atoms.get_potential_energy()) / len(atoms)
            final_stress = np.asarray(
                result.atoms.get_stress(voigt=False), dtype=float
            )
            record.update({
                "status": result.status,
                "converged": result.converged,
                "fire_converged": result.fire_converged,
                "bfgs_converged": result.bfgs_converged,
                "final_fmax_eV_A": result.final_fmax_eV_A,
                "initial_energy_eV_atom": initial_energy,
                "final_energy_eV_atom": final_energy,
                "final_stress_eV_A3": final_stress.tolist(),
                "final_max_abs_stress_eV_A3": float(np.abs(final_stress).max()),
                "relaxed_structure": _structure_payload(result.atoms),
                "finished_at": utc_now(),
            })
            counts["converged" if result.converged else "failed"] += 1
        except Exception as error:
            record.update({
                "status": "error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "finished_at": utc_now(),
            })
            counts["failed"] += 1
        atomic_json(target, record)
        print(json.dumps({
            "entry_id": entry_id, "status": record["status"],
            "converged": record["converged"],
        }), flush=True)
    manifest = {
        "version": "uma-hull-relax-shard-v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "phase_cache": str(phase_path.resolve()),
        "phase_cache_sha256": phase_hash,
        "structure_cache": str(structure_path.resolve()),
        "structure_cache_sha256": structure_hash,
        "uma_checkpoint": str(Path(args.uma_checkpoint).resolve()),
        "uma_model_sha256": checkpoint_hash,
        "task_name": args.task_name,
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "counts": counts,
        "finished_at": utc_now(),
    }
    atomic_json(root / f"manifest_shard_{args.shard_index:03d}.json", manifest)
    print(json.dumps(counts, indent=2))


def load_relax_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("records/*.json")):
        record = json.loads(path.read_text())
        if record.get("version") != RELAX_RECORD_VERSION:
            continue
        entry_id = str(record["entry_id"])
        if entry_id in records and records[entry_id] != record:
            raise ValueError(f"conflicting relaxation records for {entry_id}")
        records[entry_id] = record
    return records


def build_hull_payload(
    phase_path: Path, composition_pool_path: Path, records_root: Path,
) -> dict[str, Any]:
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    phase_payload, support = load_phase_support(phase_path)
    records = load_relax_records(records_root)
    expected_ids = {str(entry["entry_id"]) for entry in support}
    missing = sorted(expected_ids - set(records))
    unconverged = sorted(
        entry_id for entry_id in expected_ids & set(records)
        if not records[entry_id].get("converged")
    )
    if missing or unconverged:
        raise RuntimeError(
            "official UMA hull requires complete converged support: "
            f"missing={len(missing)}, unconverged={len(unconverged)}"
        )
    identity_fields = (
        "uma_model_sha256", "task_name", "protocol_sha256",
        "phase_cache_sha256", "structure_cache_sha256",
    )
    identities = {
        field: {str(records[entry_id].get(field)) for entry_id in expected_ids}
        for field in identity_fields
    }
    inconsistent = {field: values for field, values in identities.items() if len(values) != 1}
    if inconsistent:
        raise RuntimeError(f"relaxation identity mismatch: {inconsistent}")
    if next(iter(identities["phase_cache_sha256"])) != sha256_file(phase_path):
        raise RuntimeError("relaxation records do not match the phase cache")
    phase_records = {str(entry["entry_id"]): entry for entry in support}
    entries = []
    support_records = []
    for entry_id in sorted(expected_ids):
        composition = Composition(phase_records[entry_id]["composition"])
        energy_per_atom = float(records[entry_id]["final_energy_eV_atom"])
        entries.append(ComputedEntry(
            composition,
            energy_per_atom * float(composition.num_atoms),
            entry_id=entry_id,
        ))
        support_records.append({
            "entry_id": entry_id,
            "material_id": records[entry_id]["returned_material_id"],
            "composition": phase_records[entry_id]["composition"],
            "n_structure_atoms": records[entry_id]["n_atoms"],
            "final_energy_eV_atom": energy_per_atom,
            "relax_record_sha256": sha256_json(records[entry_id]),
        })
    diagram = PhaseDiagram(entries)
    pool = json.loads(composition_pool_path.read_text())
    compositions: dict[str, dict[str, Any]] = {}
    for item in pool["unique_compositions"]:
        composition = Composition(item["counts"])
        decomposition = diagram.get_decomposition(composition)
        reference = float(diagram.get_hull_energy(composition)) / float(
            composition.num_atoms
        )
        compositions[composition.reduced_formula] = {
            "counts": item["counts"],
            "reference_energy_eV_atom": reference,
            "decomposition": [
                {
                    "entry_id": str(entry.entry_id),
                    "formula": entry.composition.reduced_formula,
                    "fraction": float(fraction),
                }
                for entry, fraction in sorted(
                    decomposition.items(), key=lambda pair: str(pair[0].entry_id)
                )
            ],
        }
    stable = sorted(
        {
            str(entry.entry_id) for entry in diagram.stable_entries
        }
    )
    one_record = records[next(iter(sorted(expected_ids)))]
    payload: dict[str, Any] = {
        "version": HULL_CACHE_VERSION,
        "status": "complete",
        "definition": "E_hull=E_UMA-E_ref_UMA",
        "support_policy": SUPPORT_POLICY,
        "created_at": utc_now(),
        "phase_cache": str(phase_path.resolve()),
        "phase_cache_sha256": sha256_file(phase_path),
        "phase_cache_queried_at": phase_payload.get("queried_at"),
        "mp_compatibility": phase_payload.get("compatibility"),
        "composition_pool": str(composition_pool_path.resolve()),
        "composition_pool_sha256": sha256_file(composition_pool_path),
        "uma_model_sha256": one_record["uma_model_sha256"],
        "task_name": one_record["task_name"],
        "energy_normalization": "eV_per_atom",
        "relaxation_protocol": one_record["protocol"],
        "relaxation_protocol_sha256": one_record["protocol_sha256"],
        "structure_cache_sha256": one_record["structure_cache_sha256"],
        "expected_support_entry_count": len(expected_ids),
        "relaxed_support_entry_count": len(expected_ids),
        "extra_relax_record_count": len(set(records) - expected_ids),
        "uma_phase_diagram_stable_entry_ids": stable,
        "support_entries": support_records,
        "compositions": compositions,
    }
    payload["content_sha256"] = sha256_json(payload)
    return payload


def build_command(args: argparse.Namespace) -> None:
    payload = build_hull_payload(
        Path(args.phase_cache), Path(args.composition_pool), Path(args.records_root)
    )
    target = Path(args.out)
    atomic_json(target, payload)
    print(json.dumps({
        "output": str(target),
        "status": payload["status"],
        "support_entries": payload["relaxed_support_entry_count"],
        "stable_entries": len(payload["uma_phase_diagram_stable_entry_ids"]),
        "compositions": len(payload["compositions"]),
        "file_sha256": sha256_file(target),
    }, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch")
    fetch.add_argument("--phase-cache", required=True)
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--seed-cache")
    fetch.add_argument("--api-key-source-script")
    fetch.add_argument("--batch-size", type=int, default=50)
    fetch.add_argument("--prepare-only", action="store_true")
    fetch.set_defaults(function=fetch_command)

    relax = commands.add_parser("relax")
    relax.add_argument("--phase-cache", required=True)
    relax.add_argument("--structure-cache", required=True)
    relax.add_argument("--uma-checkpoint", required=True)
    relax.add_argument("--out-directory", required=True)
    relax.add_argument("--device", default="cuda")
    relax.add_argument("--task-name", default="omat")
    relax.add_argument("--shard-index", type=int, required=True)
    relax.add_argument("--shard-count", type=int, required=True)
    relax.add_argument("--fire-fmax", type=float, default=0.05)
    relax.add_argument("--fire-steps", type=int, default=1500)
    relax.add_argument("--bfgs-fmax", type=float, default=0.01)
    relax.add_argument("--bfgs-steps", type=int, default=1500)
    relax.add_argument("--resume", action="store_true")
    relax.set_defaults(function=relax_command)

    build = commands.add_parser("build")
    build.add_argument("--phase-cache", required=True)
    build.add_argument("--composition-pool", required=True)
    build.add_argument("--records-root", required=True)
    build.add_argument("--out", required=True)
    build.set_defaults(function=build_command)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

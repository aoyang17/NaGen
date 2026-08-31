"""Calibrate the UMA energy zero against low-energy MP reference structures."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from ._io import sha256_file as _sha256
from .energy_calibration import evaluate_zero_point
from .uma_guidance import checkpoint_sha256


_THERMO_SUFFIX = re.compile(r"-(?:GGA\+U|GGA|R2SCAN)$")


def material_id(entry_id: str) -> str:
    match = re.search(r"mp-\d+", str(entry_id))
    if match:
        return match.group(0)
    return _THERMO_SUFFIX.sub("", str(entry_id))


def select_low_energy_entries(
    payload: dict[str, Any], count: int, maximum_above_hull_eV_atom: float,
) -> list[dict[str, Any]]:
    """Select unique low-energy Na--Fe--P--O entries from the cached phase diagram."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    entries = [
        ComputedEntry(
            Composition(record["composition"]),
            float(record["energy_eV"]),
            entry_id=record["entry_id"],
        )
        for record in payload["entries"]
    ]
    diagram = PhaseDiagram(entries)
    candidates = []
    for entry in entries:
        symbols = {element.symbol for element in entry.composition.elements}
        if symbols != {"Na", "Fe", "P", "O"}:
            continue
        above_hull = float(diagram.get_e_above_hull(entry))
        if above_hull > maximum_above_hull_eV_atom:
            continue
        hull_total = float(diagram.get_hull_energy(entry.composition))
        candidates.append({
            "entry_id": str(entry.entry_id),
            "material_id": material_id(str(entry.entry_id)),
            "composition": entry.composition.reduced_formula,
            "E_MP_entry_eV_atom": float(entry.energy_per_atom),
            "E_MP_hull_eV_atom": hull_total / float(entry.composition.num_atoms),
            "mp_energy_above_hull_eV_atom": above_hull,
        })
    candidates.sort(key=lambda item: (
        item["mp_energy_above_hull_eV_atom"], item["material_id"]
    ))
    unique = []
    seen = set()
    for item in candidates:
        if item["material_id"] in seen:
            continue
        seen.add(item["material_id"])
        unique.append(item)
        if len(unique) == count:
            break
    if len(unique) < count:
        raise ValueError(
            f"only {len(unique)} unique references satisfy the low-energy filter; "
            f"{count} required"
        )
    return unique


def ase_atoms_from_structure_dict(payload: dict[str, Any]):
    """Construct ASE atoms from an ordered pymatgen structure dictionary."""
    from ase import Atoms

    symbols = []
    scaled_positions = []
    for site in payload["sites"]:
        occupied = [
            species for species in site["species"]
            if float(species.get("occu", 1.0)) > 0
        ]
        if len(occupied) != 1 or abs(float(occupied[0].get("occu", 1.0)) - 1.0) > 1e-8:
            raise ValueError("UMA calibration requires ordered MP reference structures")
        symbols.append(occupied[0]["element"])
        scaled_positions.append(site["abc"])
    return Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=payload["lattice"]["matrix"],
        pbc=payload["lattice"].get("pbc", [True, True, True]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-cache", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--structure-cache", required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--maximum-above-hull", type=float, default=0.020)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.count < 10:
        raise ValueError("at least 10 MP structures are required")

    phase_path = Path(args.phase_cache)
    phase_payload = json.loads(phase_path.read_text())
    structure_path = Path(args.structure_cache)
    if structure_path.exists():
        cached = json.loads(structure_path.read_text())
        structures = cached["structures"]
        selected = cached.get("selected_references")
        if selected is None:
            selected = select_low_energy_entries(
                phase_payload, args.count, args.maximum_above_hull
            )
            cached["selected_references"] = selected
            temporary = structure_path.with_suffix(structure_path.suffix + ".tmp")
            temporary.write_text(json.dumps(cached, indent=2) + "\n")
            temporary.replace(structure_path)
        source = "offline_cache"
    else:
        selected = select_low_energy_entries(
            phase_payload, args.count, args.maximum_above_hull
        )
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            raise RuntimeError("MP_API_KEY is required to build the one-time structure cache")
        from mp_api.client import MPRester
        identifiers = [item["material_id"] for item in selected]
        structures = {}
        returned_aliases = {}
        with MPRester(api_key, use_document_model=False) as rester:
            for requested_id in identifiers:
                documents = rester.materials.summary.search(
                    material_ids=[requested_id],
                    fields=["material_id", "structure"],
                    all_fields=False,
                )
                if len(documents) != 1:
                    raise RuntimeError(
                        f"MP returned {len(documents)} structures for {requested_id}"
                    )
                document = documents[0]
                structure = document["structure"]
                structures[requested_id] = (
                    structure.as_dict() if hasattr(structure, "as_dict") else structure
                )
                returned_aliases[requested_id] = str(document["material_id"])
        structure_path.parent.mkdir(parents=True, exist_ok=True)
        structure_payload = {
            "version": "mp-reference-structures-v1",
            "phase_cache_sha256": _sha256(phase_path),
            "material_ids": identifiers,
            "returned_aliases": returned_aliases,
            "selected_references": selected,
            "structures": structures,
        }
        temporary = structure_path.with_suffix(structure_path.suffix + ".tmp")
        temporary.write_text(json.dumps(structure_payload, indent=2) + "\n")
        temporary.replace(structure_path)
        source = "materials_project_api"

    if args.prepare_only:
        print(json.dumps({
            "n_structures": len(structures),
            "selected_references": len(selected),
            "structure_cache": str(structure_path),
        }, indent=2))
        return

    from fairchem.core import FAIRChemCalculator

    calculator = FAIRChemCalculator.from_model_checkpoint(
        args.uma_checkpoint, task_name=args.task_name, device=args.device
    )
    records = []
    for reference in selected:
        atoms = ase_atoms_from_structure_dict(structures[reference["material_id"]])
        atoms.calc = calculator
        e_uma = float(atoms.get_potential_energy()) / len(atoms)
        records.append({
            "composition": reference["composition"],
            "mp_entry_id": reference["entry_id"],
            "mp_material_id": reference["material_id"],
            "n_atoms": len(atoms),
            "atomic_numbers": atoms.numbers.tolist(),
            "E_UMA_eV_atom": e_uma,
            "E_MP_eV_atom": reference["E_MP_entry_eV_atom"],
            "E_MP_hull_eV_atom": reference["E_MP_hull_eV_atom"],
            "reference_E_hull_eV_atom": e_uma - reference["E_MP_hull_eV_atom"],
            "mp_energy_above_hull_eV_atom": reference[
                "mp_energy_above_hull_eV_atom"
            ],
        })
    report = evaluate_zero_point(
        records, checkpoint_sha256(args.uma_checkpoint)
    )
    report.update({
        "phase_cache": str(phase_path.resolve()),
        "phase_cache_sha256": _sha256(phase_path),
        "phase_cache_queried_at": phase_payload.get("queried_at"),
        "mp_compatibility": phase_payload.get("compatibility"),
        "structure_cache": str(structure_path.resolve()),
        "structure_source": source,
        "selection_maximum_mp_e_above_hull_eV_atom": args.maximum_above_hull,
        "task_name": args.task_name,
        "device": args.device,
    })
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)
    print(json.dumps({
        "n_structures": report["n_structures"],
        "median_absolute_difference_eV_atom": report[
            "median_absolute_difference_eV_atom"
        ],
        "passed": report["passed"],
        "blocking": report["blocking"],
        "output": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()

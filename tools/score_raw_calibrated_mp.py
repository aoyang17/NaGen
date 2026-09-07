"""Score the first N raw records with the reference script's mixed UMA/MP hull.

The raw records already contain an UMA-relaxed structure and UMA energy.  The
MP phase cache contains MP2020-compatible energies, so this reproduces

    max(0, E_UMA + candidate_MP2020_correction - E_MP_hull)

without contacting the MP API or modifying the read-only source dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from nagen.inverse.calibrated_mp_hull import candidate_correction_eV_atom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--phase-cache", required=True)
    parser.add_argument("--elements", nargs="+")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()

    records = []
    with open(args.input) as handle:
        for source_index, line in enumerate(handle):
            record = json.loads(line)
            structure = Structure.from_dict(record["structure"])
            element_set = {element.symbol for element in structure.composition.elements}
            if args.elements and element_set != set(args.elements):
                continue
            record["_source_index"] = source_index
            records.append(record)
            if len(records) >= args.count:
                break
    if len(records) != args.count:
        raise RuntimeError(f"requested {args.count} rows, found {len(records)}")

    phase_payload = json.loads(Path(args.phase_cache).read_text())
    if phase_payload.get("compatibility") != "MaterialsProject2020Compatibility":
        raise ValueError("phase cache is not MaterialsProject2020-compatible")
    if args.elements and set(args.elements) != set(phase_payload.get("chemical_system", [])):
        raise ValueError("requested elements do not match the phase cache")
    phase_entries = [
        ComputedEntry(item["composition"], float(item["energy_eV"]), entry_id=item["entry_id"])
        for item in phase_payload["entries"]
    ]
    diagram = PhaseDiagram(phase_entries)

    rows = []
    for index, record in enumerate(records):
            structure = Structure.from_dict(record["structure"])
            composition = structure.composition
            uma = float(record["optimized_energy_per_atom"])
            correction = candidate_correction_eV_atom(composition)
            mp_reference = float(diagram.get_hull_energy(composition)) / float(composition.num_atoms)
            raw = uma + correction - mp_reference
            rows.append({
                "index": index,
                "source_index": record["_source_index"],
                "material_id": record.get("material_id"),
                "formula": record.get("formula") or composition.reduced_formula,
                "n_atoms": int(composition.num_atoms),
                "E_UMA_eV_atom": uma,
                "MP2020_candidate_correction_eV_atom": correction,
                "E_MP_hull_reference_eV_atom": mp_reference,
                "raw_E_hull_eV_atom": raw,
                "E_hull_eV_atom": max(0.0, raw),
                "below_mp_hull": raw < 0.0,
            })
    payload = {
        "version": "raw-calibrated-mp2020-hull-v1",
        "definition": "max(0,E_UMA+MP2020_candidate_correction-E_MP_hull_reference)",
        "warning": "mixed UMA candidate / MP DFT reference energy scale; diagnostic only",
        "input": str(Path(args.input)),
        "phase_cache": str(Path(args.phase_cache)),
        "phase_entry_count": len(phase_entries),
        "selected_elements": args.elements,
        "count": len(rows),
        "rows": rows,
    }
    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

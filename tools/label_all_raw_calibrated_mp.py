"""Label every raw record with mixed UMA/MP2020 hull quantities.

Definitions (all energies are eV/atom):

    E_hull_raw = E_UMA + MP2020_candidate_correction - E_MP
    E_hull = max(0, E_hull_raw)

Here E_MP is the composition-specific energy on a phase diagram assembled
from MP2020-compatible Materials Project entries.  It is not an MP energy for
the candidate structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from monty.serialization import dumpfn, loadfn
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry

from nagen.inverse.calibrated_mp_hull import candidate_correction_eV_atom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--jsonl-out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--summary-out", required=True)
    return parser.parse_args()


def chemical_system(structure: Structure) -> str:
    return "-".join(sorted(element.symbol for element in structure.composition.elements))


def load_records(path: Path) -> tuple[list[dict], dict[str, list[str]]]:
    records: list[dict] = []
    systems: dict[str, list[str]] = {}
    with path.open() as handle:
        for source_index, line in enumerate(handle):
            raw = json.loads(line)
            structure = Structure.from_dict(raw["structure"])
            system = chemical_system(structure)
            systems[system] = [e.symbol for e in structure.composition.elements]
            records.append({
                "source_index": source_index,
                "raw": raw,
                "structure": structure,
                "chemical_system": system,
            })
    return records, systems


def phase_diagrams(systems: dict[str, list[str]], cache_dir: Path) -> tuple[dict[str, PhaseDiagram], dict[str, int]]:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY is not set")
    cache_dir.mkdir(parents=True, exist_ok=True)
    diagrams: dict[str, PhaseDiagram] = {}
    counts: dict[str, int] = {}
    with MPRester(api_key, use_progress_bar=False) as mpr:
        for position, system in enumerate(sorted(systems), start=1):
            cache_path = cache_dir / f"{system}.json.gz"
            if cache_path.exists():
                cached = loadfn(cache_path)
                entries = [
                    ComputedEntry(
                        Composition(item["composition"]),
                        float(item["energy_eV"]),
                        entry_id=item.get("entry_id"),
                    )
                    for item in cached["entries"]
                ]
                source = "cache"
            else:
                # Pin the legacy mixed GGA/GGA+U thermodynamic dataset.  New
                # mp-api releases otherwise silently include r2SCAN entries,
                # which changes E_MP relative to the reference workflow.
                entries = mpr.get_entries_in_chemsys(
                    systems[system],
                    additional_criteria={"thermo_types": ["GGA_GGA+U"]},
                )
                # Keep a lean, version-stable cache.  Recent mp-api versions
                # attach decomposition dictionaries keyed by Element objects;
                # those metadata are irrelevant to PhaseDiagram and cannot be
                # serialized as JSON by current monty.
                dumpfn({
                    "chemical_system": system,
                    "thermo_type": "GGA_GGA+U",
                    "compatibility": "MaterialsProject2020Compatibility",
                    "entries": [
                        {
                            "composition": {
                                str(element): float(amount)
                                for element, amount in entry.composition.items()
                            },
                            "energy_eV": float(entry.energy),
                            "entry_id": str(entry.entry_id),
                        }
                        for entry in entries
                    ],
                }, cache_path)
                source = "api"
            diagrams[system] = PhaseDiagram(entries)
            counts[system] = len(entries)
            print(json.dumps({
                "phase_system": system,
                "position": position,
                "total_systems": len(systems),
                "entries": len(entries),
                "source": source,
            }), flush=True)
    return diagrams, counts


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    jsonl_path = Path(args.jsonl_out)
    csv_path = Path(args.csv_out)
    summary_path = Path(args.summary_out)
    records, systems = load_records(input_path)
    diagrams, phase_entry_counts = phase_diagrams(systems, Path(args.cache_dir))

    fieldnames = [
        "source_index", "material_id", "formula", "chemical_system", "n_atoms",
        "E_UMA", "E_MP", "E_hull", "MP2020_candidate_correction",
        "E_hull_raw", "below_MP_hull", "upstream_error_present",
    ]
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    values: list[float] = []
    raw_negative = 0
    upstream_errors = 0
    system_rows = Counter()
    with jsonl_path.open("w") as json_handle, csv_path.open("w", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in records:
            raw = item["raw"]
            structure = item["structure"]
            composition = structure.composition
            system = item["chemical_system"]
            e_uma = float(raw["optimized_energy_per_atom"])
            correction = candidate_correction_eV_atom(composition)
            e_mp = float(diagrams[system].get_hull_energy(composition)) / float(composition.num_atoms)
            e_hull_raw = e_uma + correction - e_mp
            e_hull = max(0.0, e_hull_raw)
            has_error = bool(raw.get("error_msg") or raw.get("traceback"))
            row = {
                "source_index": item["source_index"],
                "material_id": raw.get("material_id"),
                "formula": raw.get("formula") or composition.reduced_formula,
                "chemical_system": system,
                "n_atoms": int(composition.num_atoms),
                "E_UMA": e_uma,
                "E_MP": e_mp,
                "E_hull": e_hull,
                "MP2020_candidate_correction": correction,
                "E_hull_raw": e_hull_raw,
                "below_MP_hull": e_hull_raw < 0.0,
                "upstream_error_present": has_error,
            }
            json_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.writerow(row)
            values.append(e_hull)
            raw_negative += int(e_hull_raw < 0.0)
            upstream_errors += int(has_error)
            system_rows[system] += 1

    ordered = sorted(values)
    summary = {
        "version": "all-raw-calibrated-mp2020-labels-v1",
        "definition": "E_hull=max(0,E_UMA+MP2020_candidate_correction-E_MP)",
        "E_MP_semantics": "composition-specific MP2020 phase-diagram hull reference, eV/atom",
        "warning": "mixed UMA candidate / MP DFT reference energy scale",
        "input": str(input_path),
        "count": len(values),
        "chemical_system_count": len(systems),
        "chemical_system_rows": dict(sorted(system_rows.items())),
        "phase_entry_counts": phase_entry_counts,
        "upstream_error_present_count": upstream_errors,
        "below_MP_hull_clipped_count": raw_negative,
        "E_hull_statistics": {
            "min": ordered[0],
            "mean": sum(ordered) / len(ordered),
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "count_le_0.15": sum(v <= 0.15 for v in ordered),
        },
        "outputs": {"jsonl": str(jsonl_path), "csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Calculate MP2020-referenced hull distances for original UMA records.

The candidate energy is the stored UMA energy.  MaterialsProject2020Compatibility
is applied exactly as in the upstream ``relaxation_n_ehull.py`` implementation,
while MP entries and PhaseDiagram objects are cached per chemical system.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from monty.serialization import dumpfn, loadfn
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.core import Composition


UMA_HUBBARDS = {
    "Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Mo": 4.38,
    "Ni": 6.2, "V": 3.25, "W": 6.2,
}
POTCAR_MAP = {
    "C": "C", "Co": "Co", "Cr": "Cr_pv", "F": "F", "Fe": "Fe_pv",
    "Mn": "Mn_pv", "Mo": "Mo_pv", "N": "N", "Na": "Na_pv",
    "Ni": "Ni_pv", "O": "O", "P": "P", "Ti": "Ti_pv",
    "V": "V_pv", "W": "W_sv",
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY is required")

    started = time.time()
    records = []
    with Path(args.raw).open() as handle:
        for index, line in enumerate(handle):
            if index >= args.count:
                break
            row = json.loads(line)
            records.append({
                "source_index": index,
                "material_id": row["material_id"],
                "formula": row["formula"],
                "composition": Composition(row["formula"]),
                "E_UMA_eV_atom": float(row["optimized_energy_per_atom"]),
            })

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    diagrams: dict[str, PhaseDiagram] = {}
    cache_sources: dict[str, str] = {}
    for chemsys in sorted({r["composition"].chemical_system for r in records}):
        cache_path = cache_dir / f"{chemsys}.json"
        if cache_path.exists():
            entries = loadfn(cache_path)
            cache_sources[chemsys] = "disk"
        else:
            elements = chemsys.split("-")
            with MPRester(api_key, use_progress_bar=False) as rester:
                entries = rester.get_entries_in_chemsys(elements)
            dumpfn(entries, cache_path)
            cache_sources[chemsys] = "materials_project_api"
        diagrams[chemsys] = PhaseDiagram(entries)

    compatibility = MaterialsProject2020Compatibility()
    output_rows = []
    for row in records:
        composition = row.pop("composition")
        atom_count = float(composition.num_atoms)
        raw_entry = ComputedEntry(
            composition,
            row["E_UMA_eV_atom"] * atom_count,
            parameters={
                "run_type": "GGA+U" if any(
                    element.symbol in UMA_HUBBARDS for element in composition.elements
                ) else "GGA",
                "hubbards": {
                    element.symbol: UMA_HUBBARDS[element.symbol]
                    for element in composition.elements
                    if element.symbol in UMA_HUBBARDS
                },
                "potcar_symbols": [
                    f"PAW_PBE {POTCAR_MAP.get(element.symbol, element.symbol)}"
                    for element in composition.elements
                ],
            },
        )
        corrected = compatibility.process_entry(raw_entry) or raw_entry
        correction = (float(corrected.energy) - float(raw_entry.energy)) / atom_count
        diagram = diagrams[composition.chemical_system]
        reference = float(diagram.get_hull_energy(composition)) / atom_count
        raw_e_hull = float(corrected.energy) / atom_count - reference
        output_rows.append({
            **row,
            "chemical_system": composition.chemical_system,
            "n_atoms_reduced_composition": atom_count,
            "MP2020_candidate_correction_eV_atom": correction,
            "E_MP_hull_eV_atom": reference,
            "raw_E_hull_eV_atom": raw_e_hull,
            "E_hull_eV_atom": max(0.0, raw_e_hull),
            "passes_0p15": raw_e_hull <= 0.15,
        })

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows_path = target.with_suffix(".jsonl")
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in output_rows))
    hull = [row["E_hull_eV_atom"] for row in output_rows]
    raw_hull = [row["raw_E_hull_eV_atom"] for row in output_rows]
    summary = {
        "definition": (
            "E_hull=max(0,E_UMA+MP2020_candidate_correction-E_MP_hull)"
        ),
        "n_records": len(output_rows),
        "n_chemical_systems": len(diagrams),
        "cache_sources": cache_sources,
        "E_hull_eV_atom": {
            "min": min(hull), "p10": _percentile(hull, 0.10),
            "median": _percentile(hull, 0.50),
            "p90": _percentile(hull, 0.90), "max": max(hull),
            "mean": sum(hull) / len(hull),
        },
        "raw_E_hull_eV_atom": {
            "min": min(raw_hull), "max": max(raw_hull),
        },
        "passes_0p15": sum(row["passes_0p15"] for row in output_rows),
        "pass_rate": sum(row["passes_0p15"] for row in output_rows) / len(output_rows),
        "elapsed_seconds": time.time() - started,
        "rows": str(rows_path.resolve()),
    }
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

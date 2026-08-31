"""Build fixed-composition hull references from an offline MP phase cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from ._io import sha256_file as _sha256

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-cache", required=True)
    parser.add_argument("--composition-pool", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    phase_path = Path(args.phase_cache)
    pool_path = Path(args.composition_pool)
    phase_payload = json.loads(phase_path.read_text())
    pool = json.loads(pool_path.read_text())
    entries = [
        ComputedEntry(
            Composition(record["composition"]),
            float(record["energy_eV"]),
            entry_id=record["entry_id"],
        )
        for record in phase_payload["entries"]
    ]
    diagram = PhaseDiagram(entries)
    records = {}
    for item in pool["unique_compositions"]:
        composition = Composition(item["counts"])
        hull_total = float(diagram.get_hull_energy(composition))
        records[composition.reduced_formula] = {
            "hull_energy_eV_atom": hull_total / float(composition.num_atoms),
            "counts": item["counts"],
            "phase_cache_sha256": _sha256(phase_path),
            "compatibility": phase_payload["compatibility"],
            "queried_at": phase_payload["queried_at"],
            "n_compatible_entries": phase_payload["n_compatible_entries"],
        }
    output = {
        "version": "mp-hull-cache-v1",
        "phase_cache": str(phase_path),
        "phase_cache_sha256": _sha256(phase_path),
        "composition_pool_sha256": _sha256(pool_path),
        "compositions": records,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps({
        "output": str(target), "compositions": len(records),
        "minimum_hull_energy_eV_atom": min(
            record["hull_energy_eV_atom"] for record in records.values()
        ),
        "maximum_hull_energy_eV_atom": max(
            record["hull_energy_eV_atom"] for record in records.values()
        ),
    }, indent=2))


if __name__ == "__main__":
    main()

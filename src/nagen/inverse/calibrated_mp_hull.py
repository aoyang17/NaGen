"""MP2020-calibrated hull evaluation matching the project reference script.

The cached Materials Project entries are already compatibility-processed.  They
must therefore be inserted into the phase diagram with ``energy_eV`` exactly as
stored.  Only the candidate UMA energy receives the MP2020 compatibility
correction constructed from the OMat24 Hubbard/POTCAR metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ._io import atomic_json, sha256_file


UMA_HUBBARDS = {
    "Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Mo": 4.38,
    "Ni": 6.2, "V": 3.25, "W": 6.2,
}

POTCAR_MAP = {
    # Relevant subset of the OMat24 POTCAR mapping used by the reference
    # script.  The transition-metal variants matter when MP2020 validates a
    # candidate before applying its corrections.
    "C": "C", "F": "F", "Fe": "Fe_pv", "Mn": "Mn_pv", "N": "N",
    "Na": "Na_pv", "O": "O", "P": "P", "Ti": "Ti_pv", "V": "V_pv",
}


class MissingCalibratedHullReferenceError(KeyError):
    pass


def candidate_parameters(composition: Any) -> dict[str, Any]:
    """Return the exact candidate metadata used by the reference script."""
    from pymatgen.core import Composition

    comp = Composition(composition)
    hubbards = {
        element.symbol: UMA_HUBBARDS[element.symbol]
        for element in comp.elements if element.symbol in UMA_HUBBARDS
    }
    return {
        "run_type": "GGA+U" if hubbards else "GGA",
        "hubbards": hubbards,
        "potcar_symbols": [
            f"PAW_PBE {POTCAR_MAP.get(element.symbol, element.symbol)}"
            for element in comp.elements
        ],
    }


def candidate_correction_eV_atom(composition: Any) -> float:
    """MP2020 correction per atom for an UMA candidate of fixed composition."""
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedEntry

    entry = ComputedEntry(composition, 0.0, parameters=candidate_parameters(composition))
    processed = MaterialsProject2020Compatibility().process_entry(entry)
    corrected = processed if processed is not None else entry
    return float(corrected.energy) / float(corrected.composition.num_atoms)


def build_calibrated_cache(
    phase_cache: str | Path,
    compositions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build fixed-composition MP references without reapplying corrections."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    phase_path = Path(phase_cache)
    payload = json.loads(phase_path.read_text())
    if payload.get("compatibility") != "MaterialsProject2020Compatibility":
        raise ValueError("phase cache is not MP2020-compatible")
    entries = [
        ComputedEntry(
            Composition(record["composition"]),
            float(record["energy_eV"]),
            entry_id=record["entry_id"],
        )
        for record in payload["entries"]
    ]
    diagram = PhaseDiagram(entries)
    records: dict[str, dict[str, Any]] = {}
    for counts in compositions:
        composition = Composition(counts)
        reference = float(diagram.get_hull_energy(composition)) / float(
            composition.num_atoms
        )
        records[composition.reduced_formula] = {
            "counts": dict(counts),
            "mp_hull_reference_eV_atom": reference,
            "mp2020_candidate_correction_eV_atom": (
                candidate_correction_eV_atom(composition)
            ),
        }
    return {
        "version": "calibrated-mp2020-hull-cache-v1",
        "status": "complete",
        "definition": (
            "max(0,E_UMA+MP2020_candidate_correction-MP_hull_reference)"
        ),
        "phase_cache": str(phase_path.resolve()),
        "phase_cache_sha256": sha256_file(phase_path),
        "compatibility": "MaterialsProject2020Compatibility",
        "mp_entries_energy_semantics": "already_compatible_energy_eV_no_recorrection",
        "entry_count": len(entries),
        "compositions": records,
    }


class CalibratedMPHullCache:
    """Read-only fixed-composition cache for differentiable D-Flow guidance."""

    def __init__(self, path: str | Path) -> None:
        from pymatgen.core import Composition

        self._composition_type = Composition
        self.path = Path(path)
        # Some shared filesystems briefly expose a just-renamed file as empty.
        # Retry the read rather than losing an entire GPU shard during startup.
        last_error: json.JSONDecodeError | None = None
        for attempt in range(8):
            try:
                self.data = json.loads(self.path.read_text())
                break
            except json.JSONDecodeError as error:
                last_error = error
                if attempt == 7:
                    raise
                time.sleep(0.10 * (attempt + 1))
        else:  # pragma: no cover - loop either succeeds or raises above
            raise last_error  # type: ignore[misc]
        if self.data.get("version") != "calibrated-mp2020-hull-cache-v1":
            raise ValueError("not a calibrated MP2020 hull cache")
        if self.data.get("status") != "complete":
            raise ValueError("calibrated MP2020 hull cache is incomplete")
        if self.data.get("mp_entries_energy_semantics") != (
            "already_compatible_energy_eV_no_recorrection"
        ):
            raise ValueError("MP entry energy semantics are not explicit")

    def constants(self, composition: Any) -> tuple[float, float]:
        key = self._composition_type(composition).reduced_formula
        record = self.data.get("compositions", {}).get(key)
        if record is None:
            raise MissingCalibratedHullReferenceError(key)
        return (
            float(record["mp_hull_reference_eV_atom"]),
            float(record["mp2020_candidate_correction_eV_atom"]),
        )

    def raw_e_hull(self, composition: Any, energy_eV_atom: float) -> float:
        reference, correction = self.constants(composition)
        return float(energy_eV_atom) + correction - reference

    def e_hull(self, composition: Any, energy_eV_atom: float) -> float:
        return max(0.0, self.raw_e_hull(composition, energy_eV_atom))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-cache", required=True)
    parser.add_argument("--composition-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    composition_payload = json.loads(Path(args.composition_json).read_text())
    rows = composition_payload.get("unique_compositions", composition_payload)
    compositions = [row.get("counts", row) for row in rows]
    payload = build_calibrated_cache(args.phase_cache, compositions)
    atomic_json(args.out, payload, sort_keys=True)
    print(json.dumps({
        "output": str(Path(args.out)),
        "entry_count": payload["entry_count"],
        "composition_count": len(payload["compositions"]),
    }, indent=2))


if __name__ == "__main__":
    main()

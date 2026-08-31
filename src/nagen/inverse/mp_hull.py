"""Offline-first Materials Project convex-hull reference cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


class MissingCompositionError(KeyError):
    pass


class MPHullCache:
    """Read-only during guidance; cache files never contain credentials."""

    def __init__(self, path: str, offline: bool = True) -> None:
        self.path = Path(path)
        self.offline = offline
        self.data: dict[str, Any] = {"version": "mp-hull-cache-v1", "compositions": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    @staticmethod
    def key(composition: Any) -> str:
        from pymatgen.core import Composition
        return Composition(composition).reduced_formula

    def get(self, composition: Any) -> float:
        key = self.key(composition)
        record = self.data["compositions"].get(key)
        if record is None:
            mode = "offline cache" if self.offline else "cache"
            raise MissingCompositionError(f"{key} is absent from {mode}")
        return float(record["hull_energy_eV_atom"])

    def put_record(self, composition: Any, record: dict[str, Any]) -> None:
        if self.offline:
            raise RuntimeError("cannot mutate an offline MP cache")
        sanitized = dict(record)
        for key in tuple(sanitized):
            if "api" in key.lower() and "key" in key.lower():
                sanitized.pop(key)
        self.data["compositions"][self.key(composition)] = sanitized

    def build_from_entries(
        self, composition: Any, compatible_entries: Iterable[Any],
        compatibility: str, queried_at: str,
    ) -> float:
        """Cache a PhaseDiagram reference from already-compatible MP entries."""
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.core import Composition
        entries = list(compatible_entries)
        phase_diagram = PhaseDiagram(entries)
        target = Composition(composition)
        hull_total = float(phase_diagram.get_hull_energy(target))
        value = hull_total / float(target.num_atoms)
        self.put_record(target, {
            "hull_energy_eV_atom": value,
            "entry_ids": [str(getattr(entry, "entry_id", "")) for entry in entries],
            "compatibility": compatibility, "queried_at": queried_at,
        })
        return value

    def save(self) -> str:
        if self.offline:
            raise RuntimeError("cannot save an offline MP cache")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(self.data, indent=2, sort_keys=True) + "\n").encode()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(self.path)
        return hashlib.sha256(encoded).hexdigest()


def apply_mp2020_compatibility(entries: Iterable[Any]) -> tuple[list[Any], str]:
    """Compatibility is applied to MP reference entries, never UMA candidates."""
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    compatibility = MaterialsProject2020Compatibility()
    processed = compatibility.process_entries(list(entries), clean=True)
    return list(processed), type(compatibility).__name__

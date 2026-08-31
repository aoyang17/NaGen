"""Offline UMA same-energy-scale convex-hull reference cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MissingUMAHullReferenceError(KeyError):
    pass


class UMAHullCache:
    """Read a fail-closed cache built from UMA-scored MP competing structures."""

    def __init__(
        self,
        path: str,
        expected_model_sha256: str | None = None,
        expected_task_name: str | None = None,
    ) -> None:
        from pymatgen.core import Composition

        self._composition_type = Composition
        self.path = Path(path)
        payload: dict[str, Any] = json.loads(self.path.read_text())
        if payload.get("version") != "uma-hull-cache-v1":
            raise ValueError("not a UMA same-energy-scale hull cache")
        if payload.get("status") != "complete":
            raise ValueError("UMA hull cache is not complete")
        if payload.get("support_policy") != "all_mp2020_compatible_entries":
            raise ValueError("UMA hull cache has an unsupported phase policy")
        expected = payload.get("expected_support_entry_count")
        relaxed = payload.get("relaxed_support_entry_count")
        if not isinstance(expected, int) or expected <= 0 or relaxed != expected:
            raise ValueError("UMA hull cache does not have complete relaxed support")
        if expected_model_sha256 is not None and (
            payload.get("uma_model_sha256") != expected_model_sha256
        ):
            raise ValueError("UMA hull cache checkpoint hash mismatch")
        if expected_task_name is not None and (
            payload.get("task_name") != expected_task_name
        ):
            raise ValueError("UMA hull cache task name mismatch")
        self.data = payload

    def get(self, composition: Any) -> float:
        key = self._composition_type(composition).reduced_formula
        record = self.data.get("compositions", {}).get(key)
        if record is None:
            raise MissingUMAHullReferenceError(
                f"{key} is absent from the offline UMA hull cache"
            )
        value = record.get("reference_energy_eV_atom")
        if value is None:
            raise MissingUMAHullReferenceError(
                f"{key} has no UMA hull reference energy"
            )
        return float(value)

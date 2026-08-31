"""Blocking UMA--MP energy-scale compatibility gate."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def evaluate_zero_point(
    records: list[dict[str, Any]], model_sha256: str,
    threshold_eV_atom: float = 0.150,
) -> dict[str, Any]:
    if len(records) < 10:
        raise ValueError("energy zero-point validation requires at least 10 MP structures")
    required = {"composition", "mp_entry_id", "E_UMA_eV_atom", "E_MP_eV_atom"}
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"calibration record is missing {sorted(missing)}")
    differences = [
        float(record["E_UMA_eV_atom"]) - float(record["E_MP_eV_atom"])
        for record in records
    ]
    median_absolute = statistics.median(abs(value) for value in differences)
    return {
        "version": "uma-mp-energy-scale-diagnostic-v2", "uma_model_sha256": model_sha256,
        "n_structures": len(records), "records": records,
        "differences_eV_atom": differences,
        "median_absolute_difference_eV_atom": median_absolute,
        "threshold_eV_atom": threshold_eV_atom,
        "passed": median_absolute <= threshold_eV_atom,
        "blocking": median_absolute > threshold_eV_atom,
        "experiment_may_continue": median_absolute <= threshold_eV_atom,
        "offset_fit": False,
        "policy": "stop_on_incompatible_energy_zero_without_offset_fit",
    }


def save_zero_point_report(report: dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

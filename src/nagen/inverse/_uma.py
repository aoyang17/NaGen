"""Lazy UMA construction shared by relaxation entry points."""

from __future__ import annotations


def load_calculator(checkpoint: str, device: str, task_name: str):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(
        checkpoint, inference_settings="default", device=device
    )
    return FAIRChemCalculator(predictor, task_name=task_name)

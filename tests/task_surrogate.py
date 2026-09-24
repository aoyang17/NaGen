"""Deterministic CPU surrogate used by task-oriented generation tests."""

from __future__ import annotations


class TaskSurrogate:
    provenance = {
        "backend": "task-test",
        "energy_unit": "eV/atom",
        "gradient_path": "torch-autograd",
    }

    def guidance_energy(self, atomic_numbers):
        def energy(frac, lattice):
            return frac.square().mean() + 0.01 * lattice.square().mean() + atomic_numbers.sum() * 0.0
        return energy

    def evaluator(self, inference_settings="default"):
        raise NotImplementedError("task test surrogate is guidance-only")


def load_dummy(spec):
    return TaskSurrogate()

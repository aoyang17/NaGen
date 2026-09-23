from __future__ import annotations

import torch

from shootingcsp.surrogate import (
    SurrogateSpec,
    available_surrogates,
    load_surrogate,
    register_surrogate,
)


class DummySurrogate:
    provenance = {"backend": "dummy"}

    def guidance_energy(self, atomic_numbers):
        def energy(frac, lattice):
            return frac.square().mean() + lattice.square().mean() + atomic_numbers.sum() * 0.0
        return energy

    def evaluator(self, inference_settings="default"):
        raise NotImplementedError


def test_surrogate_registry_and_protocol() -> None:
    register_surrogate("dummy-test", lambda spec: DummySurrogate(), replace=True)
    assert "uma" in available_surrogates()
    surrogate = load_surrogate(SurrogateSpec("dummy-test"))
    energy = surrogate.guidance_energy(torch.tensor([11, 26, 15, 8]))
    value = energy(torch.zeros(1, 4, 3), torch.zeros(1, 3, 3))
    assert value.shape == ()

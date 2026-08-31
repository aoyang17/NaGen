from __future__ import annotations

import math
import unittest

import torch

from nagen.inverse.novelty import DescriptorConfig, NoveltyIndex, crystal_descriptor


class NoveltyDescriptorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.types = torch.tensor([[1, 2, 3, 3]], dtype=torch.long)
        self.frac = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.1, 0.2, 0.3], [0.8, 0.7, 0.6]]]
        )
        self.lattice = torch.tensor(
            [[[5.0, 0.0, 0.0], [0.5, 5.5, 0.0], [0.2, 0.3, 6.0]]]
        )
        self.mask = torch.ones(1, 4, dtype=torch.bool)
        self.config = DescriptorConfig(rdf_bins=8)

    def descriptor(self, types=None, frac=None, lattice=None):
        return crystal_descriptor(
            self.types if types is None else types,
            self.frac if frac is None else frac,
            self.lattice if lattice is None else lattice,
            self.mask,
            self.config,
        )

    def test_permutation_and_translation_invariance(self) -> None:
        reference = self.descriptor()
        permutation = torch.tensor([2, 0, 3, 1])
        translated = torch.remainder(self.frac[:, permutation] + 0.371, 1.0)
        candidate = self.descriptor(self.types[:, permutation], translated)
        torch.testing.assert_close(reference, candidate, atol=2e-6, rtol=2e-6)

    def test_cartesian_rotation_invariance(self) -> None:
        angle = math.pi / 3.0
        rotation = torch.tensor(
            [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        rotated = self.lattice @ rotation
        torch.testing.assert_close(
            self.descriptor(), self.descriptor(lattice=rotated), atol=2e-6, rtol=2e-6
        )

    def test_smooth_min_has_finite_nonzero_gradient(self) -> None:
        reference = self.descriptor().detach()
        payload = {
            "config": self.config.__dict__, "descriptors": torch.cat((reference, reference + 0.2)),
            "mean": torch.zeros_like(reference[0]), "std": torch.ones_like(reference[0]),
            "ids": ["a", "b"], "formulas": ["a", "b"],
        }
        index = NoveltyIndex(payload)
        frac = self.frac.clone().requires_grad_(True)
        score = index.smooth_score(self.descriptor(frac=frac), top_k=2).sum()
        score.backward()
        self.assertTrue(torch.isfinite(frac.grad).all())
        self.assertGreater(float(frac.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()

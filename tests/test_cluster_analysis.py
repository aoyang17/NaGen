import math
import unittest

from nagen.analysis.cluster import dist_frac, percentile


class ClusterAnalysisTests(unittest.TestCase):
    def test_periodic_distance_uses_minimum_image(self):
        lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        self.assertTrue(math.isclose(dist_frac([0.95, 0, 0], [0.05, 0, 0], lattice), 1.0))

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([0.0, 10.0], 0.25), 2.5)


if __name__ == "__main__":
    unittest.main()

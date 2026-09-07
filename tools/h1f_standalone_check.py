"""Standalone check of the H1f premise -- stdlib only, no torch, no numpy, no dataset.

`tools/p1_5_source_mismatch.py` answers the same question against the real
packed dataset and should be preferred when the environment is available.  This
file exists because it has zero dependencies and runs under any Python 3, which
makes it usable when the full environment is not reachable.

The claim under test
--------------------
H1f holds that the model never saw sub-Angstrom contacts in training, because
the training X source is ``wrap(target + 0.5 * eps)`` while inference draws
``Uniform[0,1)^3``.

Analytically that should fail.  For a pair (i, j) the source offset is
``wrap(d_ij + sigma * (eps_i - eps_j))``, with per-coordinate noise scale
``sigma * sqrt(2) = 0.707``.  A wrapped normal deviates from uniform by
``2 * exp(-2 * pi^2 * s^2)`` = 1.0e-4 there, so the pair offset is uniform to
within 0.01% and the two sources should show the SAME close-contact rate.

The script also reports the analytic prediction for the minimum interatomic
distance of a uniform cloud.  Expecting one pair inside radius r gives

    N (N - 1) / 2 * (4/3) pi r^3 / V = 1   =>   r = (3 V / (2 pi N (N-1)))^(1/3)

If the observed generated median (0.33 A) sits at this scale, the generated
clouds are statistically indistinguishable from uniform noise, which means the
learned velocity field is not transporting them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


OBSERVED_GENERATED_MEDIAN_A = 0.33


def min_distance(points: list[tuple[float, float, float]], cell: float) -> float:
    """Minimum periodic distance in a cubic cell of edge ``cell`` Angstrom."""
    best = float("inf")
    n = len(points)
    for i in range(n):
        xi, yi, zi = points[i]
        for j in range(i + 1, n):
            xj, yj, zj = points[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            dx -= math.floor(dx + 0.5)
            dy -= math.floor(dy + 0.5)
            dz -= math.floor(dz + 0.5)
            squared = (dx * dx + dy * dy + dz * dz) * cell * cell
            if squared < best:
                best = squared
    return math.sqrt(best)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=91,
                        help="median primitive atom count from DATA_CARD_v1")
    parser.add_argument("--volume-per-atom", type=float, default=15.0,
                        help="Angstrom^3 per atom, typical for these phosphates")
    parser.add_argument("--sigma", type=float, default=0.50)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/h1f_standalone.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n = args.atoms
    volume = args.volume_per_atom * n
    cell = volume ** (1.0 / 3.0)

    uniform_mins: list[float] = []
    train_mins: list[float] = []
    target_mins: list[float] = []

    for _ in range(args.trials):
        # A crude stand-in for a real crystal: a jittered lattice, which unlike
        # uniform noise has a genuine minimum-separation scale.
        side = math.ceil(n ** (1.0 / 3.0))
        target = []
        for index in range(n):
            a = (index // (side * side)) % side
            b = (index // side) % side
            c = index % side
            target.append((
                (a + 0.5 + 0.06 * rng.gauss(0, 1)) / side,
                (b + 0.5 + 0.06 * rng.gauss(0, 1)) / side,
                (c + 0.5 + 0.06 * rng.gauss(0, 1)) / side,
            ))
        target_mins.append(min_distance(target, cell))

        train = [
            (
                (x + args.sigma * rng.gauss(0, 1)) % 1.0,
                (y + args.sigma * rng.gauss(0, 1)) % 1.0,
                (z + args.sigma * rng.gauss(0, 1)) % 1.0,
            )
            for x, y, z in target
        ]
        train_mins.append(min_distance(train, cell))

        uniform = [
            (rng.random(), rng.random(), rng.random()) for _ in range(n)
        ]
        uniform_mins.append(min_distance(uniform, cell))

    analytic = (3.0 * volume / (2.0 * math.pi * n * (n - 1))) ** (1.0 / 3.0)
    ripple = 2.0 * math.exp(-2.0 * math.pi ** 2 * (args.sigma * math.sqrt(2.0)) ** 2)

    train_median = median(train_mins)
    uniform_median = median(uniform_mins)
    target_median = median(target_mins)

    separated = train_median > 2.0 * uniform_median
    result = {
        "setup": {
            "atoms": n,
            "volume_A3": volume,
            "cell_edge_A": cell,
            "sigma": args.sigma,
            "trials": args.trials,
        },
        "median_min_distance_A": {
            "a_uniform_inference_source": uniform_median,
            "b_train_source_wrapped_normal": train_median,
            "c_target_jittered_lattice": target_median,
            "observed_generated": OBSERVED_GENERATED_MEDIAN_A,
        },
        "analytic": {
            "uniform_one_pair_radius_A": analytic,
            "wrapped_normal_uniformity_ripple": ripple,
            "note": (
                "ripple = 2*exp(-2*pi^2*(sigma*sqrt2)^2) bounds the pair-offset "
                "density's departure from uniform.  At sigma=0.5 it is ~1e-4, so "
                "sources (a) and (b) should be statistically indistinguishable "
                "for any pairwise statistic, including minimum distance."
            ),
        },
        "verdict": {
            "train_source_separated_from_uniform": bool(separated),
            "h1f_premise_holds": bool(separated),
            "reading": (
                "H1f requires the training source to be free of the close "
                "contacts the uniform source has.  If the two medians coincide, "
                "the model DID see sub-Angstrom contacts in training and "
                "swapping the inference source changes nothing."
                if not separated else
                "The training source is separated from uniform, so H1f's premise "
                "survives this synthetic check and should be confirmed against "
                "the real dataset with tools/p1_5_source_mismatch.py."
            ),
            "generated_consistent_with_uniform_noise": bool(
                abs(uniform_median - OBSERVED_GENERATED_MEDIAN_A) < 0.15
            ),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

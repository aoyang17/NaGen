"""P1.5 — train/inference source-distribution mismatch for fractional coordinates.

Hypothesis (H1f)
----------------
`model.flow_matching_loss` builds the X source as a target-dependent coupling::

    source_frac = wrap(target_frac + 0.50 * randn)        # model.py:216-221

but `sample.random_source` draws the X source i.i.d. uniform::

    torch.rand(batch, n_atoms, 3)                          # sample.py:57

A rectified-flow velocity field transports the *training* source marginal.  If
the inference source is a different distribution the transport guarantee is
void.  This script measures the gap directly, with no GPU and no model, by
comparing the minimum periodic interatomic distance of three point clouds:

    (c) target        - the packed quality dataset itself
    (b) train source  - wrap(target + sigma * randn)
    (a) infer source  - i.i.d. Uniform[0,1)^3

The observed generated median is 0.33 A (exp_2026-08-26 pilot_summary.json,
B0 `median_minimum_distance_A`).  If (a) reproduces that value while (b) and
(c) sit far above it, the model is passing close contacts straight through from
a source distribution it was never trained on.

Minimum-image convention matches the one the model itself uses
(`model.CrystalVectorField._radial_features`).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch


OBSERVED_GENERATED_MEDIAN_A = 0.33  # pilot_summary.json, B0 median_minimum_distance_A


def find_dataset(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    patterns = [
        "/mnt/data2/aobo/NaGen/experiments/**/*quality*.pt",
        "/mnt/data2/aobo/NaGen/**/*quality*.pt",
        "/mnt/data2/aobo/NaGen/**/naxl*.pt",
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError("could not locate a packed quality dataset; pass --dataset")


def min_pbc_distance(frac: np.ndarray, lattice: np.ndarray) -> float:
    """Smallest minimum-image interatomic distance in a single structure.

    This is the convention the *model* uses (`_radial_features`): the image is
    chosen by wrapping the fractional offset into [-0.5, 0.5)^3.
    """
    delta = frac[:, None, :] - frac[None, :, :]
    delta -= np.round(delta)
    cart = delta @ lattice
    distance = np.linalg.norm(cart, axis=-1)
    upper = np.triu_indices(len(frac), k=1)
    if len(upper[0]) == 0:
        return float("nan")
    return float(distance[upper].min())


def min_exact_distance(frac: np.ndarray, lattice: np.ndarray) -> float:
    """Same quantity under the 3x3x3 image search `constraints.pbc_distance_matrix` uses.

    For a skewed cell the nearest periodic image can lie outside [-0.5, 0.5)^3,
    so the minimum-image value is an over-estimate of the true minimum.  The gap
    between the two is exactly what the model cannot see.
    """
    delta = frac[:, None, :] - frac[None, :, :]
    delta -= np.floor(delta + 0.5)
    best = np.full(delta.shape[:2], np.inf)
    for i in (-1.0, 0.0, 1.0):
        for j in (-1.0, 0.0, 1.0):
            for k in (-1.0, 0.0, 1.0):
                cart = (delta + np.array((i, j, k))) @ lattice
                best = np.minimum(best, np.einsum("ijk,ijk->ij", cart, cart))
    distance = np.sqrt(np.maximum(best, 0.0))
    upper = np.triu_indices(len(frac), k=1)
    if len(upper[0]) == 0:
        return float("nan")
    return float(distance[upper].min())


def summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p05": float(np.quantile(finite, 0.05)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "frac_below_1.4A": float((finite < 1.4).mean()),
        "frac_below_2.0A": float((finite < 2.0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sigma", type=float, default=0.50,
                        help="coordinate_noise_sigma from model.flow_matching_loss")
    parser.add_argument("--limit", type=int, default=0, help="0 = all structures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/p1_5_source_mismatch.json")
    parser.add_argument("--figure", default="reports/p1_5_source_mismatch.png")
    args = parser.parse_args()

    path = find_dataset(args.dataset)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    offsets = payload["offsets"].numpy()
    frac_all = payload["frac_coords"].numpy().astype(float)
    lattices = payload["lattices"].numpy().astype(float)
    types_all = payload["types"].numpy()

    count = len(offsets) - 1
    if args.limit:
        count = min(count, args.limit)
    rng = np.random.default_rng(args.seed)

    target, train_source, infer_source, atom_counts = [], [], [], []
    exact_target, minimum_image_target = [], []
    for index in range(count):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        frac = frac_all[start:stop]
        lattice = lattices[index]
        if lattice.shape != (3, 3):
            lattice = lattice.reshape(3, 3)
        n = len(frac)
        atom_counts.append(n)

        wrapped = np.mod(frac, 1.0)
        target.append(min_pbc_distance(wrapped, lattice))
        noised = np.mod(frac + args.sigma * rng.standard_normal(frac.shape), 1.0)
        train_source.append(min_pbc_distance(noised, lattice))
        uniform = rng.random(frac.shape)
        infer_source.append(min_pbc_distance(uniform, lattice))

        # H1g: the model sees minimum-image; the constraint checker searches
        # 3x3x3 images.  Any positive gap is geometry the model cannot perceive.
        minimum_image_target.append(target[-1])
        exact_target.append(min_exact_distance(wrapped, lattice))

    minimum_image_target = np.array(minimum_image_target)
    exact_target = np.array(exact_target)
    convention_gap = minimum_image_target - exact_target

    result = {
        "dataset": str(path),
        "n_structures": count,
        "sigma": args.sigma,
        "n_atoms": {
            "median": float(np.median(atom_counts)),
            "min": int(np.min(atom_counts)),
            "max": int(np.max(atom_counts)),
        },
        "observed_generated_median_A": OBSERVED_GENERATED_MEDIAN_A,
        "min_pbc_distance_A": {
            "c_target": summarize(np.array(target)),
            "b_train_source": summarize(np.array(train_source)),
            "a_infer_source_uniform": summarize(np.array(infer_source)),
        },
        "h1g_image_convention": {
            "note": (
                "model._radial_features and flow_matching_loss.pair_distances use "
                "plain minimum-image; constraints.pbc_distance_matrix searches "
                "3x3x3 images.  min-image >= exact always, so the model "
                "over-estimates short contacts in skewed cells and is trained to "
                "tolerate violations the checker then rejects."
            ),
            "minimum_image": summarize(minimum_image_target),
            "exact_3x3x3": summarize(exact_target),
            "gap_A": {
                "median": float(np.median(convention_gap)),
                "max": float(convention_gap.max()),
                "frac_nonzero": float((convention_gap > 1.0e-6).mean()),
                "frac_above_0.1A": float((convention_gap > 0.1).mean()),
            },
        },
    }
    infer_median = result["min_pbc_distance_A"]["a_infer_source_uniform"]["median"]
    train_median = result["min_pbc_distance_A"]["b_train_source"]["median"]
    target_median = result["min_pbc_distance_A"]["c_target"]["median"]

    # H1f requires BOTH clauses.  The original verdict tested only the first and
    # would report success in the case that actually refutes the hypothesis.
    #
    # Analytically the second clause is expected to FAIL: for a pair (i, j) the
    # source offset is wrap(d_ij + sigma*(eps_i - eps_j)), whose per-coordinate
    # noise has scale sigma*sqrt(2) = 0.707.  A wrapped normal's departure from
    # uniform decays as exp(-2*pi^2*s^2) = 5e-5 at that scale, so the training
    # source's pair-distance law is uniform to within ~0.005% and (b) should
    # coincide with (a).  Then the model *did* see sub-Angstrom contacts in
    # training, and passing them through is not an unseen-input problem.
    infer_reproduces_generated = bool(
        abs(infer_median - OBSERVED_GENERATED_MEDIAN_A) < 0.25
    )
    train_source_is_safe = bool(train_median > 2.0 * infer_median)
    separation = float(train_median - infer_median)

    if infer_reproduces_generated and train_source_is_safe:
        verdict = "H1F_CONFIRMED"
        reading = (
            "The uniform inference source reproduces the generated median while "
            "the training source sits far above it.  The model never saw "
            "sub-Angstrom contacts and passes them through: the defect is at the "
            "inference entry point and is independent of capacity."
        )
    elif infer_reproduces_generated and not train_source_is_safe:
        verdict = "H1F_REFUTED_FLOW_IS_A_NO_OP"
        reading = (
            "The two candidate sources have the SAME close-contact statistics, so "
            "swapping them changes nothing and H1f is refuted.  But the generated "
            "median still equals the source median, which says the learned "
            "velocity field is not transporting the point cloud at all -- output "
            "approximately equals input.  That redirects the investigation to the "
            "transport itself (capacity, training budget, ODE steps), not to the "
            "three lines at the inference entry."
        )
    elif not infer_reproduces_generated:
        verdict = "H1F_REFUTED_SOURCE_DOES_NOT_EXPLAIN_GENERATED"
        reading = (
            "The uniform source does not reproduce the generated median, so the "
            "0.33 A contacts are produced by the flow rather than passed through "
            "from the source.  Neither source substitution addresses them."
        )
    else:  # pragma: no cover - kept explicit so no branch is silently absent
        verdict = "INCONCLUSIVE"
        reading = "Combination not anticipated; inspect the distributions directly."

    result["verdict"] = {
        "verdict": verdict,
        "reading": reading,
        "infer_source_reproduces_generated_median": infer_reproduces_generated,
        "train_source_is_contact_free_relative_to_infer": train_source_is_safe,
        "median_separation_train_minus_infer_A": separation,
        "medians_A": {
            "a_infer_source_uniform": infer_median,
            "b_train_source": train_median,
            "c_target": target_median,
            "observed_generated": OBSERVED_GENERATED_MEDIAN_A,
        },
        "criterion": (
            "H1f is confirmed only if the uniform source reproduces the generated "
            "median AND the training source is well separated from it.  If (a) and "
            "(b) coincide, the source substitution is a no-op and the hypothesis "
            "fails regardless of how well (a) matches the generated median."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = np.linspace(0.0, 4.0, 81)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for values, label, color in (
            (target, "(c) target — packed quality dataset", "#2e7d32"),
            (train_source, f"(b) train source — wrap(x+{args.sigma}·ε)", "#1565c0"),
            (infer_source, "(a) inference source — Uniform", "#c62828"),
        ):
            axis.hist(values, bins=bins, alpha=0.55, label=label, color=color)
        axis.axvline(OBSERVED_GENERATED_MEDIAN_A, color="black", linestyle="--",
                     label=f"observed generated median {OBSERVED_GENERATED_MEDIAN_A} Å")
        axis.set_xlabel("minimum periodic interatomic distance (Å)")
        axis.set_ylabel("structures")
        axis.set_title("P1.5 — X source distribution: training vs inference")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure_path = Path(args.figure)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=150)
        print(f"figure: {figure_path}")
    except Exception as error:  # figure is optional
        print(f"figure skipped: {error!r}")


if __name__ == "__main__":
    main()

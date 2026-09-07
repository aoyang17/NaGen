"""Assignment coupling between a uniform source and a crystal target.

Why this exists
---------------
`flow_matching_loss` currently builds the X source as a target-dependent
perturbation::

    source_frac = wrap(target_frac + 0.50 * randn)

The intent was to retain a site correspondence between source and target atoms.
At sigma = 0.50 it does not survive: for a pair (i, j) the source offset is
``wrap(d_ij + sigma * (eps_i - eps_j))``, whose per-coordinate noise has scale
``sigma * sqrt(2) = 0.707``.  A wrapped normal departs from uniform by
``2 * exp(-2 * pi^2 * s^2)``, which is ``1.0e-4`` at that scale.  So the pair
offset law is uniform to within 0.01%: the coupling carries essentially no
information, and the source is -- for every pairwise purpose -- already the
i.i.d. uniform cloud that inference draws.

Two consequences follow.  First, the train/inference source mismatch is not the
close-contact mechanism it was taken for, because both sources contain
sub-Angstrom contacts at the same rate.  Second, and more damaging, with a
near-independent coupling the regression target ``target - source`` has enormous
conditional variance, so the minimiser of the flow-matching objective collapses
toward its conditional mean, which on a torus is approximately zero.  The
velocity field then barely moves the cloud and the model returns its own input.

This module restores a genuinely informative coupling *without* perturbing the
source marginal.  The source is drawn exactly uniform -- identical to what
`sample.random_source` draws at inference, so the marginals match by
construction -- and is then permuted, within each element species, to minimise
total squared periodic displacement to the target.  Permuting a set of i.i.d.
uniform points leaves their joint law exactly uniform, so this is free: the
marginal is untouched while the per-atom displacement becomes short and
predictable.

This is the standard minibatch-OT / rectified-flow coupling argument, applied
per structure and per species because crystal atoms are a set, not a sequence,
and the index order in the packed dataset is arbitrary.

Distances use the exact 3x3x3 image search rather than the naive minimum image,
matching `constraints.pbc_distance_matrix`.  For a skewed cell the nearest
image can lie outside [-0.5, 0.5)^3, and the coupling should be built on the
metric the constraint checker actually enforces.
"""

from __future__ import annotations

import numpy as np
import torch


_IMAGES = np.array(
    [(i, j, k) for i in (-1.0, 0.0, 1.0) for j in (-1.0, 0.0, 1.0) for k in (-1.0, 0.0, 1.0)],
    dtype=np.float64,
)


def _exact_squared_cost(
    source: np.ndarray, target: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Squared periodic Cartesian distance between every source/target pair.

    ``source`` is (k, 3) fractional, ``target`` is (m, 3) fractional and
    ``matrix`` is the (3, 3) lattice mapping fractional to Cartesian.  Returns
    (k, m).  The minimum is taken over the 27 neighbouring images, so the result
    is the true periodic distance and not the minimum-image over-estimate.
    """
    delta = source[:, None, :] - target[None, :, :]
    delta -= np.floor(delta + 0.5)
    best = np.full(delta.shape[:2], np.inf)
    for image in _IMAGES:
        cart = (delta + image) @ matrix
        best = np.minimum(best, np.einsum("ijk,ijk->ij", cart, cart))
    return best


def assignment_source_frac(
    target_frac: torch.Tensor,
    target_types: torch.Tensor,
    mask: torch.Tensor,
    lattice_matrix: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw a uniform X source and match it to the target, species by species.

    Parameters
    ----------
    target_frac
        (B, N, 3) fractional coordinates of the target crystals.
    target_types
        (B, N) one-based element indices; 0 marks padding.
    mask
        (B, N) boolean, true for real atoms.
    lattice_matrix
        (B, 3, 3) mapping fractional offsets to Cartesian, as produced by
        ``CrystalVectorField._lattice_matrix``.
    generator
        Optional RNG for reproducibility.

    Returns
    -------
    torch.Tensor
        (B, N, 3) uniform source coordinates, permuted within each species so
        that atom ``n`` of the source is the matched partner of atom ``n`` of
        the target.  Padded slots carry unmatched uniform noise and are ignored
        downstream by ``mask``.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError(
            "assignment coupling needs scipy.optimize.linear_sum_assignment"
        ) from error

    source = torch.rand(
        target_frac.shape,
        dtype=target_frac.dtype,
        device=target_frac.device,
        generator=generator,
    )

    frac_np = source.detach().to(torch.float64).cpu().numpy()
    target_np = target_frac.detach().to(torch.float64).cpu().numpy()
    target_np = np.mod(target_np, 1.0)
    matrix_np = lattice_matrix.detach().to(torch.float64).cpu().numpy()
    types_np = target_types.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    matched = frac_np.copy()
    for b in range(frac_np.shape[0]):
        live = np.nonzero(mask_np[b])[0]
        if live.size == 0:
            continue
        for species in np.unique(types_np[b, live]):
            index = live[types_np[b, live] == species]
            if index.size < 2:
                continue  # a single atom is its own match
            cost = _exact_squared_cost(
                frac_np[b, index], target_np[b, index], matrix_np[b]
            )
            rows, columns = linear_sum_assignment(cost)
            # row = source slot, column = target slot: place the chosen source
            # point at the target atom it was matched to.
            matched[b, index[columns]] = frac_np[b, index[rows]]

    return torch.from_numpy(matched).to(
        dtype=target_frac.dtype, device=target_frac.device
    )


def coupling_diagnostics(
    source_frac: torch.Tensor,
    target_frac: torch.Tensor,
    mask: torch.Tensor,
    lattice_matrix: torch.Tensor,
) -> dict[str, float]:
    """Mean periodic displacement of a coupling, in Angstrom.

    A near-independent coupling on the unit torus has a displacement close to
    the cell-scale average; a useful one is markedly shorter.  This is the
    number that says whether the coupling is doing anything, so it is worth
    logging during training rather than inferring from the loss curve.
    """
    delta = target_frac - source_frac
    delta = delta - torch.floor(delta + 0.5)
    cart = torch.einsum("bnd,bdk->bnk", delta, lattice_matrix)
    distance = torch.linalg.vector_norm(cart, dim=-1)
    live = distance[mask]
    if live.numel() == 0:
        return {"mean_displacement_A": float("nan"), "median_displacement_A": float("nan")}
    return {
        "mean_displacement_A": float(live.mean()),
        "median_displacement_A": float(live.median()),
    }

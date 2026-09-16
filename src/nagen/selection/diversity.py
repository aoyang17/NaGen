"""Periodic smooth radial fingerprint for inexpensive diversity, not identity proof.

Invariance: site permutation, rigid motion, integer cell changes and supercells.
Radial fingerprints are not injective: distinct structures can share a fingerprint.
Keep thresholds small and validate finalists with StructureMatcher externally.
"""
from itertools import combinations_with_replacement
import numpy as np
from .pipeline import neighbors

DESCRIPTOR_VERSION = "periodic-pair-radial-v1-r5-dr0.1-width0.1"


def descriptor(crystal):
    pairs = list(combinations_with_replacement(sorted(("Na", "Fe", "P", "O")), 2))
    grid = np.arange(.1, 5.01, .1)
    values = np.zeros((len(pairs), len(grid)))
    index = {p: i for i, p in enumerate(pairs)}
    for i, e in enumerate(crystal.elements):
        for j, _, distance in neighbors(crystal, i, 5.):
            pair = tuple(sorted((e, crystal.elements[j])))
            taper = .5 * (1 + np.cos(np.pi * distance / 5.))
            values[index[pair]] += taper * np.exp(-.5 * ((grid-distance)/.1)**2)
    return (values / len(crystal.elements)).ravel()


def distances(a, b):
    a, b = np.asarray(a), np.asarray(b)
    # RMS of descriptor differences, in descriptor units, not Angstrom.
    from scipy.spatial.distance import cdist
    return cdist(a, b) / np.sqrt(a.shape[1])

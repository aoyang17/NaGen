from __future__ import annotations
from typing import TYPE_CHECKING

import copy
import numpy as np
from typing import List, TypeVar, Sequence, Dict, Tuple
from src.analysis_poly.coordination_polyhedron import CoordinationPolyhedron
from src.analysis_poly.atom import Atom

if TYPE_CHECKING:
    from src.analysis_poly.find_poly import PolyhedralCluster

T = TypeVar('T')

"""
Utility functions
"""


def flatten(this_list: Sequence[Sequence[T]]) -> List[T]:
    """Flattens a nested list.

    Args:
        (list): A list of lists.

    Returns:
        (list): The flattened list.

    """
    return [item for sublist in this_list for item in sublist]


def lattice_mc_string(polyhedron: CoordinationPolyhedron,
                      neighbour_list: Dict[int, Tuple[int, ...]]) -> str:
    """Returns a string representation of a polyhedron as a `lattice_mc`_
    site-input formatted site.

    .. _lattice_mc: https://github.com/bjmorgan/lattice_mc

    Args:
        polyhedron (CoordinationPolyhedron): The coordination polyhedron.
        neighbour_list (dict): Neighbour list dictionary.

    Returns:
        str

    Example:

        >>> nlist = {1: (3, 5, 8), ...}
        >>> lattice_mc_string(polyhedron, neighbour_list=nlist
        site: 1
        centre: 2.94651000 1.70116834 1.20290767
        neighbours: 3 5 8
        label: oct

    """
    string = f'site: {polyhedron.index}\n'
    string += f'centre: {" ".join(f"{c:.8f}" for c in polyhedron.central_atom.coords)}\n'
    string += f'neighbours: {" ".join(str(i) for i in neighbour_list[polyhedron.index])}\n'
    string += f'label: {polyhedron.label}\n'
    return string


def prune_neighbour_list(neighbours: Dict[int, Tuple[int, ...]], 
                         indices: List[int]) -> Dict[int, Tuple[int, ...]]:
    """TODO"""
    pruned_neighbours = {}
    for k, v in neighbours.items():
        if k in indices:
            pruned_neighbours[k] = tuple(set(v).intersection(set(indices)))
    return pruned_neighbours


class AtomWithCoords:
    def __init__(self, atom: Atom, coords: np.ndarray) -> None:
        self.atom = atom
        self.coords = coords

    def __eq__(self, other: AtomWithCoords) -> bool:
        return self.atom.index == other.atom.index and np.allclose(self.coords, other.coords)
    
    def __hash__(self):
        return hash((self.atom.index, tuple(np.round(self.coords, decimals = 8))))


def shared_atoms_with_coords(poly1: CoordinationPolyhedron | PolyhedralCluster,
                 poly2: CoordinationPolyhedron | PolyhedralCluster) -> list[Atom]:
    """Returns a list of shared atoms between two polyhedra.

    Args:
        poly1 (CoordinationPolyhedron): The first polyhedron.
        poly2 (CoordinationPolyhedron): The second polyhedron.

    Returns:
        list: A list of shared atoms.

    """
    def get_atom_set(poly):
        atom_set = set()
        if hasattr(poly, 'polyhedra'):
            for p in poly:
                for v in p.vertices:
                    atom_set.add(AtomWithCoords(v, v.site.coords))
        else:
            for v in poly.vertices:
                atom_set.add(AtomWithCoords(v, v.site.coords))
        return atom_set
    
    poly1_atoms = get_atom_set(poly1)
    poly2_atoms = get_atom_set(poly2)
    return [a.atom for a in list(poly1_atoms.intersection(poly2_atoms))]


def shared_atoms(poly1: CoordinationPolyhedron | PolyhedralCluster,
                 poly2: CoordinationPolyhedron | PolyhedralCluster) -> list[Atom]:
    """Returns a list of shared atoms between two polyhedra.

    Args:
        poly1 (CoordinationPolyhedron): The first polyhedron.
        poly2 (CoordinationPolyhedron): The second polyhedron.

    Returns:
        list: A list of shared atoms.

    """
    def get_atom_set(poly):
        atom_set = set()
        if hasattr(poly, 'polyhedra'):
            for p in poly:
                for v in p.vertices:
                    atom_set.add(v)
        else:
            for v in poly.vertices:
                atom_set.add(v)
        return atom_set
    
    poly1_atoms = get_atom_set(poly1)
    poly2_atoms = get_atom_set(poly2)
    return list(poly1_atoms.intersection(poly2_atoms))


def align(anchor: CoordinationPolyhedron, anchor_vertex: int, p: CoordinationPolyhedron, p_vertex: int) -> CoordinationPolyhedron:
    '''
    align polyhedron p to polyhedron anchor by translating p such that p_vertex overlaps with anchor_vertex
    '''
    lattice = anchor.central_atom.site.lattice

    anchor_coord = anchor.vertices[anchor_vertex].site.coords
    p_coord = p.vertices[p_vertex].site.coords
    translation = anchor_coord - p_coord

    p_aligned = copy.deepcopy(p)
    for v in p_aligned.vertices:
        v.site.coords += translation
        v.site.frac_coords = lattice.get_fractional_coords(v.site.coords)
    p_aligned.central_atom.site.coords += translation
    p_aligned.central_atom.site.frac_coords = lattice.get_fractional_coords(p_aligned.central_atom.site.coords)
    return p_aligned
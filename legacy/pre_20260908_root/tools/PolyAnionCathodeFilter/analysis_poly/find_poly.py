import numpy as np
from enum import Enum
import copy
from collections import defaultdict
from src.analysis_poly.configuration import Configuration
from src.analysis_poly.polyhedra_recipe import PolyhedraRecipe
from src.analysis_poly.coordination_polyhedron import CoordinationPolyhedron
from src.analysis_poly.atom import Atom
from src.analysis_poly.poly_utils import shared_atoms, shared_atoms_with_coords, align
from scipy.spatial import Delaunay, QhullError
from pymatgen.core import Structure


'''
和原polyhedral-analysis的区别：
1. 修复polyhedra_recipe.polyhedra_from_distance_cutoff：原函数未考虑晶胞周期性，不同晶胞中的同一个原子无法被重复识别为顶点，导致部分多面体配位数减少。
2. 修复coordination_polyhedron.CoordinationPolyhedron.convex_hull: 原函数构造convex hull时未包括中心原子，导致体积计算等功能有问题
3. 增加coordination_polyhedron.CoordinationPolyhedron.try_create: 原构造方法未判断中心加顶点是否能构成多面体，增加静态方法try_create，用于尝试创建多面体对象，如果所有原子共面则返回None
4. 修改atom.Atom：增加输出可读性

detect_polyhedra中增加过滤条件：
1. 多面体配位数至少为4
2. 顶点能构成convex hull，且中心原子必须在顶点构成的convex hull内
3. 多面体体积不少于设定阈值，过滤扁平多面体

PolyhedralCluster类：用于表示由多个CoordinationPolyhedron组成的复合多面体
'''

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial. Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        try:
            hull = Delaunay(hull)
        except QhullError as e:
            if "QH6154" in str(e) or "Initial simplex is flat" in str(e) \
                or "QH6214" in str(e) or "not enough points" in str(e):
                return False	# vertices cannot form a valid hull
            else:
                raise e

    return hull.find_simplex(p) >= 0


def detect_polyhedra(structure: Structure, recipes: list[PolyhedraRecipe], vol_th = 3.5) -> list[CoordinationPolyhedron]:
    config = Configuration(structure = structure, recipes = recipes)
    polyhedra = config.polyhedra

    filtered_polyhedra = []
    for poly in polyhedra:
        if poly.coordination_number < 4:
            continue
        if not in_hull(poly.central_atom.site.coords, [v.site.coords for v in poly.vertices]):
            continue
        if poly.volume < vol_th:
            continue
        filtered_polyhedra.append(poly)
    return filtered_polyhedra


class PolyhedralCluster:
    def __init__(self, polyhedra: list[CoordinationPolyhedron]) -> None:
        self.polyhedra = polyhedra
    
    def __getitem__(self, index) -> CoordinationPolyhedron:
        return self.polyhedra[index]
    
    def __len__(self) -> int:
        return len(self.polyhedra)


class PolyShareType(Enum):
    corner = "corner"
    edge = "edge"
    face = "face"


class Edge:
    poly_idx: int
    src_vertex_indices: list[int]
    nb_vertex_indices: list[int]
    share_type: PolyShareType

    def __init__(self, poly_idx: int, src_vertex_indices: list[int], nb_vertex_indices: list[int], share_type: PolyShareType) -> None:
        self.poly_idx = poly_idx
        self.src_vertex_indices = src_vertex_indices
        self.nb_vertex_indices = nb_vertex_indices
        self.share_type = share_type


class PolyhedralGraph:
    polyhedra: dict[int, CoordinationPolyhedron]
    edges: dict[int, list[Edge]]

    @property
    def poly_idxs(self) -> list[int]:
        return list(self.polyhedra.keys())

    def get_neighbors(self, poly: int | CoordinationPolyhedron, share_type: PolyShareType | None = None) -> list[Edge]:
        if hasattr(poly, 'central_atom'):
            poly_idx = poly.central_atom.index
        else:
            poly_idx = poly

        neighbors = []
        for edge in self.edges[poly_idx]:
            if share_type is not None and edge.share_type != share_type:
                continue

            neighbors.append(edge)
        return neighbors
    
    def is_connected(self, poly1: int | CoordinationPolyhedron, poly2: int | CoordinationPolyhedron, share_type: PolyShareType | None = None) -> bool:
        if hasattr(poly1, 'central_atom'):
            poly1_idx = poly1.central_atom.index
        else:
            poly1_idx = poly1

        if hasattr(poly2, 'central_atom'):
            poly2_idx = poly2.central_atom.index
        else:
            poly2_idx = poly2

        for edge in self.edges[poly1_idx]:
            if edge.poly_idx == poly2_idx:
                if share_type is None or edge.share_type == share_type:
                    return True
        return False
    
    def get_edge(self, poly1: int | CoordinationPolyhedron, poly2: int | CoordinationPolyhedron, share_type: PolyShareType | None = None) -> Edge | None:
        if hasattr(poly1, 'central_atom'):
            poly1_idx = poly1.central_atom.index
        else:
            poly1_idx = poly1

        if hasattr(poly2, 'central_atom'):
            poly2_idx = poly2.central_atom.index
        else:
            poly2_idx = poly2

        for edge in self.edges[poly1_idx]:
            if edge.poly_idx == poly2_idx:
                if share_type is None or edge.share_type == share_type:
                    return edge
        return None
    
    def subgraph(self, indices: list[int]) -> "PolyhedralGraph":
        subgraph = PolyhedralGraph()
        subgraph.polyhedra = {idx: self.polyhedra[idx] for idx in indices}
        subgraph.edges = {}
        for idx in indices:
            subgraph.edges[idx] = []
            for edge in self.edges[idx]:
                if edge.poly_idx in indices:
                    subgraph.edges[idx].append(edge)
        return subgraph
    
    @classmethod
    def from_polyhedra(cls, polyhedra: list[CoordinationPolyhedron]):
        graph = cls()
        graph.polyhedra = {poly.central_atom.index: poly for poly in polyhedra}
        graph.edges = {poly.central_atom.index: [] for poly in polyhedra}

        for i, poly1 in enumerate(polyhedra):
            for j in range(i, len(polyhedra)):
                poly2 = polyhedra[j]
                if i != j:
                    shared_atoms_list = [a.index for a in shared_atoms(poly1, poly2)]
                else:
                    unique, cnt = np.unique([v.index for v in poly1.vertices], return_counts = True)
                    shared_atoms_list = unique[cnt > 1]
                if len(shared_atoms_list) == 0:
                    continue

                connections = defaultdict(list) # map from aligned poly2 central coords to shared atom index pairs
                for shared_atom in shared_atoms_list:
                    for v1_idx, v1 in enumerate(poly1.vertices):
                        if v1.index != shared_atom:
                            continue
                        for v2_idx, v2 in enumerate(poly2.vertices):
                            if v2.index != shared_atom:
                                continue
                            if i == j and v2_idx <= v1_idx:
                                continue
                            aligned_coords = tuple(align(poly1, v1_idx, poly2, v2_idx).central_atom.site.coords.round(4))
                            connections[aligned_coords].append((v1_idx, v2_idx))
                for atom_pair_list in connections.values():
                    if len(atom_pair_list) == 1:
                        edge_1 = Edge(
                            poly_idx = poly2.central_atom.index,
                            src_vertex_indices = [atom_pair_list[0][0]],
                            nb_vertex_indices = [atom_pair_list[0][1]],
                            share_type = PolyShareType.corner
                        )
                        edge_2 = Edge(
                            poly_idx = poly1.central_atom.index,
                            src_vertex_indices = [atom_pair_list[0][1]],
                            nb_vertex_indices = [atom_pair_list[0][0]],
                            share_type = PolyShareType.corner
                        )
                    elif len(atom_pair_list) == 2:
                        edge_1 = Edge(
                            poly_idx = poly2.central_atom.index,
                            src_vertex_indices = [atom_pair[0] for atom_pair in atom_pair_list],
                            nb_vertex_indices = [atom_pair[1] for atom_pair in atom_pair_list],
                            share_type = PolyShareType.edge
                        )
                        edge_2 = Edge(
                            poly_idx = poly1.central_atom.index,
                            src_vertex_indices = [atom_pair[1] for atom_pair in atom_pair_list],
                            nb_vertex_indices = [atom_pair[0] for atom_pair in atom_pair_list],
                            share_type = PolyShareType.edge
                        )
                    else:
                        edge_1 = Edge(
                            poly_idx = poly2.central_atom.index,
                            src_vertex_indices = [atom_pair[0] for atom_pair in atom_pair_list],
                            nb_vertex_indices = [atom_pair[1] for atom_pair in atom_pair_list],
                            share_type = PolyShareType.face
                        )
                        edge_2 = Edge(
                            poly_idx = poly1.central_atom.index,
                            src_vertex_indices = [atom_pair[1] for atom_pair in atom_pair_list],
                            nb_vertex_indices = [atom_pair[0] for atom_pair in atom_pair_list],
                            share_type = PolyShareType.face
                        )
                    graph.edges[poly1.central_atom.index].append(edge_1)
                    graph.edges[poly2.central_atom.index].append(edge_2)
        return graph



if __name__ == "__main__":
    from pymatgen.core import Structure
    from src.analysis_poly.polyhedra_recipe import PolyhedraRecipe
    from collections import Counter

    structure = Structure.from_file("demo_cifs/Na4Fe3P4O15.cif")
    target_data = [structure]

    recipes = [
        PolyhedraRecipe(
            method = 'distance cutoff',
            coordination_cutoff = 2.9,
            central_atoms = 'Na',
            vertex_atoms = 'O'
        ),
        
        PolyhedraRecipe(
            method = 'distance cutoff',
            coordination_cutoff = 2.5,
            central_atoms = 'Fe',
            vertex_atoms = 'O'
        ),

        PolyhedraRecipe(
            method = 'distance cutoff',
            coordination_cutoff = 2.0,
            central_atoms = 'P',
            vertex_atoms = 'O'
        )
    ]


    for idx, structure in enumerate(target_data):
        polyhedra = detect_polyhedra(structure, recipes, vol_th = 0.5)
        print(f"Detected a total of {len(polyhedra)} polyhedra\n")

        # Statistics of different types of polyhedra
        coordination_numbers = Counter([len(poly.vertices) for poly in polyhedra])
        print("Coordination number statistics:")
        for cn, count in sorted(coordination_numbers.items()):
            print(f"  {cn}-coordinated: {count} polyhedra")

        # Detailed view of each polyhedron
        for i, poly in enumerate(polyhedra):
            print(f"\nPolyhedron {i+1}:")
            print(f"  Central atom: {poly.central_atom}")
            print(f"  Coordination number: {poly.coordination_number}")
            print(f"  Vertex elements: {poly.vertices}")
            print(f"  Volume: {poly.volume:.2f} ")
__all__ = []

# NOTE: analysis_poly package 的全局变量配置
__all__.extend(["TM_ATOMS_SET"])

TM_ATOMS_SET = {
    "All": {
        "Ni", "Mo", "Au", "Ru", "W", "Zn", "Ti", "Hf", "Ag", "Cu", "Zr", "Rh", "Fe", "Re", "Cr", 
        "Hg", "Sc", "Ir", "Mn", "Tc", "Y", "Os", "Cd", "Pd", "Co", "Pt", "Nb", "Ta", "V"
        },
    "NaCathode": {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Y", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "La", "Lu", "Ce"}
    }

# NOTE: General polyhedron analysis tools (from Xinyi) 
__all__.extend(["load_structures", "species_string"])
__all__.extend(["detect_polyhedra", "PolyhedralCluster", "PolyhedralGraph", "PolyShareType"])
__all__.extend(["shared_atoms", "shared_atoms_with_coords", "align"])
__all__.extend(["PolyhedraRecipe"])

from .utils import load_structures, species_string
from .find_poly import detect_polyhedra, PolyhedralCluster, PolyhedralGraph, PolyShareType
from .poly_utils import shared_atoms, shared_atoms_with_coords, align
from .polyhedra_recipe import PolyhedraRecipe


# NOTE: PolyhedronPropertyGraph (polyhedron property graph, 多面体属性图) 模块
__all__.extend(["PolyhedronPropertyGraph", "PolyhedronPropertyGraph_Extractor"])
__all__.extend([
    "PolyhedronPropertyGraph_TopoVisualizer", 
    "PolyhedronPropertyGraph_3DVisualizer",
    "PolyhedronPropertyGraph_to_Static3DImg"
    ])

from .polyhedron_property_graph.core import PolyhedronPropertyGraph, PolyhedronPropertyGraph_Extractor
from .polyhedron_property_graph.vision import PolyhedronPropertyGraph_TopoVisualizer
from .polyhedron_property_graph.vision import PolyhedronPropertyGraph_3DVisualizer
from .polyhedron_property_graph.vision import PolyhedronPropertyGraph_to_Static3DImg
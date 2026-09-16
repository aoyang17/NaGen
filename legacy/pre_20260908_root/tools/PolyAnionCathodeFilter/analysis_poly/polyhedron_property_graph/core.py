# NOTE: 项目的各种路径设定
import os, sys
current_path = os.path.abspath(os.path.dirname(__file__) if '__file__' in locals() else ".")
project_path = current_path[:current_path.find("ATLAS") + len("ATLAS")]
sys.path.append(project_path)

# NOTE: 导入基础运行环境
import numpy as np
import networkx as nx
from itertools import combinations
from typing import List, Optional

from dataclasses import dataclass, field

from pymatgen.core import Structure, Element

# pymatgen 导入路径更新，兼容新旧版本
try:
    from pymatgen.core.local_env import CrystalNN          # 新版 (≥2025)
except ImportError:
    from pymatgen.analysis.local_env import CrystalNN       # 旧版 fallback

# NOTE: 导入自定义运行环境
from src.analysis_poly import TM_ATOMS_SET

@dataclass
class PolyhedronPropertyGraph:
    """ 
    配位多面体属性图 (Polyhedron Property Graph) 数据类
    结合了图论拓扑结构 (NetworkX) 与三维几何特征 (坐标, 体积, 畸变); 用于晶体结构中局部配位环境的存储, 可视化与相似度计算
    """
    label: str
    center_element: str
    coord_number: int
    graph: nx.Graph = field(repr=False)
    volume: float = 0.0
    distortion_index: float = 0.0
    site_index: int = -1

    center_coords: Optional[np.ndarray] = field(default=None, repr=False)  # 中心原子笛卡尔坐标，shape (3,)
    ligand_coords: Optional[np.ndarray] = field(default=None, repr=False)  # 配体原子笛卡尔坐标，shape (x, 3)

    def __str__(self):
        return (f"{self.label} (site={self.site_index}, "
                f"vol={self.volume:.3f} Ang.^3, dist={self.distortion_index:.4f})")

    @property
    def canonical_coords(self) -> Optional[np.ndarray]:
        """
        获取标准化的配体坐标 (消除平移, 旋转干扰) 
        通过主成分分析 (PCA) 将多面体转换到标准坐标系中，确保两个形状相同的多面体在空间中的姿态完全一致，便于后续比较
        """
        if self.center_coords is None or self.ligand_coords is None:
            return None
        
        # 1. 消除平移：将中心原子移动到坐标原点 (0, 0, 0)
        centered = self.ligand_coords - self.center_coords
        if len(centered) < 2:
            return centered
        
        # 2. 消除旋转：计算协方差矩阵并进行特征值分解 (PCA)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 按特征值从大到小排序，找到多面体分布的三个主轴
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # 3. 消除镜像翻转：确保特征向量构成的是右手坐标系 (行列式 > 0)
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, -1] *= -1
        canonical = centered @ eigenvectors   # 将坐标投影到主轴上

        # 4. 消除方向二义性：统一各坐标轴的正方向（特征向量乘 -1 依然是特征向量）
        for axis in range(min(3, canonical.shape[1])):
            if np.sum(canonical[:, axis]) < 0:
                canonical[:, axis] *= -1  # 强制要求每个轴上所有坐标值的和为正数

        return canonical

    @staticmethod
    def rmsd(pg1: "PolyhedronPropertyGraph", pg2: "PolyhedronPropertyGraph") -> float:
        """
        计算两个多面体之间的均方根偏差 (RMSD)
        使用匈牙利算法 (Hungarian Algorithm) 自动寻找最优的配体匹配顺序, 从而消除由于配体原子索引顺序不同带来的误差
        """

        # 获取消除平移和旋转后的标准坐标
        c1 = pg1.canonical_coords
        c2 = pg2.canonical_coords

        if c1 is None or c2 is None:
            raise ValueError("缺少坐标信息")
        if c1.shape != c2.shape:
            raise ValueError(f"配位数不同: {c1.shape[0]} vs {c2.shape[0]}")
        
        from scipy.optimize import linear_sum_assignment

         # 构建代价矩阵 (Cost Matrix)：计算 c1 中的每个点到 c2 中每个点的距离平方
        cost = np.array([
            [np.sum((c1[i] - c2[j])**2) for j in range(len(c2))]
            for i in range(len(c1))
        ])

        row_ind, col_ind = linear_sum_assignment(cost)  # 最优分配：找到使总距离最小的一对一匹配关系
        c2_aligned = c2[col_ind]  # 根据匹配关系重排 c2 的配体顺序

        return float(np.sqrt(np.mean(np.sum((c1 - c2_aligned)**2, axis=1))))  # 计算并返回标准的 RMSD 值

# ══════════════════════════════════════════════════════
#  Part 3: 增强版 Extractor（★ 关键修复在这里 ★）
# ══════════════════════════════════════════════════════
class PolyhedronPropertyGraph_Extractor:
    """ 在构建图时同时保存坐标, 用以支持 3D 可视化 """

    def __init__(self, target_x=None, ligand="O", tm_symbols=None,
                 nn_algo=None, add_oo_edges=True):
        self.target_x = target_x
        self.ligand = ligand
        self.tm_symbols = tm_symbols or TM_ATOMS_SET["All"]
        # ← FIX 2: 默认关闭 cation_anion，避免未赋氧化态时静默失败
        self.nn_algo = nn_algo or CrystalNN(weighted_cn=False, cation_anion=False)
        self.add_oo_edges = add_oo_edges

    def extract(self, structure: Structure) -> List[PolyhedronPropertyGraph]:
        results = []
        for i, site in enumerate(structure):
            if site.specie.symbol not in self.tm_symbols:
                continue
            neighbors = self._get_ligand_neighbors(structure, i)
            if neighbors is None:
                continue
            pg = self._build_graph(structure, i, neighbors)
            results.append(pg)
        return results

    def extract_unique(self, polyhedra, length_tol=0.05, angle_tol=2.0):
        unique = []
        for pg in polyhedra:
            if not any(self._is_equivalent(pg, u, length_tol, angle_tol) for u in unique):
                unique.append(pg)
        return unique

    # def _get_ligand_neighbors(self, structure, center_idx):
    #     try:
    #         nn_info = self.nn_algo.get_nn_info(structure, center_idx)
    #     except Exception:
    #         return None
    #     ligands = [i for i in nn_info if i["site"].specie.symbol == self.ligand]
    #     if self.target_x is not None and len(ligands) != self.target_x:
    #         return None
    #     return ligands or None

    def _get_ligand_neighbors(self, structure, center_idx):
        try:
            # 获取第一配位层的所有近邻原子
            nn_info = self.nn_algo.get_nn_info(structure, center_idx)
        except Exception:
            return None

        target_ligands = []
        other_atoms = []

        # 遍历第一配位层，对原子进行分类
        for info in nn_info:
            if info["site"].specie.symbol == self.ligand:
                target_ligands.append(info)
            else:
                other_atoms.append(info["site"].specie.symbol)

        # 🌟 核心修复：如果第一配位层中出现了非目标原子（如 F），则判定为混合配位，直接舍弃
        if len(other_atoms) > 0:
            # 提示：如果未来需要专门研究混合配位，可以在这里将 other_atoms 的信息 return 出去
            return None

        # 检查纯配位多面体的配位数是否满足 target_x
        if self.target_x is not None and len(target_ligands) != self.target_x:
            return None

        return target_ligands or None

    def _build_graph(self, structure, center_idx, neighbors):
        G = nx.Graph()
        center_site = structure[center_idx]
        center_elem = center_site.specie.symbol
        x = len(neighbors)

        G.add_node("TM", element=center_elem, role="center", coord_number=x)

        bond_lengths, o_coords = [], []
        for j, info in enumerate(neighbors):
            # ══════════════════════════════════════════
            # ← FIX 3 (核心修复): 手动计算距离
            # CrystalNN.get_nn_info() 返回的 dict 没有 "distance" 键
            # 只有 "site", "image", "weight", "site_index"
            # 距离需要从中心 site 到近邻 site 计算
            # ══════════════════════════════════════════
            neighbor_site = info["site"]
            dist = float(np.linalg.norm(neighbor_site.coords - center_site.coords))

            bond_lengths.append(dist)
            o_cart = neighbor_site.coords
            o_coords.append(o_cart)
            G.add_node(f"O_{j}", element=self.ligand, role="ligand")
            G.add_edge("TM", f"O_{j}", bond_length=round(dist, 5), edge_type="TM-O")

        if self.add_oo_edges:
            tm_cart = center_site.coords
            for (j1, _), (j2, _) in combinations(enumerate(neighbors), 2):
                v1, v2 = o_coords[j1] - tm_cart, o_coords[j2] - tm_cart
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                oo_dist = float(np.linalg.norm(o_coords[j1] - o_coords[j2]))
                G.add_edge(f"O_{j1}", f"O_{j2}",
                           distance=round(oo_dist, 5),
                           angle_at_TM=round(angle, 3),
                           edge_type="O-O")

        bl = np.array(bond_lengths)
        mean_bl = bl.mean()
        distortion = float(np.mean(np.abs(bl - mean_bl) / mean_bl))

        try:
            from scipy.spatial import ConvexHull
            volume = float(ConvexHull(np.array(o_coords)).volume) if x >= 4 else 0.0
        except Exception:
            volume = 0.0

        return PolyhedronPropertyGraph(
            label=f"{center_elem}{self.ligand}{x}",
            center_element=center_elem,
            coord_number=x,
            graph=G,
            volume=round(volume, 4),
            distortion_index=round(distortion, 6),
            site_index=center_idx,
            center_coords=np.array(center_site.coords),
            ligand_coords=np.array(o_coords),
        )

    def _is_equivalent(self, pg1, pg2, length_tol, angle_tol):
        if pg1.label != pg2.label:
            return False
        def node_match(n1, n2):
            return n1.get("element") == n2.get("element")
        def edge_match(e1, e2):
            if e1.get("edge_type") != e2.get("edge_type"):
                return False
            if e1["edge_type"] == "TM-O":
                return abs(e1["bond_length"] - e2["bond_length"]) < length_tol
            return abs(e1["angle_at_TM"] - e2["angle_at_TM"]) < angle_tol
        return nx.is_isomorphic(pg1.graph, pg2.graph,
                                node_match=node_match, edge_match=edge_match)

if __name__ == "__main__":
    """ 使用示例 (Waiting for reparation) """

    import os, sys
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    project_dir = Path(str(current_dir)[:str(current_dir).find("ATLAS") + len("ATLAS")])
    sys.path.append(str(project_dir))

    # ── 0. 参数设定 ──
    target_x = 5
    
    # cif_dir = "/home/majestyv/struct_gallery/NaCathode/demo"          # Lingjiang
    cif_dir = "/home/LiuSW/work_space/struct_gallery/NaCathode/demo"  # HAVANA
    
    # struct_id = "mp-25288_V2O5"
    # struct_id = "mp-19346_Fe3P2O8"
    struct_id = "mp-19366_LuFe2O4"
    cif_file = f"{cif_dir}/{struct_id}.cif"
    output_dir = f"{project_dir}/results/NaCathode/work_space/{struct_id}"
    os.makedirs(output_dir, exist_ok=True)  # 保证输出目录存在

    # ── 1. 加载结构 ──
    structure = Structure.from_file(cif_file)

    # ── 2. 提取 TMO6 多面体 ──
    extractor = PolyhedronPropertyGraph_Extractor(target_x=target_x)
    polyhedra = extractor.extract(structure)

    # ← FIX 4: 防御性检查，避免空列表导致 IndexError
    if len(polyhedra) == 0:
        print(f"❌ 未找到任何 TMO{target_x} 多面体，请检查：")
        print(f"   1. 结构中的元素: {set(s.specie.symbol for s in structure)}")
        print(f"   2. 尝试 target_x=None 查看所有配位数")
        # 诊断：打印所有 TM 的实际配位数
        diag_ext = PolyhedronPropertyGraph_Extractor(target_x=None)
        diag_pgs = diag_ext.extract(structure)
        if diag_pgs:
            print(f"   3. 实际找到的配位数: {[(pg.label, pg.coord_number) for pg in diag_pgs]}")
        sys.exit(1)

    unique = extractor.extract_unique(polyhedra)
    print(f"找到 {len(polyhedra)} 个 TMO{target_x}, 去重后 {len(unique)} 种\n")

    # ── 3. 图拓扑可视化（2D）──
    PolyhedronPropertyGraph_TopoVisualizer.plot_single(unique[0])
    plt.savefig(f"{output_dir}/topo_single.png", dpi=150, bbox_inches="tight")
    plt.show()

    if len(unique) > 1:
        fig = PolyhedronPropertyGraph_TopoVisualizer.plot_compare(
            unique, ncols=4, suptitle=f"Unique TMO{target_x} Polyhedra")
        fig.savefig(f"{output_dir}/topo_compare.png", dpi=150, bbox_inches="tight")
        plt.show()

    # # ── 4. 3D 多面体可视化（交互式）──
    # fig3d = PolyhedronPropertyGraph_3DVisualizer.plot_single(unique[0])
    # PolyhedronPropertyGraph_3DVisualizer.save_html(fig3d, f"{output_dir}/polyhedron_3d.html")

    # if len(unique) > 1:
    #     fig3d_cmp = PolyhedronPropertyGraph_3DVisualizer.plot_compare(unique, ncols=3)
    #     PolyhedronPropertyGraph_3DVisualizer.save_html(fig3d_cmp, f"{output_dir}/polyhedra_compare_3d.html")

    # ── 5. 3D 多面体可视化（静态 PNG 导出）──
    print("\n正在生成 3D 多面体渲染图 (PNG)...")
    
    # 渲染单个多面体 (务必关闭 auto_show)
    fig3d = PolyhedronPropertyGraph_3DVisualizer.plot_single(unique[0], auto_show=False)
    PolyhedronPropertyGraph_3DVisualizer.save_png(fig3d, f"{output_dir}/polyhedron_3d.png")

    # 如果有多个多面体，渲染对比图
    if len(unique) > 1:
        # 适当加宽画布以容纳多个多面体
        fig3d_cmp = PolyhedronPropertyGraph_3DVisualizer.plot_compare(unique, ncols=3, auto_show=False)
        PolyhedronPropertyGraph_3DVisualizer.save_png(
            fig3d_cmp, f"{output_dir}/polyhedra_compare_3d.png", width=1200, height=600
            )

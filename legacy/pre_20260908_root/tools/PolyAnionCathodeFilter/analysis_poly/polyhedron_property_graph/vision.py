# NOTE: 项目的各种路径设定
import os, sys
current_path = os.path.abspath(os.path.dirname(__file__) if '__file__' in locals() else ".")
project_path = current_path[:current_path.find("ATLAS") + len("ATLAS")]
sys.path.append(project_path)

# NOTE: 导入基础运行环境
import numpy as np
import networkx as nx
from itertools import combinations
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from typing import List, Tuple, Dict, Union  # 用于变量类型的描述

# NOTE: 导入自定义运行环境
from src.analysis_poly import PolyhedronPropertyGraph

# NOTE: 自动识别 plotly (用于 3D 可视化)
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

class PolyhedronPropertyGraph_TopoVisualizer:
    """ 此函数类专用于图拓扑可视化 """
    ELEMENT_COLORS: Dict[str, str] = {
        "Fe": "#E07020", "Mn": "#9C27B0", "Co": "#0066CC",
        "Ni": "#50C878", "V":  "#FF4444", "Ti": "#808080",
        "Cr": "#2196F3", "Cu": "#C77532", "Zn": "#7D8590",
        "Mo": "#54B5A6", "W":  "#2C6E49", "O":  "#FF0000",
    }
    DEFAULT_COLOR = "#999999"

    @classmethod
    def plot_single(cls, pg, ax=None, show_edge_labels=True, title=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        G = pg.graph
        pos = cls._centered_layout(G)
        tmo_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "TM-O"]
        oo_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "O-O"]
        nx.draw_networkx_edges(G, pos, edgelist=tmo_edges, ax=ax, width=2.5, edge_color="#333333", style="solid")
        nx.draw_networkx_edges(G, pos, edgelist=oo_edges, ax=ax, width=1.0, edge_color="#AAAAAA", style="dashed")
        tm_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "center"]
        o_nodes  = [n for n, d in G.nodes(data=True) if d.get("role") == "ligand"]
        tm_color = cls.ELEMENT_COLORS.get(pg.center_element, cls.DEFAULT_COLOR)
        nx.draw_networkx_nodes(G, pos, nodelist=tm_nodes, ax=ax, node_size=800, node_color=tm_color, edgecolors="black", linewidths=2)
        nx.draw_networkx_nodes(G, pos, nodelist=o_nodes, ax=ax, node_size=350, node_color="#FF6666", edgecolors="black", linewidths=1)
        node_labels = {}
        for n, d in G.nodes(data=True):
            node_labels[n] = d["element"] if d.get("role") == "center" else "O"
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=10, font_weight="bold", font_color="white")
        if show_edge_labels:
            tmo_labels = {(u, v): f'{d["bond_length"]:.3f} Å' for u, v, d in G.edges(data=True) if d.get("edge_type") == "TM-O"}
            oo_labels = {(u, v): f'{d["angle_at_TM"]:.1f}°' for u, v, d in G.edges(data=True) if d.get("edge_type") == "O-O"}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=tmo_labels, ax=ax, font_size=7, font_color="#333333",
                                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
            nx.draw_networkx_edge_labels(G, pos, edge_labels=oo_labels, ax=ax, font_size=6, font_color="#888888",
                                         bbox=dict(boxstyle="round,pad=0.1", fc="#F5F5F5", ec="none", alpha=0.7))
        title = title or f"{pg.label}  |  V={pg.volume:.2f} ų  |  Δ={pg.distortion_index:.4f}"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        legend_elements = [
            Line2D([0], [0], color="#333333", lw=2.5, label="TM—O bond"),
            Line2D([0], [0], color="#AAAAAA", lw=1.0, ls="--", label="O···O (angle)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.8, edgecolor="none")
        ax.set_axis_off()
        return ax

    @classmethod
    def plot_compare(cls, polyhedra, ncols=4, figsize_per=(4.5, 4.5), suptitle=""):
        n = len(polyhedra)
        ncols = min(ncols, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
        axes = np.atleast_2d(axes)
        for idx, pg in enumerate(polyhedra):
            r, c = divmod(idx, ncols)
            cls.plot_single(pg, ax=axes[r, c])
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig

    @staticmethod
    def _centered_layout(G):
        pos = {}
        o_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "ligand"]
        tm_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "center"]
        for tm in tm_nodes:
            pos[tm] = np.array([0.0, 0.0])
        n_o = len(o_nodes)
        for i, o in enumerate(o_nodes):
            theta = 2 * np.pi * i / n_o - np.pi / 2
            pos[o] = np.array([np.cos(theta), np.sin(theta)])
        return pos

class PolyhedronPropertyGraph_3DVisualizer:
    """ 此函数类专用于 3D 多面体结构可视化 """
    ELEMENT_COLORS_3D: Dict[str, str] = {
        "Fe": "#E07020", "Mn": "#9C27B0", "Co": "#0066CC",
        "Ni": "#50C878", "V":  "#FF4444", "Ti": "#808080",
        "Cr": "#2196F3", "Cu": "#C77532", "Zn": "#7D8590",
        "O":  "#FF0000",
    }

    @classmethod
    def plot_single(cls, pg, show_faces=True, face_opacity=0.25, title=None, auto_show=True):
        if not HAS_PLOTLY:
            raise ImportError("请安装 plotly: pip install plotly")
        if pg.center_coords is None or pg.ligand_coords is None:
            raise ValueError(f"PolyhedronPropertyGraph '{pg.label}' 缺少坐标信息。")
        fig = go.Figure()
        center = pg.center_coords
        ligands = pg.ligand_coords
        tm_color = cls.ELEMENT_COLORS_3D.get(pg.center_element, "#999999")
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]], mode="markers+text",
            marker=dict(size=10, color=tm_color, line=dict(width=1, color="black")),
            text=[pg.center_element], textposition="top center", textfont=dict(size=12, color="black"),
            name=pg.center_element, hovertext=f"{pg.center_element} (center)<br>CN={pg.coord_number}", hoverinfo="text",
        ))
        fig.add_trace(go.Scatter3d(
            x=ligands[:, 0], y=ligands[:, 1], z=ligands[:, 2], mode="markers+text",
            marker=dict(size=6, color="#FF4444", line=dict(width=0.5, color="black")),
            text=["O"] * len(ligands), textposition="top center", textfont=dict(size=10, color="#CC0000"),
            name="O", hovertext=[f"O_{i}<br>d(TM-O)={np.linalg.norm(ligands[i]-center):.4f} Å" for i in range(len(ligands))],
            hoverinfo="text",
        ))
        for i, lig in enumerate(ligands):
            dist = np.linalg.norm(lig - center)
            fig.add_trace(go.Scatter3d(
                x=[center[0], lig[0]], y=[center[1], lig[1]], z=[center[2], lig[2]],
                mode="lines", line=dict(width=5, color="#555555"), showlegend=False,
                hovertext=f"TM—O_{i}: {dist:.4f} Å", hoverinfo="text",
            ))
        for i, j in combinations(range(len(ligands)), 2):
            oo_dist = np.linalg.norm(ligands[i] - ligands[j])
            max_bl = max(np.linalg.norm(l - center) for l in ligands)
            if oo_dist < max_bl * 1.8:
                fig.add_trace(go.Scatter3d(
                    x=[ligands[i][0], ligands[j][0]], y=[ligands[i][1], ligands[j][1]], z=[ligands[i][2], ligands[j][2]],
                    mode="lines", line=dict(width=2, color="#CCCCCC", dash="dash"), showlegend=False, hoverinfo="skip",
                ))
        if show_faces and len(ligands) >= 4:
            fig = cls._add_convex_hull_faces(fig, ligands, tm_color, face_opacity)
        elif show_faces and len(ligands) == 3:
            fig.add_trace(go.Mesh3d(x=ligands[:, 0], y=ligands[:, 1], z=ligands[:, 2],
                                     i=[0], j=[1], k=[2], color=tm_color, opacity=face_opacity, showlegend=False, hoverinfo="skip"))
        title = title or f"{pg.label}  |  V={pg.volume:.2f} ų  |  Δ={pg.distortion_index:.4f}"
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)", aspectmode="data",
                       camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))),
            width=650, height=550, margin=dict(l=10, r=10, t=50, b=10), showlegend=True, legend=dict(x=0.85, y=0.95),
        )
        if auto_show:
            fig.show()
        return fig

    @classmethod
    def plot_compare(cls, polyhedra, ncols=3, show_faces=True, face_opacity=0.2, auto_show=True):
        if not HAS_PLOTLY:
            raise ImportError("请安装 plotly: pip install plotly")
        fig = go.Figure()
        spacing = 5.0
        annotations = []
        for idx, pg in enumerate(polyhedra):
            if pg.center_coords is None or pg.ligand_coords is None:
                continue
            col = idx % ncols
            row = idx // ncols
            offset = np.array([col * spacing, -row * spacing, 0.0])
            center = pg.center_coords + offset
            ligands = pg.ligand_coords + offset
            tm_color = cls.ELEMENT_COLORS_3D.get(pg.center_element, "#999999")
            fig.add_trace(go.Scatter3d(
                x=[center[0]], y=[center[1]], z=[center[2]], mode="markers",
                marker=dict(size=8, color=tm_color, line=dict(width=1, color="black")),
                name=pg.label if idx == 0 or pg.label != polyhedra[idx-1].label else None,
                showlegend=(idx == 0), hovertext=f"{pg.label}<br>site={pg.site_index}", hoverinfo="text",
            ))
            fig.add_trace(go.Scatter3d(
                x=ligands[:, 0], y=ligands[:, 1], z=ligands[:, 2], mode="markers",
                marker=dict(size=4, color="#FF4444"), showlegend=False, hoverinfo="skip",
            ))
            for lig in ligands:
                fig.add_trace(go.Scatter3d(
                    x=[center[0], lig[0]], y=[center[1], lig[1]], z=[center[2], lig[2]],
                    mode="lines", line=dict(width=4, color="#555555"), showlegend=False, hoverinfo="skip",
                ))
            if show_faces and len(ligands) >= 4:
                fig = cls._add_convex_hull_faces(fig, ligands, tm_color, face_opacity)
            annotations.append(dict(x=center[0], y=center[1] - 1.5, z=center[2],
                                     text=f"{pg.label}<br>Δ={pg.distortion_index:.4f}", showarrow=False, font=dict(size=10)))
        fig.update_layout(
            title="TMOx Polyhedra Comparison",
            scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)", aspectmode="data", annotations=annotations),
            width=350 * min(ncols, len(polyhedra)), height=500, margin=dict(l=5, r=5, t=40, b=5),
        )
        if auto_show:
            fig.show()
        return fig

    @classmethod
    def save_html(cls, fig, path):
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"✅ 已保存至 {path}")

    @classmethod
    def save_png(cls, fig, path, width=800, height=600, scale=2):
        """
        新增：将 Plotly 3D 图像直接保存为高分辨率 PNG
        scale=2 表示输出 2 倍抗锯齿的高清图，适合论文发表
        """
        try:
            # write_image 依赖 kaleido 库
            fig.write_image(path, width=width, height=height, scale=scale)
            print(f"✅ 3D 可视化已成功保存为 PNG: {path}")
        except ValueError as e:
            print(f"❌ 保存失败。请确保已在服务器安装 kaleido: pip install kaleido\n错误信息: {e}")

    @staticmethod
    def _add_convex_hull_faces(fig, vertices, color, opacity):
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(vertices)
            simplices = hull.simplices
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=simplices[:, 0], j=simplices[:, 1], k=simplices[:, 2],
                color=color, opacity=opacity, showlegend=False, hoverinfo="skip", flatshading=True,
            ))
        except Exception:
            pass
        return fig

def PolyhedronPropertyGraph_to_Static3DImg(
        polyhedron_property_graphs: List[PolyhedronPropertyGraph], output_path: str, dpi: int=300, show_fig: bool=False,
        color_TM: Union[str,Tuple]="#555555", color_O: Union[str,Tuple]="#FF4444", 
        color_bond: Union[str,Tuple]="#333333", color_facet: Union[str,Tuple]="#2196F3"
        ) -> None:
    """ 生成静态的三维多面体对比图 (包含 3D + 三视图), 专用于无可视化界面的服务器集群; 每种多面体占据一行，包含 4 个子图 """
    
    ngraphs = len(polyhedron_property_graphs)
    ncols = 4
    nrows = ngraphs

    fig = plt.figure(figsize=(14, 3.5 * nrows))  # 每行高度 3.5，每列宽度 3.5

    # 定义 4 种视角的参数
    view_configs = [
        {"title": "3D View", "elev": 20, "azim": 45},
        {"title": "Top View (XY perspective)", "elev": 90, "azim": -90},
        {"title": "Front View (XZ perspective)", "elev": 0, "azim": -90},
        {"title": "Left View (YZ perspective)", "elev": 0, "azim": 0},
    ]

    for i, pg in enumerate(polyhedron_property_graphs):
        # 使用 PCA 规范化坐标摆正多面体
        if hasattr(pg, "canonical_coords") and pg.canonical_coords is not None:
            ligands = pg.canonical_coords
        else:
            ligands = pg.ligand_coords - pg.center_coords
            
        center = np.array([0.0, 0.0, 0.0])

        for j, config in enumerate(view_configs):
            ax = fig.add_subplot(nrows, ncols, i * ncols + j + 1, projection="3d")
            
            # 强制使用正交投影，消除透视畸变
            ax.set_proj_type('ortho')

            # 绘制中心 TM 原子 (深灰色)
            ax.scatter(*center, color=color_TM, s=120, edgecolors="black", zorder=5)

            # 绘制 O 配体原子 (红色)
            ax.scatter(ligands[:, 0], ligands[:, 1], ligands[:, 2], color=color_O, s=60, edgecolors="black", zorder=5)

            # 绘制 TM-O 键
            for lig in ligands:
                ax.plot([center[0], lig[0]], [center[1], lig[1]], [center[2], lig[2]], color=color_bond, lw=2.5, zorder=1)

            # 绘制多面体的面 (凸包)
            if len(ligands) >= 4:
                try:
                    hull = ConvexHull(ligands)
                    faces = [ligands[simplex] for simplex in hull.simplices]
                    poly3d = Poly3DCollection(faces, alpha=0.2, facecolor=color_facet, edgecolors="#999999", linewidths=1.0)
                    ax.add_collection3d(poly3d)
                except Exception:
                    pass

            # 设置视角
            ax.view_init(elev=config["elev"], azim=config["azim"])
            try:
                ax.set_box_aspect([1, 1, 1]) 
            except AttributeError:
                pass 
            
            ax.axis('off')
            
            # 设置标题
            if j == 0:
                ax.set_title(f"Type {i+1} | V={pg.volume:.2f} | Δ={pg.distortion_index:.4f}\n{config['title']}", fontsize=11, pad=10, fontweight='bold')
            else:
                ax.set_title(config['title'], fontsize=11, pad=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show() if show_fig else None  # 可视化图片，用于 Jupyter Notebook 等自带可视化界面的场景
    plt.close(fig)

    return
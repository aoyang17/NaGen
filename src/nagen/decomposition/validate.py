"""Step 5: 反衍生物重建验证与验收门。"""

from __future__ import annotations

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

MATCH_TOL = 0.9  # Å，同格胞位点匹配容差


def rebuild_structure(
    framework_sites: list[dict],
    na_sites: np.ndarray,
    occupancy: list[int],
    lattice: np.ndarray,
    shift: np.ndarray | None = None,
) -> Structure:
    """由母体位点 + 占据重建成员结构（使用成员晶格，可选原点平移）。"""
    shift = np.zeros(3) if shift is None else np.asarray(shift) % 1.0
    species, coords = [], []
    for s in framework_sites:
        species.append(s["el"])
        coords.append((np.array(s["frac"]) + shift) % 1.0)
    for i, occ in enumerate(occupancy):
        if occ:
            species.append("Na")
            coords.append((np.array(na_sites[i]) + shift) % 1.0)
    return Structure(lattice, species, coords)


def site_rmsd(st_a: Structure, st_b: Structure) -> dict:
    """同格胞两结构按元素/组最近镜像匹配后的 RMSD（Å）。

    TM（Ti/V/Fe/Mn）作为装饰槽位，组内任意互换仍视为同一几何位点。
    """
    TM = ("Ti", "V", "Fe", "Mn")

    def match_group(species):
        if species in TM:
            return "TM"
        return species

    diff = []
    for s in st_a:
        g = match_group(s.species_string)
        cand = [t for t in st_b if match_group(t.species_string) == g]
        if not cand:
            continue
        d = np.array([t.frac_coords for t in cand]) - s.frac_coords
        d -= np.round(d)
        dist = np.linalg.norm(d @ st_a.lattice.matrix, axis=1)
        diff.append(dist.min())
    diff = np.array(diff)
    if len(diff) == 0:
        return {"rmsd": None, "n": 0, "p90": None}
    return {"rmsd": float(np.sqrt(np.mean(diff**2))), "n": int(len(diff)), "p90": float(np.percentile(diff, 90))}


def matcher_pass(st_a: Structure, st_b: Structure, stol: float = 0.5) -> bool:
    """StructureMatcher 通过判定（TM 装饰无关：Ti/V/Fe/Mn 统一为 X）。"""
    m = StructureMatcher(ltol=0.3, stol=stol, angle_tol=5, primitive_cell=False)
    TM = ("Ti", "V", "Fe", "Mn")

    def anonymize(st):
        return Structure(
            st.lattice,
            ["X" if s.species_string in TM else s.species_string for s in st],
            [s.frac_coords for s in st],
        )

    try:
        return m.fit(anonymize(st_a), anonymize(st_b))
    except Exception:
        return False


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        order = a.argsort()
        ranks = np.empty(len(a))
        ranks[order] = np.arange(1, len(a) + 1)
        return ranks

    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    d = rx - ry
    return float(1 - 6 * np.sum(d**2) / (len(x) * (len(x) ** 2 - 1)))

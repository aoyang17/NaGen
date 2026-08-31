"""Step 2: 家族母体搜索（数据驱动：最小体积 + 最高对称性成员为候选）。

输出 parents_v2.json：理想化母体（框架均值位点 + Na 位点并集 + 对称性）。
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from nagen.decomposition import na_sublattice

FRAMEWORK_SPECIES = ("P", "Ti", "V", "Fe", "Mn", "O")
SUPERLATTICE_TOL = 0.08


def framework_only(st: Structure) -> Structure:
    sites = [(s.species_string, s.frac_coords) for s in st if s.species_string in FRAMEWORK_SPECIES]
    return Structure(st.lattice, [s[0] for s in sites], [s[1] for s in sites])


def is_integer_superlattice(Lp: np.ndarray, Lm: np.ndarray, tol: float = SUPERLATTICE_TOL):
    M = np.linalg.inv(Lp) @ Lm
    r = np.abs(M - np.round(M))
    return bool((r < tol).all()), np.round(M).astype(int), float(r.max())


def find_basis_transform(Lp: np.ndarray, Lm: np.ndarray, tol: float = 0.12):
    """在 Lm = Lp @ U 的整数基变换中搜索（U ∈ {-1,0,1}^(3x3), det=±1/±2）。

    canonical 的 primitive 格基未标准化，同格不同取向时 round(inv(Lp)@Lm)
    会失真；此处枚举小整数矩阵统一基。返回 (U, err, found)。
    """
    M = np.linalg.inv(Lp) @ Lm
    r = np.abs(M - np.round(M))
    if (r < tol).all():
        return np.round(M).astype(int), float(r.max()), True
    vals = np.array([-1, 0, 1])
    grid = np.stack(np.meshgrid(vals, vals, vals, vals, vals, vals, vals, vals, vals, indexing="ij"), axis=-1).reshape(-1, 9)
    Us = grid.reshape(-1, 3, 3)
    dets = np.linalg.det(Us)
    ok = np.isin(np.round(dets).astype(int), (1, -1, 2, -2))
    Us = Us[ok]
    diff = np.abs(M[None, :, :] - Us).max(axis=(1, 2))
    j = int(diff.argmin())
    if diff[j] < 0.5:
        return Us[j].astype(int), float(diff[j]), bool(diff[j] < tol)
    return np.round(M).astype(int), float(r.max()), False


def symmetry_ops(st: Structure, symprec: float = 0.3):
    try:
        return SpacegroupAnalyzer(st, symprec=symprec).get_symmetry_operations()
    except Exception:
        return []


def select_parent_cell(
    structures: list[tuple[int, Structure]],
    symprec: float = 0.3,
    max_candidates: int = 8,
    coarse_steps: int = 6,
):
    """以"对齐中位数最优"选择母体参考成员（能解释最多成员的格胞）。

    家族内可能存在多个格胞批次（不同松弛/多形体）；参考必须选在
    最大的批次上，否则全体成员的对齐残差系统性偏大。
    """
    by_size = {}
    for idx, st in sorted(structures, key=lambda x: (x[1].num_sites, x[0])):
        key = (st.num_sites, round(st.volume / 8) * 8)
        by_size.setdefault(key, idx)
    candidates = list(by_size.values())[:max_candidates]

    best = None
    for cidx in candidates:
        ref_st = dict(structures)[cidx]
        Lp = ref_st.lattice.matrix
        ref_fw = np.array([s.frac_coords for s in ref_st if s.species_string in ("P", "Ti", "V", "Fe", "Mn")])
        scores = []
        for idx, st in structures:
            T = st.lattice.matrix @ np.linalg.inv(Lp)
            fw = np.array([s.frac_coords for s in st if s.species_string in ("P", "Ti", "V", "Fe", "Mn")]) @ T
            _, sc = na_sublattice.best_shift(ref_fw, fw, Lp, steps=coarse_steps)
            scores.append(sc)
        scores = np.array(scores)
        ops = len(symmetry_ops(framework_only(ref_st), symprec))
        key = (float(np.mean(scores < 0.5)), -float(np.median(scores)), -ops, cidx)
        if best is None or key > best[0]:
            best = (key, cidx, ref_st, ops, scores)

    best_idx, best_st, best_ops, best_scores = best[1], best[2], best[3], best[4]
    Lp = best_st.lattice.matrix
    m_map = {}
    for idx, st in structures:
        ok, M, err = is_integer_superlattice(Lp, st.lattice.matrix)
        m_map[idx] = {
            "M": M.tolist(),
            "det": int(round(abs(np.linalg.det(M)))),
            "err": err,
            "exact": ok,
        }
    return {
        "ref_idx": best_idx,
        "lattice": Lp.tolist(),
        "symmetry_ops": best_ops,
        "superlattice_map": m_map,
        "n_candidates": len(structures),
        "ref_align_median": float(np.median(best_scores)),
        "ref_align_lt05": float(np.mean(best_scores < 0.5)),
    }


def _nearest_match(member_frac: np.ndarray, ref_frac: np.ndarray, lattice: np.ndarray, tol: float = 0.9):
    """每个 member 原子 -> 最近 ref 位点下标（PBC 最小镜像），失败返回 -1。"""
    d = member_frac[:, None, :] - ref_frac[None, :, :]
    d -= np.round(d)
    dist = np.linalg.norm(np.einsum("nmk,kl->nml", d, lattice), axis=2)
    j = dist.argmin(axis=1)
    ok = dist[np.arange(len(member_frac)), j] < tol
    return j, ok


TM_GROUP = ("Ti", "V", "Fe", "Mn")


def _same_slot(ref_sp: str, mem_sp: str) -> bool:
    """TM 位点是装饰槽：任意 TM 元素可落同一母体位点。"""
    if ref_sp in TM_GROUP and mem_sp in TM_GROUP:
        return True
    return ref_sp == mem_sp


def mean_framework(members: Iterable[tuple[list[str], np.ndarray]], ref_species: list[str], ref_frac: np.ndarray, lattice: np.ndarray) -> dict:
    """对已映射到母体框架（+shift）的成员位点求均值分数坐标。"""
    accum = {i: [] for i in range(len(ref_species))}
    n_used, n_failed = 0, 0

    for species, frac in members:
        j, ok = _nearest_match(frac, ref_frac, lattice)
        if ok.mean() < 0.8:
            n_failed += 1
            continue
        for k, (sp, mi) in enumerate(zip(species, j)):
            if ok[k] and _same_slot(ref_species[mi], sp):
                delta = frac[k] - ref_frac[mi]
                delta -= np.round(delta)
                accum[mi].append(delta)
        n_used += 1

    sites = []
    for mi in range(len(ref_species)):
        ds = accum[mi]
        if ds:
            mean_delta = np.mean(ds, axis=0)
            frac = (ref_frac[mi] + mean_delta) % 1.0
        else:
            frac = ref_frac[mi] % 1.0
        sites.append({"el": ref_species[mi], "frac": frac.tolist(), "n_samples": len(ds)})
    return {"sites": sites, "n_used": n_used, "n_failed": n_failed, "n_sites": len(sites)}


def build_parent(
    parent_cell: dict,
    framework_mean: dict,
    na_sites: list[dict],
    symprec: float = 0.3,
) -> dict:
    lattice = np.array(parent_cell["lattice"])
    fw_sites = framework_mean["sites"]
    framework_st = Structure(
        lattice,
        [s["el"] for s in fw_sites],
        [s["frac"] for s in fw_sites],
    )
    sg_num = 1
    wyckoffs = None
    try:
        sga = SpacegroupAnalyzer(framework_st, symprec=symprec)
        sg_num = sga.get_space_group_number()
        wyckoffs = sga.get_symmetry_dataset()["wyckoffs"]
    except Exception:
        pass
    return {
        "parent_id": "parent-pyro-p2o7",
        "ref_idx": parent_cell["ref_idx"],
        "lattice": lattice.tolist(),
        "framework_sites": fw_sites,
        "na_sites": na_sites,
        "spacegroup_number": sg_num,
        "wyckoffs": wyckoffs,
        "symprec": symprec,
        "symmetry_ops": parent_cell["symmetry_ops"],
        "superlattice_map": parent_cell["superlattice_map"],
        "n_fw_samples": framework_mean["n_used"],
        "n_fw_fail": framework_mean["n_failed"],
    }


def save_parent(parent: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"version": "v2", "parent": parent}, f, indent=1)

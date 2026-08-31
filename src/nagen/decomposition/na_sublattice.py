"""Step 3: Na 亚晶格推断（并集聚类 + 对称性约简）与占据向量。"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure

NA_MIN_SEP = 1.2  # Å，两个不同 Na 位点的最小间距（数据实测 Na-Na min 2.2）
NA_OCC_TOL = 0.8  # Å，成员 Na 原子到母体位点的匹配容差
TM_MATCH_TOL = 0.9  # Å


def frac_to_parent(frac: np.ndarray, M: np.ndarray) -> np.ndarray:
    """成员分数坐标 -> 母体分数坐标（L_m = L_p @ M）。"""
    return frac @ M


def best_shift(ref_frac: np.ndarray, member_frac: np.ndarray, lattice: np.ndarray, steps: int = 12):
    """网格搜索 + 局部细化，求成员框架对齐到母体框架的分数平移。"""
    def evaluate(shifts):
        # shifts (S,3); member(S,R,1,3) - ref(1,1,M,3)
        d = member_frac[None, :, None, :] + shifts[:, None, None, :] - ref_frac[None, None, :, :]
        d -= np.round(d)
        dc = np.einsum("srmk,kl->srml", d, lattice)
        dist = np.linalg.norm(dc, axis=3).min(axis=2).mean(axis=1)
        return dist

    axis = np.linspace(0, 1, steps, endpoint=False)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    score = evaluate(grid)
    best = grid[score.argmin()]
    # 局部细化
    fine = best + np.stack(
        np.meshgrid(
            np.linspace(-1 / steps, 1 / steps, 5),
            np.linspace(-1 / steps, 1 / steps, 5),
            np.linspace(-1 / steps, 1 / steps, 5),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    score2 = evaluate(fine)
    shift = fine[score2.argmin()] % 1.0
    return shift.tolist(), float(score2.min())


def best_shift_guided(
    ref_fw: np.ndarray,
    ref_na: np.ndarray,
    member_fw: np.ndarray,
    member_na: np.ndarray,
    lattice: np.ndarray,
    na_weight: float = 2.0,
    na_tol: float = 0.8,
    steps: int = 16,
):
    """框架 + Na 位点联合打分的平移搜索。

    框架对齐存在周期多解（错误局部极小），用参考母体的 Na 位点作锚
    消歧：候选平移需同时让成员 Na 落在已知母体 Na 位点上。
    """

    def fw_score(shifts):
        d = member_fw[None, :, None, :] + shifts[:, None, None, :] - ref_fw[None, None, :, :]
        d -= np.round(d)
        dc = np.einsum("srmk,kl->srml", d, lattice)
        return np.linalg.norm(dc, axis=3).min(axis=2).mean(axis=1)

    def na_miss(shifts):
        if len(member_na) == 0 or len(ref_na) == 0:
            return np.zeros(len(shifts))
        d = member_na[None, :, None, :] + shifts[:, None, None, :] - ref_na[None, None, :, :]
        d -= np.round(d)
        dc = np.einsum("srmk,kl->srml", d, lattice)
        dist = np.linalg.norm(dc, axis=3).min(axis=2)
        return (dist >= na_tol).mean(axis=1)

    axis = np.linspace(0, 1, steps, endpoint=False)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    score = fw_score(grid) + na_weight * na_miss(grid)
    best = grid[score.argmin()]
    fine = best + np.stack(
        np.meshgrid(
            np.linspace(-1 / steps, 1 / steps, 5),
            np.linspace(-1 / steps, 1 / steps, 5),
            np.linspace(-1 / steps, 1 / steps, 5),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    score2 = fw_score(fine) + na_weight * na_miss(fine)
    shift = fine[score2.argmin()] % 1.0
    return shift.tolist(), float(score2.min()), float(fw_score(shift[None, :])[0])


def cluster_sites(frac_list: np.ndarray, lattice: np.ndarray, min_sep: float = NA_MIN_SEP):
    """贪心聚类 Na 位点（笛卡尔 PBC 距离）。返回中心（分数坐标）。"""
    if len(frac_list) == 0:
        return np.zeros((0, 3))
    cart = frac_list @ lattice
    centers, counts = [], []
    for c in cart:
        best_i, best_d = None, np.inf
        for i, cc in enumerate(centers):
            d = c - cc
            d -= np.round(d @ np.linalg.inv(lattice)) @ lattice
            dist = np.linalg.norm(d)
            if dist < best_d:
                best_i, best_d = i, dist
        if best_i is not None and best_d < min_sep:
            n = counts[best_i]
            # PBC 感知平均：先把新点移到中心最近镜像，再累加
            delta = c - centers[best_i]
            delta -= np.round(delta @ np.linalg.inv(lattice)) @ lattice
            centers[best_i] = centers[best_i] + delta / (n + 1)
            counts[best_i] += 1
        else:
            centers.append(c.copy())
            counts.append(1)
    centers = np.array(centers)
    return np.linalg.solve(lattice.T, centers.T).T % 1.0


def symmetry_reduce(sites_frac: np.ndarray, ops, lattice: np.ndarray, tol: float = 0.6):
    """用母体框架空间群操作约简 Na 位点并集。"""
    if len(sites_frac) == 0:
        return np.zeros((0, 3))
    canonical = []
    for f in sites_frac:
        cand = [f]
        for op in ops:
            cand.append(np.array(op.operate(f)) % 1.0)
        for c in cand:
            cc = c @ lattice
            dup = False
            for exist in canonical:
                d = cc - exist
                d -= np.round(d @ np.linalg.inv(lattice)) @ lattice
                if np.linalg.norm(d) < tol:
                    dup = True
                    break
            if not dup:
                canonical.append(cc)
    canonical = np.array(canonical)
    return np.linalg.solve(lattice.T, canonical.T).T % 1.0


def occupancy_for_member(
    na_frac: np.ndarray, sites_frac: np.ndarray, lattice: np.ndarray, tol: float = 1.0
) -> list[int]:
    """每个 Na 原子贪心分配到最近位点（防止两个位点同时被一个 Na 双计）。"""
    occ = []
    if len(na_frac) == 0:
        return [0] * len(sites_frac)
    d = na_frac[:, None, :] - sites_frac[None, :, :]
    d -= np.round(d)
    dist = np.linalg.norm(np.einsum("nmk,kl->nml", d, lattice), axis=2)  # (N_na, K)
    occ = [0] * len(sites_frac)
    order = sorted(range(len(na_frac)), key=lambda i: dist[i].min())
    for i in order:
        j = int(dist[i].argmin())
        if dist[i, j] < tol and not occ[j]:
            occ[j] = 1
    return occ


def tm_decoration(
    elements: list[str],
    fracs: np.ndarray,
    parent_tm_sites: np.ndarray,
    lattice: np.ndarray,
    tol: float = TM_MATCH_TOL,
) -> list[str]:
    out = [[] for _ in range(len(parent_tm_sites))]
    for el, f in zip(elements, fracs):
        d = parent_tm_sites - f
        d -= np.round(d)
        dist = np.linalg.norm(d @ lattice, axis=1)
        j = int(dist.argmin())
        if dist[j] < tol:
            out[j].append(el)
    return ["/".join(sorted(set(x))) if x else "" for x in out]

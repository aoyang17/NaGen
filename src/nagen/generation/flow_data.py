"""把多母体 registry 转成 flow matching 训练张量。

表示：每个成员 = 理想化母体模板（框架位点 + 被占据 Na 位点）+
畸变场（delta 分数坐标 + delta 晶格）。flow 学习畸变场分布。
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from pymatgen.core import Structure

ELEM2Z = {"Na": 11, "P": 15, "Ti": 22, "V": 23, "Fe": 26, "Mn": 25, "O": 8, "F": 9}
Z2ELEM = {v: k for k, v in ELEM2Z.items()}
TM = ("Ti", "V", "Fe", "Mn")
N_MAX = 56  # 40 框架 + K≤12 Na + 余量
K_MAX = 12


def pbc_wrap_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    d -= np.round(d)
    return d


def load_registry(reg_path: str) -> list[dict]:
    return [json.loads(l) for l in open(reg_path)]


def load_parents(parents_path: str) -> dict[str, dict]:
    return {p["parent_id"]: p for p in json.load(open(parents_path))["parents"]}


def load_structures(canonical: str, idxs: set[int]) -> dict[int, Structure]:
    out = {}
    with open(canonical) as f:
        for line in f:
            d = json.loads(line)
            if d["idx"] in idxs:
                out[d["idx"]] = Structure.from_dict(d["primitive_structure"])
    return out


def build_dataset(reg_path: str, parents_path: str, canonical: str, out_dir: str) -> dict:
    registry = load_registry(reg_path)
    parents = load_parents(parents_path)
    idxs = {r["idx"] for r in registry}
    structs = load_structures(canonical, idxs)

    X_delta, L_delta, Z, mask, cond, energies, align, meta = [], [], [], [], [], [], [], []
    X_tpl_all, L_tpl_all, N_atoms = [], [], []
    cond_dim = None
    for r in registry:
        p = parents[r["parent_id"]]
        st = structs[r["idx"]]
        fw = p["framework_sites"]  # list of {el, frac}
        na_sites = np.array(p["na_sites"])
        K = len(na_sites)
        occ = np.array(r["occupancy"], dtype=int)
        if len(occ) != K:
            continue
        Lp = np.array(p["lattice"])
        # 成员映射到母体框架
        T = st.lattice.matrix @ np.linalg.inv(Lp)
        # 对齐平移（registry 未存 shift；用框架最近匹配重算）
        from nagen.decomposition import na_sublattice

        ref_fw = np.array([s["frac"] for s in fw if s["el"] in ("P", "Ti", "V", "Fe", "Mn")])
        fw_m = np.array([s.frac_coords for s in st if s.species_string in ("P", "Ti", "V", "Fe", "Mn")]) @ T
        shift, _ = na_sublattice.best_shift(ref_fw, fw_m, Lp)
        shift = np.array(shift)

        # 模板原子顺序：框架位点（母体顺序）+ 被占据 Na 位点
        n_fw = len(fw)
        n_occ = int(occ.sum())
        N = n_fw + n_occ
        z = np.zeros(N, dtype=np.int64)
        x_tpl = np.zeros((N, 3))
        x_mem = np.zeros((N, 3))
        m = np.ones(N, dtype=np.float32)
        # 框架
        tm_slots = [i for i, s in enumerate(fw) if s["el"] in TM]
        for i, s in enumerate(fw):
            z[i] = ELEM2Z[s["el"]]
            x_tpl[i] = s["frac"]
        # 成员框架原子 -> 母体槽位（最近匹配，TM 组无关）
        fw_mem_frac = np.array([s.frac_coords for s in st if s.species_string in ("P", "Ti", "V", "Fe", "Mn", "O")]) @ T
        fw_mem_el = [s.species_string for s in st if s.species_string in ("P", "Ti", "V", "Fe", "Mn", "O")]
        fw_mem_frac = (fw_mem_frac + shift) % 1.0
        for k, (el, f) in enumerate(zip(fw_mem_el, fw_mem_frac)):
            # 找同物种母体槽位（TM 组内任意）
            cand = [
                i
                for i, s in enumerate(fw)
                if (s["el"] == el) or (el in TM and s["el"] in TM)
            ]
            if not cand:
                continue
            d = np.array([fw[i]["frac"] for i in cand]) - f
            d -= np.round(d)
            j = cand[int(np.linalg.norm(d @ Lp, axis=1).argmin())]
            if el in TM:
                z[j] = ELEM2Z[el]  # 装饰元素写入模板槽位
            x_mem[j] = f
        # Na：成员 Na -> 被占据 Na 位点
        na_m = np.array([s.frac_coords for s in st if s.species_string == "Na"])
        if len(na_m):
            na_m = (na_m @ T + shift) % 1.0
        na_idx = np.where(occ == 1)[0]
        for ii, ni in enumerate(na_idx):
            j = n_fw + ii
            z[j] = 11
            x_tpl[j] = na_sites[ni]
        # 成员 Na 与位点一对一最近分配
        assigned = np.zeros(len(na_m), dtype=bool)
        for ii, ni in enumerate(na_idx):
            if len(na_m) == 0:
                break
            dd = na_sites[ni] - na_m
            dd -= np.round(dd)
            dist = np.linalg.norm(dd @ Lp, axis=1)
            k_best = int(dist.argmin())
            if not assigned[k_best]:
                x_mem[n_fw + ii] = na_m[k_best]
                assigned[k_best] = True
        # 畸变场
        delta = pbc_wrap_delta(x_mem, x_tpl)
        dL = (st.lattice.matrix - Lp).reshape(-1) / 5.0  # 归一化（Å 量级）
        # 条件：parent one-hot + occupancy(K) + TM 装饰 + 组成计数
        parent_list = sorted({r["parent_id"] for r in registry})
        c_parent = [1.0 if r["parent_id"] == pid else 0.0 for pid in parent_list]
        c_occ = occ.tolist() + [0.0] * (K_MAX - len(occ))
        tm_dec = [0.0] * (4 * 5)
        for si in range(4):
            if si < len(tm_slots):
                el = Z2ELEM.get(int(z[tm_slots[si]]), "")
                if el in TM:
                    tm_dec[si * 5 + TM.index(el)] = 1.0
        elem_order = ["Na", "P", "Ti", "V", "Fe", "Mn", "O", "F"]
        comp = [0.0] * len(elem_order)
        for zz in z:
            if int(zz) in Z2ELEM:
                comp[elem_order.index(Z2ELEM[int(zz)])] += 1.0
        c = c_parent + c_occ + tm_dec + comp
        cond_dim = len(c)

        X_delta.append(delta)
        L_delta.append(dL)
        X_tpl_all.append(x_tpl)
        L_tpl_all.append(Lp.reshape(-1) / 5.0)
        N_atoms.append(N)
        Z.append(z)
        mask.append(m)
        cond.append(c)
        energies.append(r["energy_mlff"] if r["energy_mlff"] is not None else float("nan"))
        align.append(r["align_rmsd"])
        meta.append({"idx": r["idx"], "parent_id": r["parent_id"], "canonical_id": r["canonical_id"]})

    # padding
    N = max(len(z) for z in Z)
    Xp, Lp_all, Zp, Mp = [], [], [], []
    for z, d, dl, m in zip(Z, X_delta, L_delta, mask):
        pad = N - len(z)
        Xp.append(np.vstack([d, np.zeros((pad, 3))]))
        Zp.append(np.concatenate([z, np.zeros(pad, dtype=np.int64)]))
        Mp.append(np.concatenate([m, np.zeros(pad)]))
        Lp_all.append(dl)

    Xt = []
    for d, n in zip(X_tpl_all, N_atoms):
        pad = N - n
        Xt.append(np.vstack([d, np.zeros((pad, 3))]))

    ds = {
        "X_delta": torch.tensor(np.stack(Xp), dtype=torch.float32),
        "L_delta": torch.tensor(np.stack(Lp_all), dtype=torch.float32),
        "Z": torch.tensor(np.stack(Zp), dtype=torch.long),
        "mask": torch.tensor(np.stack(Mp), dtype=torch.float32),
        "cond": torch.tensor(np.stack(cond), dtype=torch.float32),
        "energy": torch.tensor(energies, dtype=torch.float32),
        "align": torch.tensor(align, dtype=torch.float32),
        "X_tpl": torch.tensor(np.stack(Xt), dtype=torch.float32),
        "L_tpl": torch.tensor(np.stack(L_tpl_all), dtype=torch.float32),
        "N_atoms": torch.tensor(N_atoms, dtype=torch.long),
        "N_max": N,
        "cond_dim": cond_dim,
        "parents": sorted({r["parent_id"] for r in registry}),
        "meta": meta,
    }
    os.makedirs(out_dir, exist_ok=True)
    torch.save(ds, os.path.join(out_dir, "flow_dataset.pt"))
    return ds


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="data/decomposition/multi_parent/registry_v2.jsonl")
    ap.add_argument("--parents", default="data/decomposition/multi_parent/parents_v2.json")
    ap.add_argument("--canonical", default="data/audit/canonical_v1.jsonl")
    ap.add_argument("--out", default="data/generation")
    args = ap.parse_args()
    ds = build_dataset(args.registry, args.parents, args.canonical, args.out)
    print(
        f"samples={ds['X_delta'].shape[0]} N={ds['N_max']} cond_dim={ds['cond_dim']} "
        f"parents={len(ds['parents'])}"
    )

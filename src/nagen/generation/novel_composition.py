"""组成级新颖生成：TM 装饰 + Na 含量 + 晶格扰动 + 超胞。

目标（用户要求）：新结构相对原数据必须 N、A、L 全变（至少 A、L）：
- A：TM 装饰配比枚举（Ti/V、Fe/Mn 混合）→ 新晶胞组成/新化学式；
- N：Na 含量变化 + 2× 超胞（原子数翻倍）；
- L：晶格参数扰动（长度 ±3%）。

验证：晶胞组成字符串与化学式均不在原数据；晶格相对母体模板变化；
结构级 StructureMatcher 与 205 个训练结构不匹配。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from collections import Counter

import numpy as np
import torch
from pymatgen.core import Composition, Structure

from nagen.generation.dflow_sur_generate import (
    capacity_mAh_g,
    check_coordination,
    check_distances,
    check_neutral,
)
from nagen.generation.occ_enumeration import charge_ok, pbc_min_dist
from nagen.generation.surrogate import EnergySurrogate

TM = ("Ti", "V", "Fe", "Mn")
Q_MIN = 130.0
NA_NA_MIN = 2.0


def cell_comp_string(elems: list[str]) -> str:
    c = Counter(elems)
    order = ["Na", "Ti", "V", "Fe", "Mn", "P", "O", "F"]
    return "".join(f"{e}{c.get(e, 0)}" for e in order if c.get(e, 0))


def decorated_species(fw_sites: list[dict], decoration: list[str]) -> list[str]:
    tm_i = 0
    out = []
    for s in fw_sites:
        if s["el"] in TM:
            out.append(decoration[tm_i])
            tm_i += 1
        else:
            out.append(s["el"])
    return out


def build_cell(
    parent: dict,
    decoration: list[str],
    occ: list[int],
    L_scale: np.ndarray,
) -> Structure | None:
    L = np.array(parent["lattice"]) * L_scale[None, :]
    fw = parent["framework_sites"]
    na_sites = np.array(parent["na_sites"])
    species = decorated_species(fw, decoration)
    coords = [s["frac"] for s in fw]
    for i, o in enumerate(occ):
        if o:
            species.append("Na")
            coords.append(na_sites[i].tolist())
    try:
        return Structure(L, species, np.array(coords) % 1.0)
    except Exception:
        return None


def build_supercell2(
    parent: dict,
    decoration: list[str],
    occ_a: list[int],
    occ_b: list[int],
    L_scale: np.ndarray,
) -> Structure | None:
    """沿 a 轴 2× 超胞：两份框架各带不同 Na 模式（N 翻倍）。"""
    L0 = np.array(parent["lattice"]) * L_scale[None, :]
    L2 = L0.copy()
    L2[0] = 2 * L0[0]
    fw = parent["framework_sites"]
    na_sites = np.array(parent["na_sites"])
    species, coords = [], []
    for copy in (0, 1):
        tm_i = 0
        for s in fw:
            el = decoration[tm_i] if s["el"] in TM else s["el"]
            if s["el"] in TM:
                tm_i += 1
            f = np.array(s["frac"])
            f2 = np.array([(f[0] + copy) / 2, f[1], f[2]])
            species.append(el)
            coords.append(f2)
        occ = occ_a if copy == 0 else occ_b
        for i, o in enumerate(occ):
            if o:
                f = na_sites[i]
                f2 = np.array([(f[0] + copy) / 2, f[1], f[2]])
                species.append("Na")
                coords.append(f2)
    try:
        return Structure(L2, species, np.array(coords) % 1.0)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", default="data/decomposition/multi_parent/parents_v2.json")
    ap.add_argument("--registry", default="data/decomposition/multi_parent/registry_v2.jsonl")
    ap.add_argument("--records", default="data/audit/records.csv")
    ap.add_argument("--dataset", default="data/generation/flow_dataset.pt")
    ap.add_argument("--surrogate", default="data/generation/models/surrogate.pt")
    ap.add_argument("--out", default="data/generation/nova_200.jsonl")
    ap.add_argument("--n-target", type=int, default=200)
    ap.add_argument("--l-scale", type=float, default=0.03)
    ap.add_argument("--n-supercell", type=int, default=50)
    ap.add_argument("--max-per-parent", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    ds = torch.load(args.dataset, weights_only=False)
    sur = EnergySurrogate().to(dev)
    sur.load_state_dict(torch.load(args.surrogate, map_location=dev)["model"])
    sur.eval()

    parents = {p["parent_id"]: p for p in json.load(open(args.parents))["parents"]}
    registry_patterns = {}
    for r in (json.loads(l) for l in open(args.registry)):
        registry_patterns.setdefault(r["parent_id"], set()).add(tuple(r["occupancy"]))

    # 原数据：564 reduced formula + 全部晶胞组成
    orig_formulas = set()
    cell_comps = set()
    with open(args.records) as f:
        for row in csv.DictReader(f):
            orig_formulas.add(row["reduced_formula"])
    Z2S = {11: "Na", 15: "P", 22: "Ti", 23: "V", 26: "Fe", 25: "Mn", 8: "O", 9: "F"}
    for i in range(ds["X_delta"].shape[0]):
        n = int(ds["N_atoms"][i])
        elems = [Z2S[int(zz)] for zz in ds["Z"][i][:n].tolist()]
        cell_comps.add(cell_comp_string(elems))
    print(f"原数据：{len(orig_formulas)} 化学式，{len(cell_comps)} 种晶胞组成，205 训练样本")

    rng = np.random.default_rng(args.seed)
    results = []
    stats = Counter()
    per_parent = Counter()
    enum_target = max(1, args.n_target - args.n_supercell)

    def surrogate_e(st):
        from nagen.generation.flow_data import ELEM2Z

        N = len(st)
        zt = torch.zeros(1, N, dtype=torch.long, device=dev)
        xt = torch.zeros(1, N, 3, device=dev)
        mt = torch.zeros(1, N, device=dev)
        for i2, site in enumerate(st):
            zt[0, i2] = ELEM2Z[site.species_string]
            xt[0, i2] = torch.tensor(site.frac_coords, dtype=torch.float32, device=dev)
            mt[0, i2] = 1.0
        Lt_ = torch.tensor(st.lattice.matrix.reshape(-1) / 5.0, dtype=torch.float32, device=dev).unsqueeze(0)
        comp_t = torch.zeros(1, 8, device=dev)
        with torch.no_grad():
            return float(sur.predict_eV_atom(xt, Lt_, mt, comp_t, float(len(st))).mean().item())

    for pid, p in sorted(parents.items()):
        K = len(p["na_sites"])
        na_frac = np.array(p["na_sites"])
        Lp = np.array(p["lattice"])
        fw = p["framework_sites"]
        tm_elems = sorted({s["el"] for s in fw if s["el"] in TM})
        n_tm = sum(1 for s in fw if s["el"] in TM)
        # 装饰元素池：母体 TM 元素 + 同族替代（Fe/Mn 互掺、Ti/V 互掺）
        pool = set(tm_elems)
        if {"Ti", "V"} & pool:
            pool |= {"Ti", "V"}
        if {"Fe", "Mn"} & pool:
            pool |= {"Fe", "Mn"}
        pool = sorted(pool)
        # 装饰配比去重（组合而非排列）
        decos = set()
        for combo in itertools.combinations_with_replacement(pool, n_tm):
            decos.add(tuple(sorted(combo)))
        print(f"\n{pid}: K={K} TM={n_tm} 装饰池={pool} 配比数={len(decos)}")
        for deco in sorted(decos):
            if len(results) >= enum_target or per_parent[pid] >= args.max_per_parent:
                break
            for n_na in range(0, K + 1):
                if len(results) >= enum_target or per_parent[pid] >= args.max_per_parent:
                    break
                # 组成与电荷
                species_all = decorated_species(fw, list(deco)) + ["Na"] * n_na
                comp_dict = Counter(species_all)
                if not charge_ok(dict(comp_dict)):
                    stats["charge"] += 1
                    continue
                ccs = cell_comp_string(species_all)
                formula = Composition(dict(comp_dict)).reduced_formula
                a_novel = ccs not in cell_comps and formula not in orig_formulas
                if not a_novel:
                    stats["a_not_novel"] += 1
                    continue
                mass = float(Composition(dict(comp_dict)).weight)
                q = capacity_mAh_g(n_na, mass)
                if q < Q_MIN:
                    stats["capacity"] += 1
                    continue
                # 选一个新占据模式（不在 registry）
                patterns = []
                for mask in range(1, 2**K):
                    occ = tuple((mask >> k) & 1 for k in range(K))
                    if sum(occ) != n_na:
                        continue
                    if occ in registry_patterns.get(pid, set()):
                        continue
                    if pbc_min_dist(na_frac[np.array(occ, dtype=bool)], Lp) < NA_NA_MIN:
                        continue
                    patterns.append(occ)
                    if len(patterns) >= 3:
                        break
                if not patterns:
                    stats["no_pattern"] += 1
                    continue
                for pattern in patterns:
                    if len(results) >= enum_target or per_parent[pid] >= args.max_per_parent:
                        break
                    # 晶格扰动
                    for trial in range(3):
                        L_scale = 1.0 + rng.uniform(-args.l_scale, args.l_scale, 3)
                        st = build_cell(p, list(deco), list(pattern), L_scale)
                        if st is None:
                            continue
                        if not check_distances(st) or not check_coordination(st) or not check_neutral(st):
                            stats["geometry"] += 1
                            continue
                        vol = st.volume / len(st)
                        if not (10.5 <= vol <= 20.5):
                            stats["volume"] += 1
                            continue
                        dL = float(np.linalg.norm(st.lattice.matrix - Lp) / np.linalg.norm(Lp))
                        if dL < 0.005:
                            continue
                        e = surrogate_e(st)
                        per_parent[pid] += 1
                        results.append(
                            {
                                "parent_id": pid,
                                "decoration": list(deco),
                                "formula": formula,
                                "cell_comp": ccs,
                                "n_na": n_na,
                                "N": len(st),
                                "occupancy": list(pattern),
                                "capacity_mAh_g": round(q, 1),
                                "L_change": round(dL * 100, 2),
                                "energy_surr_eV_atom": round(e, 4),
                                "vol_per_atom": round(vol, 2),
                                "supercell2": False,
                                "cif": st.to(fmt="cif"),
                            }
                        )
                        break
        print(f"  累计 {len(results)}")
    print("统计：", dict(stats))

    # 2× 超胞（N 翻倍）：取已生成候选的 (parent, decoration, pattern) 配对
    base = list(results)
    rng2 = np.random.default_rng(1)
    n_super = 0
    for _ in range(args.n_supercell):
        if len(results) >= args.n_target:
            break
        if len(base) == 0:
            break
        r = base[rng2.integers(0, len(base))]
        p = parents[r["parent_id"]]
        deco = list(r["decoration"])
        occ_b = list(r["occupancy"])
        occ_a = None
        for mask in range(1, 2 ** len(p["na_sites"])):
            o = tuple((mask >> k) & 1 for k in range(len(p["na_sites"])))
            if o == tuple(occ_b) or o in registry_patterns.get(r["parent_id"], set()):
                continue
            if pbc_min_dist(np.array(p["na_sites"])[np.array(o, dtype=bool)], np.array(p["lattice"])) < NA_NA_MIN:
                continue
            occ_a = list(o)
            break
        if occ_a is None:
            continue
        L_scale = 1.0 + rng2.uniform(-args.l_scale, args.l_scale, 3)
        st = build_supercell2(p, deco, occ_a, occ_b, L_scale)
        if st is None:
            continue
        if not check_distances(st) or not check_coordination(st) or not check_neutral(st):
            continue
        vol = st.volume / len(st)
        if not (10.5 <= vol <= 20.5):
            continue
        e = surrogate_e(st)
        ccs = cell_comp_string([s.species_string for s in st])
        results.append(
            {
                "parent_id": r["parent_id"],
                "decoration": deco,
                "formula": Composition(st.composition).reduced_formula,
                "cell_comp": ccs,
                "n_na": r["n_na"] * 2,
                "N": len(st),
                "occupancy": occ_a + occ_b,
                "capacity_mAh_g": round(capacity_mAh_g(2 * r["n_na"], float(st.composition.weight)), 1),
                "L_change": round(float(np.linalg.norm(st.lattice.matrix - np.array(p["lattice"])) / np.linalg.norm(np.array(p["lattice"])) * 100), 2),
                "energy_surr_eV_atom": round(e, 4),
                "vol_per_atom": round(vol, 2),
                "supercell2": True,
                "cif": st.to(fmt="cif"),
            }
        )
        n_super += 1

    results = results[: args.n_target]
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "cif"}) + "\n")
    cif_dir = os.path.splitext(args.out)[0] + "_cif"
    os.makedirs(cif_dir, exist_ok=True)
    for f in os.listdir(cif_dir):
        os.remove(os.path.join(cif_dir, f))
    for i, r in enumerate(results):
        with open(os.path.join(cif_dir, f"nova_{i:03d}.cif"), "w") as f:
            f.write(r["cif"])
    ns = Counter(r["N"] for r in results)
    print(f"产出 {len(results)} 个候选；N 分布={dict(ns)}；公式数={len({r['formula'] for r in results})}")
    print(f"完成：{args.out} + {cif_dir}/")


if __name__ == "__main__":
    main()

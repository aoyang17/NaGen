"""占据枚举：在母体 K 个 Na 位点上枚举数据中未出现过的占据模式，
生成与原数据不同的结构（新化学式 或 新 Na 排序），并过约束过滤。

新颖性保证（双重校验）：
1) 组成级：reduced formula 不在原始 564 个化学式内；
2) 结构级：占据向量不在 registry（205 个训练样本）内；
3) 指纹级：RDF 指纹到 205 个训练结构的最小距离 > 阈值。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter

import numpy as np
import torch
from pymatgen.core import Composition, Structure

from nagen.generation.flow_data import ELEM2Z, Z2ELEM
from nagen.generation.flow_model import VelocityNet, euler_sample
from nagen.generation.surrogate import EnergySurrogate
from nagen.generation.dflow_sur_generate import (
    capacity_mAh_g,
    check_coordination,
    check_distances,
    check_neutral,
)

Q_MIN = 130.0
NA_NA_MIN = 2.0  # Å
FP_NOVELTY_MIN = 0.005  # 归一化指纹 L2 距离阈值（RDF 对 Na 排序不敏感，仅排除近重复）


def pbc_min_dist(fracs: np.ndarray, L: np.ndarray) -> float:
    if len(fracs) < 2:
        return 1e9
    best = 1e9
    for i in range(len(fracs)):
        for j in range(i + 1, len(fracs)):
            d = fracs[i] - fracs[j]
            d -= np.round(d)
            best = min(best, float(np.linalg.norm(d @ L)))
    return best


def build_from_pattern(
    parent: dict, occ: list[int], noise: np.ndarray | None = None, rng: np.random.Generator | None = None
) -> Structure | None:
    """母体模板 + 占据 → 结构（Na 放在占据位点，可选加噪）。"""
    L = np.array(parent["lattice"])
    fw = parent["framework_sites"]
    na_sites = np.array(parent["na_sites"])
    species = [s["el"] for s in fw]
    coords = [s["frac"] for s in fw]
    for i, o in enumerate(occ):
        if o:
            species.append("Na")
            coords.append(na_sites[i].tolist())
    coords = np.array(coords) % 1.0
    if noise is not None:
        coords = (coords + noise[: len(coords)]) % 1.0
    try:
        return Structure(L, species, coords)
    except Exception:
        return None


def fingerprint(st: Structure, ds: dict, sur: EnergySurrogate, dev: str) -> np.ndarray:
    from nagen.generation.surrogate import rdf_descriptor

    n = len(st)
    z = torch.zeros(1, ds["N_max"], dtype=torch.long, device=dev)
    x = torch.zeros(1, ds["N_max"], 3, device=dev)
    m = torch.zeros(1, ds["N_max"], device=dev)
    for i, s in enumerate(st):
        z[0, i] = ELEM2Z[s.species_string]
        x[0, i] = torch.tensor(s.frac_coords, dtype=torch.float32, device=dev)
        m[0, i] = 1.0
    L = torch.tensor(st.lattice.matrix.reshape(-1) / 5.0, dtype=torch.float32, device=dev).unsqueeze(0)
    comp = torch.zeros(1, 8, device=dev)
    with torch.no_grad():
        return rdf_descriptor(x, L, m, comp).cpu().numpy()[0]


def charge_ok(comp_dict: dict) -> bool:
    """氧化态猜测可得中性组合（返回 False 若无法确定）。"""
    try:
        comp = Composition(comp_dict)
        guesses = comp.oxi_state_guesses()
        return len(guesses) > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", default="data/decomposition/multi_parent/parents_v2.json")
    ap.add_argument("--registry", default="data/decomposition/multi_parent/registry_v2.jsonl")
    ap.add_argument("--records", default="data/audit/records.csv")
    ap.add_argument("--dataset", default="data/generation/flow_dataset.pt")
    ap.add_argument("--flow", default="data/generation/models/flow.pt")
    ap.add_argument("--surrogate", default="data/generation/models/surrogate.pt")
    ap.add_argument("--out", default="data/generation/novel_200.jsonl")
    ap.add_argument("--n-target", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--noise-scale", type=float, default=0.0)
    ap.add_argument("--delta-scale", type=float, default=0.1)
    ap.add_argument("--max-per-parent", type=int, default=60)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    ds = torch.load(args.dataset, weights_only=False)
    flow = VelocityNet(cond_dim=ds["cond_dim"]).to(dev)
    flow.load_state_dict(torch.load(args.flow, map_location=dev)["model"])
    flow.eval()
    sur = EnergySurrogate().to(dev)
    sur.load_state_dict(torch.load(args.surrogate, map_location=dev)["model"])
    sur.eval()

    parents = {p["parent_id"]: p for p in json.load(open(args.parents))["parents"]}
    # 训练集占据模式 + 指纹
    registry_patterns = {}
    train_fps = []
    for r in (json.loads(l) for l in open(args.registry)):
        registry_patterns.setdefault(r["parent_id"], set()).add(tuple(r["occupancy"]))
    for i in range(ds["X_delta"].shape[0]):
        st = None  # 指纹用模板结构
        train_fps.append(None)
    # 预计算训练指纹（模板结构）
    Z2S = {11: "Na", 15: "P", 22: "Ti", 23: "V", 26: "Fe", 25: "Mn", 8: "O", 9: "F"}
    for i in range(ds["X_delta"].shape[0]):
        n = int(ds["N_atoms"][i])
        z = [Z2S[int(zz)] for zz in ds["Z"][i][:n].tolist()]
        x = ds["X_tpl"][i][:n].numpy() % 1.0
        L = (ds["L_tpl"][i].numpy() * 5.0).reshape(3, 3)
        st = Structure(L, z, x)
        train_fps[i] = fingerprint(st, ds, sur, dev)

    # 原始 564 化学式
    import csv
    orig_formulas = set()
    with open(args.records) as f:
        for row in csv.DictReader(f):
            orig_formulas.add(row["reduced_formula"])
    print(f"原始化学式：{len(orig_formulas)}，训练结构：{len(train_fps)}，母体：{len(parents)}")

    rng = np.random.default_rng(0)
    results, seen_patterns = [], set()
    stats = Counter()
    per_parent_count = Counter()
    charge_cache = {}
    train_fp_norm = [t / max(np.linalg.norm(t), 1e-12) for t in train_fps]
    for pid, p in sorted(parents.items()):
        K = len(p["na_sites"])
        na_frac = np.array(p["na_sites"])
        Lp = np.array(p["lattice"])
        fw_elems = [s["el"] for s in p["framework_sites"]]
        n_pat = 0
        for mask in range(1, 2**K):
            occ = [(mask >> k) & 1 for k in range(K)]
            if tuple(occ) in registry_patterns.get(pid, set()):
                stats["in_registry"] += 1
                continue
            if tuple(occ) in seen_patterns:
                continue
            n_na = sum(occ)
            # Na-Na 距离
            occ_frac = na_frac[np.array(occ, dtype=bool)]
            if pbc_min_dist(occ_frac, Lp) < NA_NA_MIN:
                stats["na_na"] += 1
                continue
            # 组成 + 电荷（按 (parent, n_na) 缓存）
            comp_dict = Counter(fw_elems)
            comp_dict["Na"] = n_na
            ckey = (pid, n_na)
            if ckey not in charge_cache:
                charge_cache[ckey] = charge_ok(dict(comp_dict))
            if not charge_cache[ckey]:
                stats["charge"] += 1
                continue
            formula = Composition(dict(comp_dict)).reduced_formula
            comp_novel = formula not in orig_formulas
            mass = float(Composition(dict(comp_dict)).weight)
            q = capacity_mAh_g(n_na, mass)
            if q < Q_MIN:
                stats["capacity"] += 1
                continue
            # 生成几何（纯模板；可选 flow 畸变噪声）
            for s in range(args.seeds):
                noise = (
                    rng.normal(0, args.noise_scale, (len(fw_elems) + n_na, 3))
                    if args.noise_scale > 0
                    else None
                )
                st = build_from_pattern(p, occ, noise)
                if st is None:
                    continue
                if not check_distances(st) or not check_coordination(st):
                    stats["geometry"] += 1
                    continue
                if not check_neutral(st):
                    stats["charge2"] += 1
                    continue
                vol = st.volume / len(st)
                if not (10.5 <= vol <= 20.5):
                    stats["volume"] += 1
                    continue
                fp = fingerprint(st, ds, sur, dev)
                fp = fp / max(np.linalg.norm(fp), 1e-12)
                min_fp = float(min(np.linalg.norm(fp - t) for t in train_fp_norm))
                if min_fp < FP_NOVELTY_MIN:
                    stats["fp_novel"] += 1
                    continue
                with torch.no_grad():
                    # 代理能量（新结构，代理外推，仅供排序）
                    zt = torch.zeros(1, ds["N_max"], dtype=torch.long, device=dev)
                    xt = torch.zeros(1, ds["N_max"], 3, device=dev)
                    mt = torch.zeros(1, ds["N_max"], device=dev)
                    for i2, site in enumerate(st):
                        zt[0, i2] = ELEM2Z[site.species_string]
                        xt[0, i2] = torch.tensor(site.frac_coords, dtype=torch.float32, device=dev)
                        mt[0, i2] = 1.0
                    Lt_ = torch.tensor(st.lattice.matrix.reshape(-1) / 5.0, dtype=torch.float32, device=dev).unsqueeze(0)
                    comp_t = torch.zeros(1, 8, device=dev)
                    e = float(sur.predict_eV_atom(xt, Lt_, mt, comp_t, float(len(st))).mean().item())
                n_pat += 1
                per_parent_count[pid] += 1
                results.append(
                    {
                        "parent_id": pid,
                        "formula": formula,
                        "comp_novel": comp_novel,
                        "n_na": n_na,
                        "occupancy": occ,
                        "seed": s,
                        "capacity_mAh_g": round(q, 1),
                        "energy_surr_eV_atom": round(e, 4),
                        "min_fp_to_train": round(min_fp, 3),
                        "min_dist": round(float(min(st.get_distance(a, b) for a in range(len(st)) for b in range(a + 1, len(st)))), 3),
                        "vol_per_atom": round(vol, 2),
                        "cif": st.to(fmt="cif"),
                    }
                )
                seen_patterns.add(tuple(occ))
                if len(results) >= args.n_target or per_parent_count[pid] >= args.max_per_parent:
                    break
            if len(results) >= args.n_target:
                break
        print(f"{pid}: 新增 {n_pat} 个新结构（K={K}）")
        if len(results) >= args.n_target:
            break

    print("统计：", dict(stats))
    novel_comp = sum(1 for r in results if r["comp_novel"])
    print(f"产出 {len(results)} 个新结构，其中新化学式 {novel_comp} 个")
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "cif"}) + "\n")
    cif_dir = os.path.splitext(args.out)[0] + "_cif"
    os.makedirs(cif_dir, exist_ok=True)
    for i, r in enumerate(results):
        with open(os.path.join(cif_dir, f"novel_{i:03d}.cif"), "w") as f:
            f.write(r["cif"])
    print(f"完成：{args.out}（{len(results)} 条） + {cif_dir}/")


if __name__ == "__main__":
    main()

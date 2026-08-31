"""Dflow-SUR 可控生成：源空间优化 + docs/OPTIMIZATION.md 约束硬过滤。

约束口径（docs/OPTIMIZATION.md）：
- 电中性：氧化态猜测成功且电荷和为 0
- 键距：按 HARD_CONSTRAINTS_SPEC 键型下限（P-O 1.40 / TM-O 1.55 / Na-O 2.00 /
  O-O 2.00 / P-P 2.60 / Na-Na 2.20 Å）
- 容量：Q = n_Na·F/(3.6·M) ≥ 130 mAh/g（n_Na = 可脱嵌 Na 数）
- 热力学稳定（代理口径）：ΔE = E_surr − E_min(parent) ≤ 0.150 eV/atom
  （E_min 取该母体家族×Na 含量能量网格最小值；非 DFT hull，需 Dflow 验证）
- 配位：P 仅 PO4（CN=4），TM 仅 TMO4/5/6
- 体积：10.5–20.5 Å³/atom
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import Counter

import numpy as np
import torch
from pymatgen.core import Structure

from nagen.generation.dflow_infer import build_structure, validity_report
from nagen.generation.flow_model import VelocityNet, euler_sample
from nagen.generation.surrogate import EnergySurrogate, rdf_descriptor
from nagen.generation.flow_data import ELEM2Z, Z2ELEM
from pymatgen.core import Composition

FARADAY = 96485.0
Q_MIN = 130.0
DHULL_MAX = 0.150
VOL_WIN = (10.5, 20.5)
DIST_MIN_TABLE = {
    ("P", "O"): 1.40,
    ("O", "P"): 1.40,
    ("Ti", "O"): 1.55,
    ("V", "O"): 1.55,
    ("Fe", "O"): 1.55,
    ("Mn", "O"): 1.55,
    ("O", "Ti"): 1.55,
    ("O", "V"): 1.55,
    ("O", "Fe"): 1.55,
    ("O", "Mn"): 1.55,
    ("Na", "O"): 2.00,
    ("O", "Na"): 2.00,
    ("O", "O"): 2.00,
    ("P", "P"): 2.60,
    ("Na", "Na"): 2.20,
}

# 全局（fork 继承）
_G = {}


def capacity_mAh_g(n_na: int, mass_g: float) -> float:
    return n_na * FARADAY / (3.6 * mass_g)


def check_distances(st: Structure) -> bool:
    for i in range(len(st)):
        for j in range(i + 1, len(st)):
            a, b = st[i].species_string, st[j].species_string
            dmin = DIST_MIN_TABLE.get((a, b), DIST_MIN_TABLE.get((b, a), 1.2))
            if st.get_distance(i, j) < dmin:
                return False
    return True


def check_coordination(st: Structure) -> bool:
    for i, s in enumerate(st):
        if s.species_string == "P":
            cn = sum(1 for j in range(len(st)) if j != i and st.get_distance(i, j) < 2.15)
            if cn != 4:
                return False
        elif s.species_string in ("Ti", "V", "Fe", "Mn"):
            cn = sum(1 for j in range(len(st)) if j != i and st.get_distance(i, j) < 2.4)
            if cn not in (4, 5, 6):
                return False
    return True


def check_neutral(st: Structure) -> bool:
    try:
        s2 = st.copy()
        s2.add_oxidation_state_by_guess()
        total = 0.0
        for site in s2:
            for sp, occu in site.species.items():
                oxi = getattr(sp, "oxi_state", 0)
                total += (oxi or 0) * occu
        return abs(total) < 1e-6
    except Exception:
        return False


def surrogate_energy_tensor(sur, x_full, L_full, mask, comp, n_atoms) -> float:
    with torch.no_grad():
        e = sur.predict_eV_atom(x_full, L_full, mask, comp, n_atoms)
    return float(e.mean().item())


def fingerprint(st: Structure, sur: EnergySurrogate, dev: str) -> np.ndarray:
    n = len(st)
    z = torch.zeros(1, _G["N"], dtype=torch.long, device=dev)
    x = torch.zeros(1, _G["N"], 3, device=dev)
    m = torch.zeros(1, _G["N"], device=dev)
    for i, s in enumerate(st):
        z[0, i] = _G["Z2I"][s.species_string]
        x[0, i] = torch.tensor(s.frac_coords, device=dev)
        m[0, i] = 1.0
    L = torch.tensor(st.lattice.matrix.reshape(-1) / 5.0, dtype=torch.float32, device=dev).unsqueeze(0)
    comp = torch.zeros(1, 8, device=dev)
    with torch.no_grad():
        desc = rdf_descriptor(x, L, m, comp)
    return desc.cpu().numpy()[0]


def dflow_sur_candidate(args):
    """单个候选：源空间优化 + 结构重建 + 能量。返回 dict 或 None。"""
    idx, seed, ode_steps, opt_steps, lr, delta_scale = args
    ds, flow, sur, dev = _G["ds"], _G["flow"], _G["sur"], _G["dev"]
    torch.manual_seed(seed)
    Xt = ds["X_tpl"][idx].unsqueeze(0).to(dev)
    Lt = ds["L_tpl"][idx].unsqueeze(0).to(dev)
    z = ds["Z"][idx].unsqueeze(0).to(dev)
    mask = ds["mask"][idx].unsqueeze(0).to(dev)
    cond = ds["cond"][idx].unsqueeze(0).to(dev)
    n_atoms = float(mask.sum())
    comp = ds["cond"][idx, -8:].unsqueeze(0).to(dev)
    x0 = torch.randn_like(Xt, requires_grad=True)
    L0 = torch.randn_like(Lt, requires_grad=True)
    opt = torch.optim.Adam([x0, L0], lr=lr)
    for _ in range(opt_steps):
        opt.zero_grad()
        delta, dL = euler_sample(flow, x0, L0, z, mask, cond, steps=ode_steps)
        delta = delta * delta_scale
        dL = dL * delta_scale
        x_full = torch.remainder(Xt + delta + 0.5, 1.0)
        L_full = (Lt + dL) * 5.0
        e = sur.predict_eV_atom(x_full, L_full, mask, comp, n_atoms)
        dist = _G["pbc"](x_full, L_full.reshape(-1, 3, 3), mask)
        penalty = (torch.relu(1.2 - dist) ** 2).mean()
        loss = e.mean() + 5.0 * penalty
        loss.backward()
        opt.step()
    with torch.no_grad():
        delta, dL = euler_sample(flow, x0.detach(), L0.detach(), z, mask, cond, steps=ode_steps)
        delta = delta * delta_scale
        dL = dL * delta_scale
        x_full = torch.remainder(Xt + delta + 0.5, 1.0)
        L_full = (Lt + dL) * 5.0
        e_final = float(sur.predict_eV_atom(x_full, L_full, mask, comp, n_atoms).mean().item())
    st = build_structure(z.cpu().numpy()[0], x_full.cpu().numpy()[0], L_full.cpu().numpy()[0], mask.cpu().numpy()[0])
    return {"idx": idx, "seed": seed, "energy_surr": e_final, "structure": st}


def hard_filter(st: Structure, energy_surr: float, parent_min_e: float, n_na: int) -> tuple[bool, list[str]]:
    fails = []
    if st is None:
        return False, ["parse"]
    if not check_distances(st):
        fails.append("distance")
    if not check_coordination(st):
        fails.append("coordination")
    if not check_neutral(st):
        fails.append("charge")
    vol = st.volume / len(st)
    if not (VOL_WIN[0] <= vol <= VOL_WIN[1]):
        fails.append(f"volume({vol:.1f})")
    mass = float(st.composition.weight)
    q = capacity_mAh_g(n_na, mass)
    if q < Q_MIN:
        fails.append(f"capacity({q:.0f})")
    dh = energy_surr - parent_min_e
    if dh > DHULL_MAX:
        fails.append(f"dhull_proxy({dh:.3f})")
    return len(fails) == 0, fails


def worker(args):
    return dflow_sur_candidate(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/generation/flow_dataset.pt")
    ap.add_argument("--flow", default="data/generation/models/flow.pt")
    ap.add_argument("--surrogate", default="data/generation/models/surrogate.pt")
    ap.add_argument("--parents", default="data/decomposition/multi_parent/parents_v2.json")
    ap.add_argument("--report", default="data/decomposition/multi_parent/report_v2.json")
    ap.add_argument("--out", default="data/generation/candidates_200.jsonl")
    ap.add_argument("--summary", default="data/generation/candidates_summary.md")
    ap.add_argument("--n-target", type=int, default=200)
    ap.add_argument("--seeds-per-target", type=int, default=40)
    ap.add_argument("--ode-steps", type=int, default=8)
    ap.add_argument("--opt-steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--delta-scale", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-per-target", type=int, default=24)
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
    from nagen.generation import surrogate as _sur

    _G.update(
        {
            "ds": ds,
            "flow": flow,
            "sur": sur,
            "dev": dev,
            "N": ds["N_max"],
            "Z2I": ELEM2Z,
            "pbc": _sur.pbc_pair_distances,
        }
    )

    # 目标列表：(parent, n_na) 组合，容量 Q≥130 才进生成目标。
    # parent_min_e 直接由数据集 MLFF 能量按母体取最小（与代理学习口径一致）。
    parent_min_e = {}
    for i, m in enumerate(ds["meta"]):
        e = float(ds["energy"][i])
        if not np.isnan(e):
            parent_min_e[m["parent_id"]] = min(parent_min_e.get(m["parent_id"], 1e9), e)
    # 从 dataset 中选每个 (parent, n_na, 装饰) 的代表样本
    groups = {}
    for i, m in enumerate(ds["meta"]):
        n_na = int(ds["Z"][i][ds["mask"][i] > 0].tolist().count(11))
        key = (m["parent_id"], n_na)
        groups.setdefault(key, []).append(i)
    targets = []
    for key, idxs in sorted(groups.items()):
        pid, n_na = key
        rep = ds["meta"][idxs[0]]
        # 模板质量（mass 用母体模板组成近似）
        z = ds["Z"][idxs[0]][: ds["N_atoms"][idxs[0]]]
        comp_dict = {}
        for zz in z.tolist():
            el = Z2ELEM.get(int(zz))
            if el:
                comp_dict[el] = comp_dict.get(el, 0) + 1
        mass = float(Composition(comp_dict).weight)
        q = capacity_mAh_g(n_na, mass)
        if q >= Q_MIN:
            targets.append({"parent": pid, "n_na": n_na, "rep_idx": idxs[0], "Q": q, "mass": mass})
    print(f"目标组合（Q≥{Q_MIN:.0f}）：{len(targets)} 个")
    for t in targets:
        print(f"  {t['parent']} n_Na={t['n_na']} Q={t['Q']:.0f}")

    # 并行 Dflow-SUR
    jobs = []
    for t in targets:
        for s in range(args.seeds_per_target):
            jobs.append((t["rep_idx"], s, args.ode_steps, args.opt_steps, args.lr, args.delta_scale))

    results = []
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(worker, jobs):
            results.append(r)

    # 硬过滤 + 去重
    passed, stage_counts = [], Counter()
    fps = []
    seen_fp = []
    per_target = Counter()
    for r in results:
        st = r["structure"]
        meta = ds["meta"][r["idx"]]
        n_na = int(ds["Z"][r["idx"]][ds["mask"][r["idx"]] > 0].tolist().count(11))
        ok, fails = hard_filter(st, r["energy_surr"], parent_min_e.get(meta["parent_id"], 0.0), n_na)
        stage_counts["optimized"] += 1
        if st is not None:
            stage_counts["parsed"] += 1
        if ok:
            fp = fingerprint(st, sur, dev)
            dup = False
            for f0 in seen_fp:
                if np.linalg.norm(fp - f0) < 0.03:
                    dup = True
                    break
            if not dup:
                seen_fp.append(fp)
                per_target[meta["parent_id"]] += 1
                passed.append(
                    {
                        "idx": r["idx"],
                        "seed": r["seed"],
                        "parent_id": meta["parent_id"],
                        "n_na": n_na,
                        "formula": st.composition.reduced_formula,
                        "energy_surr_eV_atom": round(r["energy_surr"], 4),
                        "min_dist": validity_report(st)["min_dist"],
                        "vol_per_atom": round(st.volume / len(st), 2),
                        "capacity_mAh_g": round(capacity_mAh_g(n_na, float(st.composition.weight)), 1),
                        "dhull_proxy": round(r["energy_surr"] - parent_min_e.get(meta["parent_id"], 0.0), 4),
                        "cif": st.to(fmt="cif"),
                    }
                )
        if len(passed) >= args.n_target or (
            len(passed) >= args.n_target * 0.7 and all(per_target[p] >= args.max_per_target for p in per_target)
        ):
            break

    with open(args.out, "w") as f:
        for c in passed:
            f.write(json.dumps({k: v for k, v in c.items() if k != "cif"}) + "\n")
    # CIF 目录
    cif_dir = os.path.splitext(args.out)[0] + "_cif"
    os.makedirs(cif_dir, exist_ok=True)
    for i, c in enumerate(passed):
        with open(os.path.join(cif_dir, f"candidate_{i:03d}.cif"), "w") as f:
            f.write(c["cif"])

    print(f"\n优化候选：{len(results)}，硬过滤通过并去重：{len(passed)}")
    print("阶段统计：", dict(stage_counts))
    # 汇总报告
    lines = [
        "# Dflow-SUR 可控生成候选汇总",
        "",
        f"- 日期：2026-08-18",
        f"- 优化候选：{len(results)}，解析成功：{stage_counts.get('parsed', 0)}，"
        f"通过全部硬约束并去重：{len(passed)}",
        f"- 生成参数：ode_steps={args.ode_steps}, opt_steps={args.opt_steps}, "
        f"delta_scale={args.delta_scale}, seeds/target={args.seeds_per_target}",
        "",
        "## 约束（docs/OPTIMIZATION.md）",
        "",
        "- 电中性（氧化态猜测）；键距按 HARD_CONSTRAINTS_SPEC 下限；",
        "- P 仅 PO4、TM 仅 TMO4/5/6；体积 10.5–20.5 Å³/atom；",
        "- 容量 Q = n_Na·F/(3.6·M) ≥ 130 mAh/g；",
        "- ΔE_hull 代理 = E_surr − E_min(parent) ≤ 0.150 eV/atom（需 Dflow/DFT 验证）。",
        "",
        "## 通过候选（按母体）",
        "",
        "| 母体 | 数量 | 平均代理能量 (eV/atom) | 平均容量 (mAh/g) | 平均 ΔE_proxy |",
        "|---|---|---|---|---|",
    ]
    by_parent = {}
    for c in passed:
        by_parent.setdefault(c["parent_id"], []).append(c)
    for pid, cs in sorted(by_parent.items()):
        lines.append(
            f"| {pid} | {len(cs)} | {np.mean([c['energy_surr_eV_atom'] for c in cs]):.3f} | "
            f"{np.mean([c['capacity_mAh_g'] for c in cs]):.0f} | "
            f"{np.mean([c['dhull_proxy'] for c in cs]):.3f} |"
        )
    lines += ["", "## 候选明细（前 20）", "", "| # | 母体 | 化学式 | n_Na | 能量 | 容量 | ΔE_proxy | min_d | vol |", "|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(passed[:20]):
        lines.append(
            f"| {i} | {c['parent_id']} | {c['formula']} | {c['n_na']} | {c['energy_surr_eV_atom']:.3f} | "
            f"{c['capacity_mAh_g']:.0f} | {c['dhull_proxy']:.3f} | {c['min_dist']} | {c['vol_per_atom']} |"
        )
    with open(args.summary, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"完成：{args.out}（{len(passed)} 条） + {cif_dir}/")


if __name__ == "__main__":
    main()

"""D-Flow 推理：对 flow 初始噪声 x0 做源空间梯度优化。

目标：最小化可微能量代理（+ 几何有效性软惩罚），
反传路径：x0 -> 显式欧拉 ODE -> 畸变场 -> 结构 -> RDF 代理 -> 能量。
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from pymatgen.core import Structure

from nagen.generation.flow_model import VelocityNet, euler_sample
from nagen.generation.surrogate import EnergySurrogate, pbc_pair_distances

Z2SYM = {11: "Na", 15: "P", 22: "Ti", 23: "V", 26: "Fe", 25: "Mn", 8: "O", 9: "F"}
DIST_MIN = 1.4  # Å


def build_structure(z: np.ndarray, x: np.ndarray, L: np.ndarray, mask: np.ndarray) -> Structure | None:
    """张量 -> pymatgen Structure（模板顺序，去 padding）。"""
    n = int(mask.sum())
    species = []
    coords = []
    for i in range(n):
        sym = Z2SYM.get(int(z[i]))
        if sym is None:
            return None
        species.append(sym)
        coords.append(np.array(x[i]) % 1.0)
    L3 = np.array(L).reshape(3, 3)
    if abs(np.linalg.det(L3)) < 1e-3:
        return None
    try:
        return Structure(L3, species, coords)
    except Exception:
        return None


def min_pair_distance(st: Structure) -> float:
    best = 1e9
    for i in range(len(st)):
        for j in range(i + 1, len(st)):
            d = st.get_distance(i, j)
            if d < best:
                best = d
    return best


def validity_report(st: Structure) -> dict:
    dmin = min_pair_distance(st)
    p_ok = all(
        sum(1 for j in range(len(st)) if j != i and st.get_distance(i, j) < 2.15) == 4
        for i, s in enumerate(st)
        if s.species_string == "P"
    )
    vol = st.volume / len(st)
    return {"min_dist": round(dmin, 3), "p_coord_ok": bool(p_ok), "vol_per_atom": round(vol, 2)}


def sample_target(ds: dict, flow: VelocityNet, idx: int, steps: int = 16, seed: int = 0, delta_scale: float = 0.3):
    """按数据集样本 idx 的模板/条件，从 flow 采样一个结构。"""
    torch.manual_seed(seed)
    dev = next(flow.parameters()).device
    Xt = ds["X_tpl"][idx].unsqueeze(0).to(dev)
    Lt = ds["L_tpl"][idx].unsqueeze(0).to(dev)
    z = ds["Z"][idx].unsqueeze(0).to(dev)
    mask = ds["mask"][idx].unsqueeze(0).to(dev)
    cond = ds["cond"][idx].unsqueeze(0).to(dev)
    x0 = torch.randn_like(Xt)
    L0 = torch.randn_like(Lt)
    delta, dL = euler_sample(flow, x0, L0, z, mask, cond, steps=steps)
    delta = delta * delta_scale
    dL = dL * delta_scale
    x_full = torch.remainder(Xt + delta + 0.5, 1.0)
    L_full = (Lt + dL) * 5.0
    return (
        x_full.detach().cpu().numpy()[0],
        L_full.detach().cpu().numpy()[0],
        z.cpu().numpy()[0],
        mask.cpu().numpy()[0],
        (x0, L0),
    )


def dflow_optimize(
    ds: dict,
    flow: VelocityNet,
    surrogate: EnergySurrogate,
    idx: int,
    steps: int = 12,
    opt_steps: int = 30,
    lr: float = 0.08,
    lam_valid: float = 5.0,
    seed: int = 0,
    delta_scale: float = 0.3,
):
    """源空间优化：x0 最小化代理能量 + 有效性软惩罚。返回轨迹。"""
    torch.manual_seed(seed)
    dev = next(flow.parameters()).device
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
    hist = []
    for it in range(opt_steps):
        opt.zero_grad()
        delta, dL = euler_sample(flow, x0, L0, z, mask, cond, steps=steps)
        delta = delta * delta_scale
        dL = dL * delta_scale
        x_full = torch.remainder(Xt + delta + 0.5, 1.0)
        L_full = (Lt + dL) * 5.0
        e = surrogate.predict_eV_atom(x_full, L_full, mask, comp, n_atoms)  # eV/atom
        # 有效性软惩罚：过近原子对
        dist = pbc_pair_distances(x_full, L_full.reshape(-1, 3, 3), mask)
        penalty = (torch.relu(DIST_MIN - dist) ** 2).mean()
        loss = e.mean() + lam_valid * penalty
        loss.backward()
        opt.step()
        with torch.no_grad():
            hist.append(
                {
                    "step": it,
                    "loss": loss.item(),
                    "energy": e.mean().item(),
                    "penalty": penalty.item(),
                }
            )
    with torch.no_grad():
        delta, dL = euler_sample(flow, x0.detach(), L0.detach(), z, mask, cond, steps=steps)
        delta = delta * delta_scale
        dL = dL * delta_scale
        x_full = torch.remainder(Xt + delta + 0.5, 1.0)
        L_full = (Lt + dL) * 5.0
    return (
        x_full.cpu().numpy()[0],
        L_full.cpu().numpy()[0],
        z.cpu().numpy()[0],
        mask.cpu().numpy()[0],
        hist,
    )


def main():
    import argparse
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/generation/flow_dataset.pt")
    ap.add_argument("--flow", default="data/generation/models/flow.pt")
    ap.add_argument("--surrogate", default="data/generation/models/surrogate.pt")
    ap.add_argument("--out", default="data/generation/dflow_results.json")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--delta-scale", type=float, default=0.3)
    ap.add_argument("--n-cand", type=int, default=16)
    args = ap.parse_args()

    ds = torch.load(args.dataset, weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flow = VelocityNet(cond_dim=ds["cond_dim"]).to(dev)
    flow.load_state_dict(torch.load(args.flow, map_location=dev)["model"])
    flow.eval()
    sur = EnergySurrogate().to(dev)
    ck = torch.load(args.surrogate, map_location=dev)
    sur.load_state_dict(ck["model"])
    sur.eval()

    # 代理一致性（训练集）：预测 vs MLFF
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = ds["X_delta"].to(dev)
    Xt = ds["X_tpl"].to(dev)
    Lt = ds["L_tpl"].to(dev)
    mask = ds["mask"].to(dev)
    comp = ds["cond"][:, -8:].to(dev)
    n_atoms = ds["mask"].sum(dim=1, keepdim=True).to(dev)
    with torch.no_grad():
        x_full = torch.remainder(Xt + X + 0.5, 1.0)
        L_full = (Lt + ds["L_delta"].to(dev)) * 5.0
        pred = sur.predict_eV_atom(x_full, L_full, mask, comp, n_atoms)
    true = ds["energy"]
    ok = ~torch.isnan(true)
    mae = (pred[ok] - true[ok]).abs().mean().item()
    ss_res = ((pred[ok] - true[ok]) ** 2).sum().item()
    ss_tot = ((true[ok] - true[ok].mean()) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot
    print(f"代理一致性：MAE={mae:.3f} eV/atom, R²={r2:.3f}（{int(ok.sum())} 样本）")

    # 评测目标：每母体取一个代表样本 + 若干随机样本
    results = {"samples": [], "ranked": [], "dflow": []}
    n = ds["X_delta"].shape[0]
    targets = list(range(0, n, max(1, n // args.n_samples)))[: args.n_samples]
    for idx in targets:
        meta = ds["meta"][idx]
        xf, Lf, z, mask, _ = sample_target(ds, flow, idx, seed=idx)
        st = build_structure(z, xf, Lf, mask)
        s_rep = {"idx": idx, "parent": meta["parent_id"], "formula": meta["canonical_id"], "valid": st is not None}
        if st is not None:
            s_rep.update(validity_report(st))
        results["samples"].append(s_rep)
        # 采样-排序：多候选 + 几何过滤 + 代理能量排序
        cands = []
        for c in range(args.n_cand):
            xc, Lc, zc, mc, _ = sample_target(ds, flow, idx, seed=idx * 100 + c, delta_scale=args.delta_scale)
            sc = build_structure(zc, xc, Lc, mc)
            if sc is None:
                continue
            vr = validity_report(sc)
            if vr["min_dist"] >= 1.2 and 10 <= vr["vol_per_atom"] <= 20:
                cands.append((sc, vr, c))
        ranked = None
        if cands:
            dev = next(flow.parameters()).device
            best_e = None
            for sc, vr, c in cands:
                zt = torch.tensor(z, dtype=torch.long).unsqueeze(0).to(dev)
                mt = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(dev)
                xt = torch.tensor(np.array([s.frac_coords for s in sc] + [[0, 0, 0]] * (ds["N_max"] - len(sc))), dtype=torch.float32).unsqueeze(0).to(dev)
                Lt_ = torch.tensor(sc.lattice.matrix.reshape(-1) / 5.0, dtype=torch.float32).unsqueeze(0).to(dev)
                comp = ds["cond"][idx, -8:].unsqueeze(0).to(dev)
                e = sur.predict_eV_atom(xt, Lt_, mt, comp, float(mt.sum())).item()
                if best_e is None or e < best_e[0]:
                    best_e = (e, vr, c)
            ranked = {"idx": idx, "parent": meta["parent_id"], "energy": best_e[0], "cand": best_e[2], **best_e[1]}
        results["ranked"].append(ranked)
        # D-Flow 优化（子集）
        if idx % 3 == 0:
            xo, Lo, z2, m2, hist = dflow_optimize(
                ds, flow, sur, idx, opt_steps=args.opt_steps, seed=idx, delta_scale=args.delta_scale
            )
            st_o = build_structure(z2, xo, Lo, m2)
            rep = {
                "idx": idx,
                "parent": meta["parent_id"],
                "energy_before": None,
                "energy_after": None,
                "loss_first": hist[0]["loss"],
                "loss_last": hist[-1]["loss"],
                "energy_first": hist[0]["energy"],
                "energy_last": hist[-1]["energy"],
                "valid": st_o is not None,
            }
            if st_o is not None:
                rep.update(validity_report(st_o))
            results["dflow"].append(rep)
        print(f"idx={idx} sample_valid={s_rep['valid']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    ok = sum(1 for s in results["samples"] if s["valid"])
    print(f"采样有效性：{ok}/{len(results['samples'])}")
    ranked_ok = [r for r in results["ranked"] if r is not None]
    if ranked_ok:
        es = [r["energy"] for r in ranked_ok]
        print(f"采样-排序：{len(ranked_ok)}/{len(results['ranked'])} 目标获得有效低能候选，代理能量中位 {np.median(es):.3f} eV/atom")
    if results["dflow"]:
        e0 = np.mean([r["energy_first"] for r in results["dflow"]])
        e1 = np.mean([r["energy_last"] for r in results["dflow"]])
        print(f"D-Flow 代理能量：{e0:.3f} -> {e1:.3f} eV（优化 {len(results['dflow'])} 个目标）")
    print(f"完成：{args.out}")


if __name__ == "__main__":
    main()

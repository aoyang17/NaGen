"""多母体（几何多形体）分解：把 el-aware 家族按框架对齐残差拆成几何一致核心，
每个核心一个理想化母体，产出 registry + 能量网格 + 容量表。

对应 TASK_DECOMPOSITION_PLAN.md §8–9（生成之前）：
- 多母体迭代扩展，提高分解覆盖率
- 家族 × Na 含量 MLFF 能量网格（hull 标签的输入，非 DFT hull）
- 容量 Q = ΔNa·F/(3.6·M)（ΔNa 按同一母体相邻 Na 含量占据差定义）

用法:
    python -m nagen.decomposition.multi_parent \\
        --decomp data/decomposition --records-family data/decomposition/record_family_v2.jsonl \\
        --canonical data/audit/canonical_v1.jsonl --records data/audit/records.csv
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
from pymatgen.core import Structure

from nagen.decomposition import na_sublattice, parent_search, validate
from nagen.decomposition._records import load_records

GATE = 0.35  # Å，几何一致核心门控
MIN_CORE = 5
FW_SPECIES = ("P", "Ti", "V", "Fe", "Mn", "O")
PTM_SPECIES = ("P", "Ti", "V", "Fe", "Mn")
TM_SPECIES = ("Ti", "V", "Fe", "Mn")
FARADAY = 96485.0  # C/mol


def load_pure_pyro(records: list[dict], framework_id: str | None = None) -> list[dict]:
    out = [
        r
        for r in records
        if r["q_label"] == "Q1"
        and "contains_F" not in r["flags"]
        and "contains_CN" not in r["flags"]
    ]
    if framework_id:
        out = [r for r in out if r["framework_id"] == framework_id]
    return out


def align_to_reference(structs: dict, idxs: list[int], ref_idx: int, Lp: np.ndarray):
    """全体成员映射到参考框架并做分数平移对齐，返回 {idx: (shift, score)}。"""
    ref_st = structs[ref_idx]
    ref_fw = np.array([s.frac_coords for s in ref_st if s.species_string in PTM_SPECIES])
    out = {}
    for idx in idxs:
        st = structs[idx]
        T = st.lattice.matrix @ np.linalg.inv(Lp)
        fw = np.array([s.frac_coords for s in st if s.species_string in PTM_SPECIES]) @ T
        sh, score = na_sublattice.best_shift(ref_fw, fw, Lp)
        out[idx] = {"shift": sh, "align_rmsd": score, "basis_T": T.tolist()}
    return out


def find_cores(structs: dict, family: list[dict], gate: float = GATE) -> tuple[list[dict], list[int]]:
    """迭代式几何核心发现：每次用对齐最优参考，单一参考门控取核心。"""
    remaining = sorted(r["idx"] for r in family)
    cores = []
    while len(remaining) >= MIN_CORE:
        pc = parent_search.select_parent_cell(
            [(idx, structs[idx]) for idx in remaining],
            max_candidates=8,
            coarse_steps=6,
        )
        ref = pc["ref_idx"]
        Lp = np.array(pc["lattice"])
        al = align_to_reference(structs, remaining, ref, Lp)
        core = [idx for idx in remaining if al[idx]["align_rmsd"] < gate]
        if len(core) < MIN_CORE:
            break
        cores.append({"members": sorted(core), "ref_idx": ref, "align": al})
        remaining = [idx for idx in remaining if idx not in core]
    return cores, remaining


def build_core_parent(
    structs: dict,
    core: list[int],
    ref_idx: int,
    align: dict,
    family_records: dict[int, dict],
    io_err: dict[int, bool],
) -> dict:
    """对几何一致核心构建理想化母体 + 占据 + 能量/容量表。"""
    ref_st = structs[ref_idx]
    Lp = ref_st.lattice.matrix

    ref_species = [s.species_string for s in ref_st if s.species_string in FW_SPECIES]
    ref_fw_all = np.array([s.frac_coords for s in ref_st if s.species_string in FW_SPECIES])
    ref_na = np.array([s.frac_coords for s in ref_st if s.species_string == "Na"])

    # 框架均值
    mapped = []
    for idx in core:
        st = structs[idx]
        T = np.array(align[idx]["basis_T"])
        shift = np.array(align[idx]["shift"])
        sp, fr = [], []
        for s in st:
            if s.species_string in FW_SPECIES:
                sp.append(s.species_string)
                fr.append(s.frac_coords)
        mapped.append((sp, (np.array(fr) @ T + shift) % 1.0))
    framework_mean = parent_search.mean_framework(mapped, ref_species, ref_fw_all, Lp)

    # Na 位点并集（PBC 感知空间聚类）
    na_union = []
    for idx in core:
        st = structs[idx]
        T = np.array(align[idx]["basis_T"])
        shift = np.array(align[idx]["shift"])
        na = np.array([s.frac_coords for s in st if s.species_string == "Na"])
        if len(na):
            na_union.append((na @ T + shift) % 1.0)
    na_union = np.vstack(na_union) if na_union else np.zeros((0, 3))
    na_sites = na_sublattice.cluster_sites(na_union, Lp, min_sep=1.2)

    parent_tm = np.array(
        [s["frac"] for s in framework_mean["sites"] if s["el"] in TM_SPECIES]
    )

    rows, rmsds, matchers = [], [], []
    energy_rows = []
    for idx in sorted(core):
        st = structs[idx]
        rec = family_records[idx]
        T = np.array(align[idx]["basis_T"])
        shift = np.array(align[idx]["shift"])
        na = np.array([s.frac_coords for s in st if s.species_string == "Na"])
        na_frac = (na @ T + shift) % 1.0 if len(na) else np.zeros((0, 3))
        occ = na_sublattice.occupancy_for_member(na_frac, na_sites, Lp)
        tm_frac = (
            np.array([s.frac_coords for s in st if s.species_string in TM_SPECIES]) @ T + shift
        ) % 1.0
        tm_el = [s.species_string for s in st if s.species_string in TM_SPECIES]
        tm = na_sublattice.tm_decoration(tm_el, tm_frac, parent_tm, Lp)
        inv_T = np.linalg.inv(T)
        rebuilt = validate.rebuild_structure(
            [{**s, "frac": (np.array(s["frac"]) - shift) @ inv_T % 1.0} for s in framework_mean["sites"]],
            (na_sites - shift) @ inv_T % 1.0,
            occ,
            st.lattice.matrix,
            None,
        )
        rmsd = validate.site_rmsd(rebuilt, st)
        rmsds.append(rmsd["rmsd"])
        matchers.append(validate.matcher_pass(rebuilt, st))
        n_na = sum(occ)
        rows.append(
            {
                "idx": idx,
                "canonical_id": rec["canonical_id"],
                "reduced_formula": rec["reduced_formula"],
                "n_na_observed": rec["n_na"],
                "n_na_occupancy": n_na,
                "occupancy": occ,
                "tm_decoration": tm,
                "align_rmsd": align[idx]["align_rmsd"],
                "energy_mlff": rec["energy_mlff"],
                "relax_ok": not io_err.get(idx, True),
            }
        )
        if rec["energy_mlff"] is not None:
            energy_rows.append(
                (n_na, rec["energy_mlff"], "/".join(tm), not io_err.get(idx, False))
            )

    # 能量网格（全部能量；relax_ok 单独计数）
    grid = collections.defaultdict(list)
    grid_relax = collections.defaultdict(list)
    for n_na, e, _, ok in energy_rows:
        grid[n_na].append(e)
        if ok:
            grid_relax[n_na].append(e)
    energy_grid = {
        str(k): {
            "n": len(v),
            "n_relax_ok": len(grid_relax[k]),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
        }
        for k, v in sorted(grid.items())
    }

    # 容量：相邻 Na 含量对，Q = ΔNa·F/(3.6·M_lowNa)
    capacities = {}
    na_levels = sorted(grid)
    for a, b in zip(na_levels, na_levels[1:]):
        d_na = b - a
        rep = next(r for r in rows if r["n_na_occupancy"] == a)
        st = structs[rep["idx"]]
        m_low = float(st.composition.weight)
        q = d_na * FARADAY / (3.6 * m_low)
        capacities[f"{a}->{b}"] = {"d_na": d_na, "Q_mAh_g": round(q, 1)}

    rmsds = np.array(rmsds)
    spearman = None
    relax_rows = [(a, b) for a, b, _, ok in energy_rows if ok]
    if len(relax_rows) >= 3:
        spearman = validate.spearman(
            np.array([a for a, _ in relax_rows], dtype=float),
            np.array([b for _, b in relax_rows]),
        )
    return {
        "ref_idx": ref_idx,
        "members": sorted(core),
        "n_members": len(core),
        "lattice": Lp.tolist(),
        "framework_sites": framework_mean["sites"],
        "na_sites": na_sites.tolist(),
        "K": int(len(na_sites)),
        "align_median": float(np.median([align[i]["align_rmsd"] for i in core])),
        "rmsd_median": float(np.median(rmsds)),
        "rmsd_le_0.3": float((rmsds <= 0.3).mean()),
        "matcher_rate": float(np.mean(matchers)),
        "energy_grid": energy_grid,
        "capacity": capacities,
        "spearman_energy_vs_nna": spearman,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", default="data/decomposition")
    ap.add_argument("--records-family", default=None)
    ap.add_argument("--canonical", default="data/audit/canonical_v1.jsonl")
    ap.add_argument("--records", default="data/audit/records.csv")
    ap.add_argument("--raw", default="/mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl")
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument(
        "--framework-id",
        default="0a746d6a684be22700341acc01071890",
        help="framework_id（默认 12 节点小焦磷酸框架；传 all 处理全部纯焦磷酸）",
    )
    args = ap.parse_args()
    out = os.path.join(args.decomp, "multi_parent")
    os.makedirs(out, exist_ok=True)
    gate = args.gate

    records = load_records(args.records_family or os.path.join(args.decomp, "record_family_v2.jsonl"))
    pure = load_pure_pyro(records, None if args.framework_id == "all" else args.framework_id)
    member_idx = {r["idx"] for r in pure}
    structs = {}
    with open(args.canonical) as f:
        for line in f:
            d = json.loads(line)
            if d["idx"] in member_idx:
                structs[d["idx"]] = Structure.from_dict(d["primitive_structure"])
    print(f"纯焦磷酸成员：{len(pure)}")

    # I/O error 标记
    import csv
    idx_of_mid = {}
    with open(args.records) as f:
        for row in csv.DictReader(f):
            idx_of_mid[row["material_id"]] = int(row["idx"])
    io_err = {}
    with open(args.raw) as f:
        for line in f:
            d = json.loads(line)
            mid = d["material_id"]
            if mid in idx_of_mid:
                io_err[idx_of_mid[mid]] = d.get("error_msg") is not None

    family_records = {r["idx"]: r for r in pure}
    by_family = collections.defaultdict(list)
    for r in pure:
        by_family[r["family_id"]].append(r)

    cores_all, registry, parents_out = [], [], []
    for fam_id, fam in sorted(by_family.items(), key=lambda kv: -len(kv[1])):
        print(f"\n家族 {fam_id[:10]} ({len(fam)} 成员)：找几何核心...")
        cores, tail = find_cores(structs, fam, gate)
        print(f"  核心数：{len(cores)}")
        for ci, core in enumerate(cores):
            res = build_core_parent(
                structs,
                core["members"],
                core["ref_idx"],
                core["align"],
                family_records,
                io_err,
            )
            parent_id = f"parent-{fam_id[:8]}-{ci}"
            res["parent_id"] = parent_id
            res["family_id"] = fam_id
            cores_all.append({k: v for k, v in res.items() if k != "rows"})
            parents_out.append(
                {
                    "parent_id": parent_id,
                    "family_id": fam_id,
                    "ref_idx": res["ref_idx"],
                    "lattice": res["lattice"],
                    "framework_sites": res["framework_sites"],
                    "na_sites": res["na_sites"],
                    "K": res["K"],
                }
            )
            for row in res["rows"]:
                registry.append({"parent_id": parent_id, "family_id": fam_id, **row})
            print(
                f"  {parent_id}: n={res['n_members']} K={res['K']} "
                f"align_med={res['align_median']:.3f} rmsd_med={res['rmsd_median']:.3f} "
                f"rmsd<=0.3={res['rmsd_le_0.3']:.2f} matcher={res['matcher_rate']:.2f}"
            )

    # 未进核心的成员：就近指派到最近母体（align<0.8 才计入 registry）
    assigned = {r["idx"] for r in registry}
    tail = sorted(member_idx - assigned)
    parent_refs = [
        (c["parent_id"], c["ref_idx"], np.array(c["lattice"])) for c in cores_all
    ]
    tail_assigned = 0
    for idx in tail:
        st = structs[idx]
        best = None
        for parent_id, ref_idx, Lp in parent_refs:
            ref_st = structs[ref_idx]
            ref_fw = np.array([s.frac_coords for s in ref_st if s.species_string in PTM_SPECIES])
            T = st.lattice.matrix @ np.linalg.inv(Lp)
            fw = np.array([s.frac_coords for s in st if s.species_string in PTM_SPECIES]) @ T
            _, score = na_sublattice.best_shift(ref_fw, fw, Lp)
            if best is None or score < best[0]:
                best = (score, parent_id, ref_idx, Lp)
        if best[0] < 0.8:
            score, parent_id, ref_idx, Lp = best
            ref_st = structs[ref_idx]
            ref_fw = np.array([s.frac_coords for s in ref_st if s.species_string in PTM_SPECIES])
            T = st.lattice.matrix @ np.linalg.inv(Lp)
            sh, _ = na_sublattice.best_shift(
                ref_fw,
                np.array([s.frac_coords for s in st if s.species_string in PTM_SPECIES]) @ T,
                Lp,
            )
            pdata = next(c for c in cores_all if c["parent_id"] == parent_id)
            na_sites = np.array(pdata["na_sites"])
            na = np.array([s.frac_coords for s in st if s.species_string == "Na"])
            na_frac = (na @ T + np.array(sh)) % 1.0 if len(na) else np.zeros((0, 3))
            occ = na_sublattice.occupancy_for_member(na_frac, na_sites, Lp)
            parent_tm = np.array(
                [s["frac"] for s in pdata["framework_sites"] if s["el"] in TM_SPECIES]
            )
            tm_frac = (
                np.array([s.frac_coords for s in st if s.species_string in TM_SPECIES]) @ T
                + np.array(sh)
            ) % 1.0
            tm_el = [s.species_string for s in st if s.species_string in TM_SPECIES]
            tm = na_sublattice.tm_decoration(tm_el, tm_frac, parent_tm, Lp)
            rec = family_records[idx]
            registry.append(
                {
                    "parent_id": parent_id,
                    "family_id": rec["family_id"],
                    "idx": idx,
                    "canonical_id": rec["canonical_id"],
                    "reduced_formula": rec["reduced_formula"],
                    "n_na_observed": rec["n_na"],
                    "n_na_occupancy": sum(occ),
                    "occupancy": occ,
                    "tm_decoration": tm,
                    "align_rmsd": score,
                    "core_member": False,
                    "energy_mlff": rec["energy_mlff"],
                    "relax_ok": not io_err.get(idx, False),
                }
            )
            tail_assigned += 1
    print(f"\n未归入核心：{len(tail)}，就近指派成功：{tail_assigned}，无法指派：{len(tail) - tail_assigned}")

    with open(os.path.join(out, "cores_v2.json"), "w") as f:
        json.dump({"gate": GATE, "n_pure": len(pure), "n_parents": len(cores_all), "cores": cores_all}, f, indent=1)
    with open(os.path.join(out, "parents_v2.json"), "w") as f:
        json.dump({"version": "v2", "parents": parents_out}, f, indent=1)
    with open(os.path.join(out, "registry_v2.jsonl"), "w") as f:
        for r in registry:
            f.write(json.dumps(r) + "\n")

    n_registry = len(registry)
    core_idx = {r["idx"] for r in registry if r.get("core_member", True)}
    coverage = len(core_idx) / len(member_idx)
    summary = {
        "n_pure_pyro": len(member_idx),
        "n_parents": len(cores_all),
        "n_core_members": len(core_idx),
        "core_coverage_fraction": float(coverage),
        "n_registry": n_registry,
        "registry_coverage_fraction": float(n_registry / len(member_idx)),
        "n_tail": len(tail),
        "tail_assigned": tail_assigned,
        "tail_unassigned": len(tail) - tail_assigned,
        "per_parent": [
            {
                "parent_id": c["parent_id"],
                "family": c["family_id"],
                "n": c["n_members"],
                "K": c["K"],
                "align_median": c["align_median"],
                "rmsd_median": c["rmsd_median"],
                "rmsd_le_0.3": c["rmsd_le_0.3"],
                "matcher_rate": c["matcher_rate"],
                "energy_grid": c["energy_grid"],
                "capacity": c["capacity"],
                "spearman_energy_vs_nna": c.get("spearman_energy_vs_nna"),
            }
            for c in cores_all
        ],
    }
    with open(os.path.join(out, "report_v2.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(
        f"核心覆盖率：{coverage:.1%}（{len(core_idx)}/{len(member_idx)}），"
        f"registry 覆盖率：{n_registry}/{len(member_idx)}，母体数：{len(cores_all)}"
    )
    print(f"完成：{out}/")


if __name__ == "__main__":
    main()

"""MVP 主流程：家族选择 -> 母体搜索 -> Na 亚晶格推断 -> 验证 -> registry + 报告。

用法:
    python -m nagen.decomposition.run_decomposition \\
        --decomp data/decomposition --canonical data/audit/canonical_v1.jsonl \\
        --records data/audit/records.csv --raw /mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl \\
        --family-id <el-aware family_id>
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from nagen.decomposition import na_sublattice, parent_search, validate
from nagen.decomposition._records import load_records

ALIGN_GATE = 0.35  # Å，框架一致核心门控
FW_SPECIES = ("P", "Ti", "V", "Fe", "Mn", "O")
PTM_SPECIES = ("P", "Ti", "V", "Fe", "Mn")
TM_SPECIES = ("Ti", "V", "Fe", "Mn")


def select_pure_pyro(records: list[dict], family_id: str | None) -> tuple[str, list[dict]]:
    """纯焦磷酸记录：q_label=Q1 且无 F/C/N。

    family_id 指定时按 el-aware 家族过滤；否则自动选最小框架图中
    成员最多的 el-aware 子家族（单一母体几何）。
    """
    pure = [
        r
        for r in records
        if r["q_label"] == "Q1"
        and "contains_F" not in r["flags"]
        and "contains_CN" not in r["flags"]
    ]
    if not pure:
        raise RuntimeError("未找到纯焦磷酸记录")
    if family_id:
        members = [r for r in pure if r["family_id"] == family_id]
        if not members:
            raise RuntimeError(f"family_id 不在纯焦磷酸集合内: {family_id}")
        return members[0]["framework_id"], members
    groups = collections.defaultdict(list)
    for r in pure:
        groups[r["framework_id"]].append(r)
    target = min(
        groups,
        key=lambda k: (min(r["n_fw_nodes"] for r in groups[k]), -len(groups[k])),
    )
    fam = collections.Counter(r["family_id"] for r in groups[target])
    chosen = fam.most_common(1)[0][0]
    return target, [r for r in groups[target] if r["family_id"] == chosen]


def load_io_error(raw_path: str, records_path: str) -> dict[int, bool]:
    """原始文件 material_id -> 是否有 I/O error；与 idx 关联。"""
    idx_of_mid = {}
    with open(records_path) as f:
        for row in csv.DictReader(f):
            idx_of_mid[row["material_id"]] = int(row["idx"])
    err = {}
    with open(raw_path) as f:
        for line in f:
            d = json.loads(line)
            mid = d["material_id"]
            if mid in idx_of_mid:
                err[idx_of_mid[mid]] = d.get("error_msg") is not None
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", default="data/decomposition")
    ap.add_argument("--canonical", default="data/audit/canonical_v1.jsonl")
    ap.add_argument("--records", default="data/audit/records.csv")
    ap.add_argument("--raw", default="/mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl")
    ap.add_argument("--family-id", default=None, help="el-aware family_id（默认自动选最大子家族）")
    ap.add_argument(
        "--records-family",
        default=None,
        help="record_family_v2.jsonl 路径（默认 <decomp>/record_family_v2.jsonl）",
    )
    args = ap.parse_args()
    os.makedirs(args.decomp, exist_ok=True)

    records = load_records(args.records_family or os.path.join(args.decomp, "record_family_v2.jsonl"))
    target_fw, members = select_pure_pyro(records, args.family_id)
    args.family_id = members[0]["family_id"]
    member_idx = {r["idx"] for r in members}
    print(
        f"纯焦磷酸 framework_id={target_fw[:12]} family_id={args.family_id[:12]} "
        f"members={len(members)}"
    )

    structs = {}
    with open(args.canonical) as f:
        for line in f:
            d = json.loads(line)
            if d["idx"] in member_idx:
                structs[d["idx"]] = Structure.from_dict(d["primitive_structure"])
    print(f"载入结构 {len(structs)}")
    io_err = load_io_error(args.raw, args.records)

    # Step 2: 母体搜索（参考成员 = 最高对称性最小体积成员）
    parent_cell = parent_search.select_parent_cell(
        [(idx, structs[idx]) for idx in sorted(member_idx)]
    )
    ref_idx = parent_cell["ref_idx"]
    ref_st = structs[ref_idx]
    Lp = np.array(parent_cell["lattice"])
    print(
        f"母体候选 idx={ref_idx} n_atoms={len(ref_st)} vol={ref_st.volume:.1f} "
        f"sym_ops={parent_cell['symmetry_ops']}"
    )

    ref_species = [s.species_string for s in ref_st if s.species_string in FW_SPECIES]
    ref_fw_all = np.array([s.frac_coords for s in ref_st if s.species_string in FW_SPECIES])
    ref_fw_ptm = np.array([s.frac_coords for s in ref_st if s.species_string in PTM_SPECIES])
    ref_na = np.array([s.frac_coords for s in ref_st if s.species_string == "Na"])

    def map_member(idx: int, kind: str):
        st = structs[idx]
        T = st.lattice.matrix @ np.linalg.inv(Lp)
        keep = FW_SPECIES if kind == "all" else PTM_SPECIES if kind == "ptm" else TM_SPECIES if kind == "tm" else ("Na",)
        arr = np.array([s.frac_coords for s in st if s.species_string in keep])
        return T, (arr @ T if len(arr) else arr)

    # Pass 1: 纯框架对齐（同族内松弛差异即 0.2~0.5 Å，以框架残差定核心）
    align1 = {}
    for idx in sorted(member_idx):
        T, fw = map_member(idx, "ptm")
        sh, score = na_sublattice.best_shift(ref_fw_ptm, fw, Lp)
        align1[idx] = {"basis_T": T.tolist(), "shift": sh, "align_rmsd": score}
    align = align1

    scores = np.array([a["align_rmsd"] for a in align.values()])
    print(
        f"align 分布: mean={scores.mean():.3f} p50={np.median(scores):.3f} "
        f"p90={np.percentile(scores, 90):.3f} <0.25:{np.mean(scores < 0.25):.2f} "
        f"<{ALIGN_GATE}:{np.mean(scores < ALIGN_GATE):.2f}"
    )
    ok_align = [idx for idx, a in align.items() if a["align_rmsd"] < ALIGN_GATE]
    print(f"框架一致核心（<{ALIGN_GATE}Å）= {len(ok_align)}/{len(member_idx)}")

    # Step 3b: 框架均值（理想化母体）
    mapped_members = []
    for idx in ok_align:
        T = np.array(align[idx]["basis_T"])
        shift = np.array(align[idx]["shift"])
        sp, fr = [], []
        for s in structs[idx]:
            if s.species_string in FW_SPECIES:
                sp.append(s.species_string)
                fr.append(s.frac_coords)
        mapped_members.append((sp, (np.array(fr) @ T + shift) % 1.0))
    framework_mean = parent_search.mean_framework(mapped_members, ref_species, ref_fw_all, Lp)
    print(
        f"框架均值：n_sites={framework_mean['n_sites']} n_used={framework_mean['n_used']} "
        f"n_failed={framework_mean['n_failed']}"
    )

    # Step 3c: Na 位点并集 + 对称性约简
    na_union = []
    for idx in ok_align:
        T = np.array(align[idx]["basis_T"])
        shift = np.array(align[idx]["shift"])
        _, na = map_member(idx, "na")
        if len(na):
            na_union.append((na + shift) % 1.0)
    na_union = np.vstack(na_union) if na_union else np.zeros((0, 3))
    na_sites = na_sublattice.cluster_sites(na_union, Lp)
    fw_st = Structure(
        Lp,
        [s["el"] for s in framework_mean["sites"]],
        [s["frac"] for s in framework_mean["sites"]],
    )
    ops = SpacegroupAnalyzer(fw_st, symprec=0.3).get_symmetry_operations()
    # Na 位点并集只做空间聚类：真实 Na 亚晶格不一定继承理想框架的
    # 全部对称操作，做框架对称性约简会生成伪镜像位点（16 -> 64）。
    print(f"Na 位点并集 K={len(na_sites)}（空间聚类，min_sep=1.2Å）")

    parent = parent_search.build_parent(parent_cell, framework_mean, na_sites.tolist())
    parent_search.save_parent(parent, os.path.join(args.decomp, "parents_v2.json"))

    # Step 4/5: 占据向量 + TM 装饰 + 重建验证（全体成员）
    parent_tm = np.array(
        [s["frac"] for s in parent["framework_sites"] if s["el"] in TM_SPECIES]
    )
    occ_path = os.path.join(args.decomp, "occupancy_v2.jsonl")
    ds_path = os.path.join(args.decomp, "dataset_v2.jsonl")
    rmsds, matcher_ok, n_mi, n_nonmi = [], [], 0, 0
    energy_rows = []
    with open(occ_path, "w") as fo, open(ds_path, "w") as fd:
        for idx in sorted(member_idx):
            st = structs[idx]
            rec = next(r for r in members if r["idx"] == idx)
            a = align[idx]
            basis_T = np.array(a["basis_T"])
            shift = np.array(a["shift"])
            align_ok = idx in ok_align
            _, na_raw = map_member(idx, "na")
            na_frac = (na_raw + shift) % 1.0 if len(na_raw) else np.zeros((0, 3))
            occ = na_sublattice.occupancy_for_member(na_frac, na_sites, Lp)
            _, tm_raw = map_member(idx, "tm")
            tm_frac = (tm_raw + shift) % 1.0
            tm_el = [s.species_string for s in st if s.species_string in TM_SPECIES]
            tm = na_sublattice.tm_decoration(tm_el, tm_frac, parent_tm, Lp)
            n_na = sum(occ)
            row = {
                "idx": idx,
                "canonical_id": rec["canonical_id"],
                "framework_id": rec["framework_id"],
                "family_id": rec["family_id"],
                "reduced_formula": rec["reduced_formula"],
                "n_na_observed": rec["n_na"],
                "n_na_occupancy": n_na,
                "occupancy": occ,
                "tm_decoration": tm,
                "basis_T": a["basis_T"],
                "align_rmsd": a["align_rmsd"],
                "align_ok": align_ok,
                "energy_mlff": rec["energy_mlff"],
                "relax_ok": not io_err.get(idx, True),
            }
            fo.write(json.dumps(row) + "\n")
            if n_na != rec["n_na"]:
                row["warning"] = "na_occupancy_mismatch"
            fd.write(json.dumps({"dataset": "v2-pointer", **row, "structure_pointer": f"canonical_v1.jsonl#{idx}"}) + "\n")

            inv_T = np.linalg.inv(basis_T)
            rebuilt = validate.rebuild_structure(
                [{**s, "frac": (np.array(s["frac"]) - shift) @ inv_T % 1.0} for s in parent["framework_sites"]],
                (na_sites - shift) @ inv_T % 1.0,
                occ,
                st.lattice.matrix,
                None,
            )
            if align_ok:
                rmsd = validate.site_rmsd(rebuilt, st)
                rmsds.append(rmsd["rmsd"])
                n_mi += 1
            else:
                n_nonmi += 1
            matcher_ok.append(validate.matcher_pass(rebuilt, st))
            if rec["energy_mlff"] is not None and not io_err.get(idx, True):
                energy_rows.append((n_na, rec["energy_mlff"], "/".join(tm)))

    # 验收门
    rmsds = np.array(rmsds)
    cov = {
        "n_core": n_mi,
        "core_fraction": float(n_mi / len(member_idx)),
        "rmsd_le_0.3": float((rmsds <= 0.3).mean()) if len(rmsds) else None,
        "rmsd_le_0.5": float((rmsds <= 0.5).mean()) if len(rmsds) else None,
        "rmsd_median": float(np.median(rmsds)) if len(rmsds) else None,
        "rmsd_p90": float(np.percentile(rmsds, 90)) if len(rmsds) else None,
        "matcher_pass_rate": float(np.mean(matcher_ok)) if matcher_ok else None,
        "n_tail": n_nonmi,
    }
    spe = {}
    if energy_rows:
        allx = np.array([r[0] for r in energy_rows], dtype=float)
        ally = np.array([r[1] for r in energy_rows])
        spe["overall"] = validate.spearman(allx, ally)
        for deco in sorted({r[2] for r in energy_rows}):
            sub = [r for r in energy_rows if r[2] == deco]
            if len(sub) >= 5:
                spe[deco] = validate.spearman(
                    np.array([r[0] for r in sub], dtype=float),
                    np.array([r[1] for r in sub]),
                )
    print("Gates:", json.dumps({**cov, "spearman": spe}, indent=1))

    report = {
        "framework_id": target_fw,
        "family_id": args.family_id,
        "n_members": len(members),
        "n_formulas": len({r["reduced_formula"] for r in members}),
        "align_gate_angstrom": ALIGN_GATE,
        "parent": {
            "ref_idx": ref_idx,
            "n_framework_sites": len(parent["framework_sites"]),
            "n_na_sites": len(na_sites),
            "spacegroup": parent["spacegroup_number"],
            "sym_ops": parent["symmetry_ops"],
            "wyckoffs": parent["wyckoffs"],
        },
        "gates": cov,
        "spearman_energy_vs_nna": spe,
    }
    with open(os.path.join(args.decomp, "decomposition_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"完成：{occ_path} / {ds_path} / decomposition_report.json")


if __name__ == "__main__":
    main()

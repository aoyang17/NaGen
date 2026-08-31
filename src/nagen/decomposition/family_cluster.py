"""Step 1: 全量 P-O-TM 骨架图聚类，产出 families_v2。

对每条记录的 primitive 结构构建 P-O-TM 连通图（角共享边 + P-O-P 桥），
分别计算元素相关（family_id）与元素无关（framework_id）的 WL hash。

用法:
    python -m nagen.decomposition.family_cluster \\
        --canonical data/audit/canonical_v1.jsonl \\
        --records data/audit/records.csv \\
        --out data/decomposition --workers 16
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing as mp
import os

import networkx as nx
from pymatgen.core import Structure

FRAMEWORK_TM = ("Ti", "V", "Fe", "Mn")
P_CUT = 2.15
TM_CUT = 2.55
WL_ITERATIONS = 6


def build_framework_graph(st: Structure, p_cut: float = P_CUT, tm_cut: float = TM_CUT) -> nx.Graph:
    """P/TM 为节点，共享 O（角共享）为边；双桥氧连接的两个 P 之间加 P-P 边。"""
    G = nx.Graph()
    p_idx = [i for i, s in enumerate(st) if s.species_string == "P"]
    tm_idx = [i for i, s in enumerate(st) if s.species_string in FRAMEWORK_TM]
    for i in p_idx + tm_idx:
        G.add_node(i, el=st[i].species_string)
    o_idx = [i for i, s in enumerate(st) if s.species_string == "O"]
    for oi in o_idx:
        pn = [j for j in p_idx if st.get_distance(oi, j) < p_cut]
        tn = [j for j in tm_idx if st.get_distance(oi, j) < tm_cut]
        for a in pn:
            for b in tn:
                G.add_edge(a, b)
        if len(pn) == 2:
            G.add_edge(*pn)
    return G


def graph_hash(G: nx.Graph, include_elements: bool = True) -> str:
    return nx.weisfeiler_lehman_graph_hash(
        G,
        node_attr="el" if include_elements else None,
        iterations=WL_ITERATIONS,
    )


def _worker(args):
    idx, sdict = args
    st = Structure.from_dict(sdict)
    G = build_framework_graph(st)
    n_na = sum(1 for s in st if s.species_string == "Na")
    return (
        idx,
        graph_hash(G, True),
        graph_hash(G, False),
        G.number_of_nodes(),
        G.number_of_edges(),
        n_na,
    )


def cluster(canonical_path: str, records_path: str, out_dir: str, workers: int = 16) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    meta = {}
    with open(records_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["idx"])
            meta[idx] = {
                "canonical_id": row["canonical_id"],
                "material_id": row["material_id"],
                "reduced_formula": row["reduced_formula"],
                "flags": row["flags"],
                "q_label": row["q_label"],
                "energy_mlff": (
                    float(row["optimized_energy_per_atom_value"])
                    if row["optimized_energy_per_atom_present"] == "1"
                    else None
                ),
            }

    family = collections.defaultdict(list)
    framework = collections.defaultdict(list)
    rec_path = os.path.join(out_dir, "record_family_v2.jsonl")

    pool = mp.Pool(workers)
    with open(canonical_path) as f, open(rec_path, "w") as out:
        it = pool.imap(_worker, ((idx, d) for idx, d in _stream_records(f, meta)), chunksize=128)
        for idx, el_hash, ag_hash, n_nodes, n_edges, n_na in it:
            m = meta[idx]
            rec = {
                "idx": idx,
                "canonical_id": m["canonical_id"],
                "family_id": el_hash,
                "framework_id": ag_hash,
                "n_fw_nodes": n_nodes,
                "n_fw_edges": n_edges,
                "n_na": n_na,
                "reduced_formula": m["reduced_formula"],
                "flags": m["flags"],
                "q_label": m["q_label"],
                "energy_mlff": m["energy_mlff"],
            }
            out.write(json.dumps(rec) + "\n")
            family[el_hash].append(idx)
            framework[ag_hash].append(idx)
    pool.close()
    pool.join()

    def summarize(groups):
        out = {}
        for key, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            formulas = collections.Counter(meta[i]["reduced_formula"] for i in idxs)
            flags = collections.Counter(meta[i]["flags"] for i in idxs)
            q = collections.Counter(meta[i]["q_label"] for i in idxs)
            out[key] = {
                "n_records": len(idxs),
                "n_formulas": len(formulas),
                "top_formulas": dict(formulas.most_common(10)),
                "flags": dict(flags.most_common()),
                "q_labels": dict(q.most_common()),
                "members": idxs,
            }
        return out

    fam_summary = summarize(family)
    fw_summary = summarize(framework)
    # 将 members 从 family 摘要中剥离为引用文件，保持 JSON 轻量
    fam_light = {}
    for k, v in fam_summary.items():
        fam_light[k] = {kk: vv for kk, vv in v.items() if kk != "members"}
    fw_light = {}
    for k, v in fw_summary.items():
        fw_light[k] = {kk: vv for kk, vv in v.items() if kk != "members"}

    with open(os.path.join(out_dir, "families_v2.json"), "w") as f:
        json.dump(
            {
                "version": "v2",
                "n_records": len(meta),
                "n_family": len(fam_summary),
                "n_framework": len(fw_summary),
                "params": {"p_cut": P_CUT, "tm_cut": TM_CUT, "wl_iterations": WL_ITERATIONS},
                "families": fam_light,
                "frameworks": fw_light,
                "member_file": "record_family_v2.jsonl",
            },
            f,
            indent=1,
        )
    print(f"records={len(meta)} family={len(fam_summary)} framework={len(fw_summary)}")
    print("top frameworks:")
    for k, v in sorted(fw_summary.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"  {v['n_records']:6d} recs  {v['n_formulas']:4d} formulas  nodes={v['members'] and None or ''}  {k[:10]}")
    return fw_summary


def _stream_records(f, meta):
    for line in f:
        d = json.loads(line)
        idx = d["idx"]
        if idx in meta:
            yield idx, d["primitive_structure"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default="data/audit/canonical_v1.jsonl")
    ap.add_argument("--records", default="data/audit/records.csv")
    ap.add_argument("--out", default="data/decomposition")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    cluster(args.canonical, args.records, args.out, args.workers)


if __name__ == "__main__":
    main()

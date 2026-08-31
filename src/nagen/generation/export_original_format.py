"""把 nova_200 候选导出为与原数据 parent_RDF_opt.jsonl 相同 schema 的 JSONL。

原 schema（首条实测）：
material_id, formula, structure(ASR dict), energy_above_hull, energy_per_atom,
num_NaO3..num_PO12(30 个，原数据全 null), optimized_energy_per_atom

说明：optimized_energy_per_atom 填入代理能量（RDF-MLP，非 MLFF/DFT），
并附加 generation 元数据字段与 energy_origin 标记。
"""

from __future__ import annotations

import argparse
import json
import os

from pymatgen.core import Structure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="data/generation/nova_200.jsonl")
    ap.add_argument("--cif-dir", default="data/generation/nova_200_cif")
    ap.add_argument("--out", default="data/generation/nova_200_original_format.jsonl")
    args = ap.parse_args()

    cands = [json.loads(l) for l in open(args.cands)]
    num_keys = [f"num_{a}O{b}" for b in range(3, 13) for a in ("Na", "TM", "P")]
    with open(args.out, "w") as f:
        for i, c in enumerate(cands):
            st = Structure.from_file(os.path.join(args.cif_dir, f"nova_{i:03d}.cif"))
            st.properties = {"charge": 0, "spin": 0}
            formula = st.composition.reduced_formula
            rec = {
                "material_id": (
                    f"NaGen_NOVA_{i:04d}_{formula.replace(' ', '')}_"
                    f"{''.join(c['decoration'])}_{c['n_na']}Na"
                ),
                "formula": formula,
                "structure": st.as_dict(),
                "energy_above_hull": None,
                "energy_per_atom": None,
            }
            for k in num_keys:
                rec[k] = None
            rec["optimized_energy_per_atom"] = c["energy_surr_eV_atom"]
            # 附加元数据（原数据无，用于谱系与诚实标注）
            rec["energy_origin"] = "surrogate_rdf_mlp_not_mlff_dft"
            rec["generation"] = {
                "parent_id": c["parent_id"],
                "decoration": c["decoration"],
                "occupancy": c["occupancy"],
                "supercell2": c.get("supercell2", False),
                "L_change_pct": c["L_change"],
                "capacity_mAh_g": c["capacity_mAh_g"],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"完成：{args.out}（{len(cands)} 条，原 schema 对齐）")


if __name__ == "__main__":
    main()

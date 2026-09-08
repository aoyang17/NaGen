#!/usr/bin/env python3
"""Export the first completed relaxation records as explicitly provisional CIFs."""

import argparse
import hashlib
import json
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relax-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=7)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    relax_dir = Path(args.relax_dir)
    out = Path(args.out)
    cif_dir = out / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(line) for line in (relax_dir / "pairs.jsonl").read_text().splitlines() if line.strip()]
    completed = [row for row in pairs if row.get("status") == "completed"][: args.count]
    structures = {
        row["material_id"]: row
        for row in map(json.loads, (relax_dir / "relaxed.jsonl").read_text().splitlines())
    }
    if len(completed) < args.count:
        raise RuntimeError(f"Only {len(completed)} completed records; requested {args.count}")

    records = []
    for order, pair in enumerate(completed, 1):
        material_id = pair["id"]
        raw = structures[material_id]
        structure = Structure(Lattice(raw["L_matrix"]), raw["A"], raw["X_frac"])
        path = cif_dir / f"{order:02d}_{material_id}_PROVISIONAL.cif"
        CifWriter(structure, symprec=None, significant_figures=10).write_file(path)
        reread = Structure.from_file(path)
        if len(reread) != len(structure) or reread.composition != structure.composition:
            raise RuntimeError(f"CIF round-trip failed for {material_id}")
        after_gate = pair.get("gates", {}).get("after", {})
        records.append({
            "order": order,
            "id": material_id,
            "file": path.name,
            "formula": structure.composition.reduced_formula,
            "optimizer_converged": pair.get("converged"),
            "steps": pair.get("steps"),
            "energy_eV_atom_CHGNet": pair.get("after", {}).get("energy_eV_atom"),
            "force_max_eV_A_CHGNet": pair.get("after", {}).get("force_max_eV_A"),
            "hard_gate_passed_current": after_gate.get("passed"),
            "hard_gate_checks_current": after_gate.get("checks"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "roundtrip_atoms": len(reread),
        })

    manifest = {
        "provisional": True,
        "job_id": args.job_id,
        "selection": f"first {args.count} completed records in pairs.jsonl order",
        "warning": "Running-job snapshot; not final-ranked or diversity-selected; records may fail hard gates.",
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "README.md").write_text(
        f"# NaGen job {args.job_id}: provisional first {args.count}\n\n"
        "运行中快照，按 pairs.jsonl 完成顺序导出。它们不是最终候选：尚未完成全批次 "
        "Robust Ranking 与 Diversity Selection，且可能未通过 Hard Gates。\n\n"
        "组成：Na6Fe6P8O32；生成变量包含 Na/Fe/P/O 坐标与晶格。详情见 manifest.json。\n"
    )
    print(json.dumps({"exported": len(records), "out": str(out)}))


if __name__ == "__main__":
    main()

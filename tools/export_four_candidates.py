#!/usr/bin/env python3
"""Export and visualize the four provisionally hull-feasible candidates.

The public-facing export intentionally uses neutral candidate identifiers and
the same core JSON representation as the supplied reference dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors
from pymatgen.core import Structure

from nagen.inverse.constraints import bond_cutoff, minimum_distance, pbc_distance_matrix
from nagen.inverse.spec import DEFAULT_SPEC


SOURCE = Path(
    "/mnt/data2/aobo/NaGen/inverse_v1/candidates/"
    "final_v3_verified/provisional_hull_pass.jsonl"
)

ELEMENT_COLORS = {
    "Na": "#9B8CFF",
    "P": "#F28C28",
    "O": "#E53935",
    "Mn": "#9C7AC7",
    "Ti": "#A7A9AC",
    "V": "#55A868",
    "N": "#3768B0",
    "F": "#8BD646",
    "C": "#333333",
    "Fe": "#D47A2C",
}


def load_records(source: Path) -> list[dict]:
    with source.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def oxidation_assignment(elements: list[str]) -> dict[str, Counter]:
    """Return one exact site-charge assignment, grouped by element."""
    states: dict[int, list[tuple[str, int]]] = {0: []}
    for element in elements:
        next_states: dict[int, list[tuple[str, int]]] = {}
        for total, assigned in states.items():
            for charge in DEFAULT_SPEC.allowed_charges[element]:
                candidate = total + charge
                next_states.setdefault(candidate, assigned + [(element, charge)])
        states = next_states
    assignment = states[0]
    grouped: dict[str, Counter] = {}
    for element, charge in assignment:
        grouped.setdefault(element, Counter())[charge] += 1
    return grouped


def assignment_text(grouped: dict[str, Counter]) -> str:
    chunks = []
    for element in DEFAULT_SPEC.elements:
        if element not in grouped:
            continue
        states = ",".join(
            f"{charge:+d}x{count}" for charge, count in sorted(grouped[element].items())
        )
        chunks.append(f"{element}({states})")
    return "; ".join(chunks)


def structure_metrics(structure: Structure) -> dict:
    elements = [site.specie.symbol for site in structure]
    frac = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    distances = pbc_distance_matrix(frac, lattice)

    limiting = None
    distance_margin = float("inf")
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            threshold = minimum_distance(elements[i], elements[j])
            margin = float(distances[i, j] - threshold)
            if margin < distance_margin:
                distance_margin = margin
                limiting = (elements[i], elements[j], float(distances[i, j]), threshold)

    p_counts: list[int] = []
    tm_counts: list[int] = []
    cutoff_gap = float("inf")
    all_oxygen = True
    for i, center in enumerate(elements):
        if center != "P" and center not in DEFAULT_SPEC.transition_metals:
            continue
        neighbors = []
        for j, other in enumerate(elements):
            if i == j:
                continue
            cutoff = bond_cutoff(center, other)
            cutoff_gap = min(cutoff_gap, abs(float(distances[i, j]) - cutoff))
            if distances[i, j] <= cutoff:
                neighbors.append(other)
        all_oxygen &= all(other == "O" for other in neighbors)
        (p_counts if center == "P" else tm_counts).append(len(neighbors))

    return {
        "frac_min": float(frac.min()),
        "frac_max": float(frac.max()),
        "det_lattice_A3": float(np.linalg.det(lattice)),
        "minimum_pair_clearance_A": distance_margin,
        "limiting_pair": f"{limiting[0]}-{limiting[1]}",
        "limiting_pair_distance_A": limiting[2],
        "limiting_pair_threshold_A": limiting[3],
        "coordination_cutoff_gap_A": cutoff_gap,
        "p_coordination_profile": dict(sorted(Counter(p_counts).items())),
        "tm_coordination_profile": dict(sorted(Counter(tm_counts).items())),
        "all_p_tm_neighbors_oxygen": bool(all_oxygen),
    }


def cell_edges(lattice: np.ndarray):
    origin = np.zeros(3)
    a, b, c = lattice
    vertices = [origin, a, b, c, a + b, a + c, b + c, a + b + c]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6),
             (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    return vertices, pairs


def render_structure(structure: Structure, title: str, output: Path, ax=None) -> None:
    own_figure = ax is None
    if own_figure:
        fig = plt.figure(figsize=(8, 7), dpi=180)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    positions = np.asarray(structure.cart_coords)
    lattice = np.asarray(structure.lattice.matrix)
    elements = [site.specie.symbol for site in structure]

    # Draw the same P-O/TM-O bond definition used by the constraint audit.
    seen_bonds = set()
    for i, site in enumerate(structure):
        center = site.specie.symbol
        if center != "P" and center not in DEFAULT_SPEC.transition_metals:
            continue
        cutoff = DEFAULT_SPEC.p_o_bond_cutoff if center == "P" else DEFAULT_SPEC.tm_o_bond_cutoff
        for neighbor in structure.get_neighbors(site, cutoff):
            if neighbor.specie.symbol != "O":
                continue
            key = (i, neighbor.index, tuple(int(v) for v in neighbor.image))
            if key in seen_bonds:
                continue
            seen_bonds.add(key)
            p0, p1 = positions[i], np.asarray(neighbor.coords)
            ax.plot(*zip(p0, p1), color="#7A7A7A", linewidth=1.1, alpha=0.6, zorder=1)

    # ASE's open element data provide conventional Jmol colors/radii; use a
    # fixed palette where available so all four panels remain consistent.
    for element in sorted(set(elements)):
        indices = [i for i, value in enumerate(elements) if value == element]
        xyz = positions[indices]
        number = atomic_numbers[element]
        color = ELEMENT_COLORS.get(element, jmol_colors[number])
        size = 115.0 * max(0.55, covalent_radii[number]) ** 1.35
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size, c=[color],
                   edgecolors="black", linewidths=0.35, depthshade=True,
                   label=element, zorder=3)

    vertices, pairs = cell_edges(lattice)
    for i, j in pairs:
        ax.plot(*zip(vertices[i], vertices[j]), color="black", linewidth=0.8, alpha=0.75)

    all_points = np.vstack([positions, np.asarray(vertices)])
    mins, maxs = all_points.min(axis=0), all_points.max(axis=0)
    center = (mins + maxs) / 2
    span = max(float((maxs - mins).max()), 1.0) * 0.58
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=21, azim=36)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=2)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    if own_figure:
        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.out / "images"
    cif_dir = args.out / "cif"
    image_dir.mkdir(exist_ok=True)
    cif_dir.mkdir(exist_ok=True)

    records = load_records(args.source)
    public_records = []
    summary_rows = []
    check_rows = []
    fig = plt.figure(figsize=(14, 12), dpi=180)

    for index, record in enumerate(records, 1):
        public_id = f"NaGen-Candidate-{index:03d}"
        structure = Structure.from_dict(record["structure"])
        metrics = structure_metrics(structure)
        feasibility = record["provisional_data_relative_feasibility"]
        values = feasibility["values"]
        thermo = record["thermodynamic_stability_score"]
        novelty = record["novelty_validation"]
        assignment = oxidation_assignment([site.specie.symbol for site in structure])
        cell_formula = structure.composition.formula.replace(" ", "")
        reduced_formula = structure.composition.reduced_formula
        capacity_margin = values["capacity_mAh_g"] - DEFAULT_SPEC.capacity_min_mAh_g
        hull_margin = DEFAULT_SPEC.hull_max_eV_atom - values["delta_e_hull_eV_atom"]

        public = {
            "material_id": public_id,
            "formula": reduced_formula,
            "structure": structure.as_dict(),
            "energy_above_hull": None,
            "energy_per_atom": None,
            **{f"num_{role}O{cn}": None for cn in range(3, 13) for role in ("Na", "TM", "P")},
            "optimized_energy_per_atom": thermo["relaxed_energy_eV_atom"],
            "candidate_metadata": {
                "cell_formula": cell_formula,
                "reduced_formula": reduced_formula,
                "energy_model": "CHGNet-0.3.0",
                "data_relative_energy_above_hull_eV_atom": values["delta_e_hull_eV_atom"],
                "strict_complete_dft_hull_status": "not_evaluated",
                "capacity_mAh_g": values["capacity_mAh_g"],
                "extractable_Na_n_e": values["n_e"],
                "novelty_score": record["novelty_score"],
                "composition_novel": novelty["composition_in_reference"] is False,
                "structure_novel": novelty["structure_novel"],
            },
        }
        public_records.append(public)

        p_profile = ", ".join(f"CN{k}x{v}" for k, v in metrics["p_coordination_profile"].items())
        tm_profile = ", ".join(f"CN{k}x{v}" for k, v in metrics["tm_coordination_profile"].items())
        summary_rows.append({
            "material_id": public_id,
            "cell_formula": cell_formula,
            "reduced_formula": reduced_formula,
            "N": len(structure),
            "elements": ",".join(sorted({site.specie.symbol for site in structure})),
            "capacity_mAh_g": values["capacity_mAh_g"],
            "capacity_margin_mAh_g": capacity_margin,
            "n_e": values["n_e"],
            "data_relative_delta_e_hull_eV_atom": values["delta_e_hull_eV_atom"],
            "provisional_hull_margin_meV_atom": 1000 * hull_margin,
            "novelty_score": record["novelty_score"],
            "composition_novel": True,
            "structure_novel": novelty["structure_novel"],
            "minimum_pair_clearance_A": metrics["minimum_pair_clearance_A"],
            "coordination_cutoff_gap_A": metrics["coordination_cutoff_gap_A"],
            "P_coordination": p_profile,
            "TM_coordination": tm_profile,
            "final_max_force_eV_A": thermo["final_max_force_eV_A"],
        })

        checks = feasibility["checks"]
        common = {"material_id": public_id, "formula": reduced_formula}
        rows = [
            ("N positive integer", checks["positive_atom_count"], str(len(structure)), "> 0", "exact"),
            ("A allowed elements", checks["element_domain"], ",".join(sorted(set(structure.symbol_set))), "subset of E", "exact"),
            ("X fractional domain", checks["fractional_coordinate_domain"], f"[{metrics['frac_min']:.6f}, {metrics['frac_max']:.6f}]", "[0,1)", "float64"),
            ("L positive determinant", checks["positive_lattice_determinant"], f"{metrics['det_lattice_A3']:.6f} A^3", "> 0", "float64"),
            ("charge neutrality", checks["charge_neutrality"], "integer residual 0; " + assignment_text(assignment), "sum q = 0", "exact integer DP"),
            ("PBC minimum distances", checks["minimum_pbc_distances"], f"min clearance {metrics['minimum_pair_clearance_A']:.6f} A ({metrics['limiting_pair']} {metrics['limiting_pair_distance_A']:.6f} vs {metrics['limiting_pair_threshold_A']:.6f})", ">= pair threshold", "27-image float64"),
            ("specific capacity", checks["specific_capacity"], f"{values['capacity_mAh_g']:.6f}; margin {capacity_margin:.6f} mAh/g; n_e={values['n_e']}", ">= 130 mAh/g", "float64 + exact n_e"),
            ("P/TM bonded neighbors O", checks["p_all_bonded_neighbors_oxygen"] and checks["tm_all_bonded_neighbors_oxygen"], f"all O; nearest cutoff boundary {metrics['coordination_cutoff_gap_A']:.6f} A", "all bonded neighbors O", "fixed cutoffs"),
            ("P coordination", checks["p_coordination_eq_4"], p_profile, "CN=4", "integer count"),
            ("TM coordination", checks["tm_coordination_in_4_5_6"], tm_profile, "CN in {4,5,6}", "integer count"),
            ("hull stability (provisional)", checks["thermodynamic_stability"], f"{values['delta_e_hull_eV_atom']:.9f} eV/atom; margin {1000*hull_margin:.3f} meV/atom", "<= 0.150 eV/atom", "CHGNet/data-relative"),
            ("hull stability (strict DFT)", None, "not evaluated", "<= 0.150 eV/atom", "complete consistent DFT hull required"),
        ]
        for constraint, passed, measured, requirement, precision in rows:
            check_rows.append({**common, "constraint": constraint, "pass": passed,
                               "measured": measured, "requirement": requirement,
                               "audit_precision": precision})

        structure.to(filename=str(cif_dir / f"{public_id}.cif"))
        render_structure(structure, f"{public_id}  {reduced_formula}", image_dir / f"{public_id}.png")
        ax = fig.add_subplot(2, 2, index, projection="3d")
        render_structure(structure, f"{public_id}\n{reduced_formula}", image_dir / "unused.png", ax=ax)

    fig.suptitle("Four generated sodium cathode candidates", fontsize=16)
    fig.tight_layout()
    fig.savefig(image_dir / "four_candidates_montage.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with (args.out / "candidates_original_schema.jsonl").open("w") as handle:
        for record in public_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    for filename, rows in (("candidate_summary.csv", summary_rows), ("constraint_audit.csv", check_rows)):
        with (args.out / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    readme = """# Four candidate structures

This package uses neutral public identifiers and mirrors the reference JSONL's
core representation (`material_id`, `formula`, pymatgen `structure`, and energy
fields). `energy_above_hull` remains `null` because a complete consistent DFT
hull has not been evaluated. The provisional CHGNet/data-relative value is
stored explicitly in `candidate_metadata`.

Generation input: `{source}`

Images were rendered with the open-source pymatgen, ASE, NumPy, and Matplotlib
packages. Lines show P-O and TM-O bonds under the same fixed cutoffs used in the
constraint audit; the black wireframe is the periodic unit cell.

Feasibility meaning:

- Provisional result: all four pass 11/11 checks when the supplied-reference
  CHGNet data-relative hull is used.
- Strict result: all four pass 10/10 non-hull checks; the complete consistent
  DFT hull constraint is unevaluated, not proven.
- Novelty is an optimization objective rather than a hard pass/fail constraint.
  All four compositions are absent from the 17,774-row reference set, and all
  four fail to match any reference structure under pymatgen StructureMatcher.
""".format(source=args.source)
    (args.out / "README.md").write_text(readme)

    lines = [
        "# 四个候选结构及约束核查",
        "",
        "## 组成与关键可行性余量",
        "",
        "| 编号 | 晶胞组成（最简式） | N | 元素 | n_e | 容量 / 余量 (mAh g⁻¹) | 最小距离余量 (Å) | P 配位 | TM 配位 | 数据相对 ΔE_hull / 余量 (meV atom⁻¹) | 新颖性 |",
        "|---|---|---:|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in summary_rows:
        formula = row["cell_formula"]
        if row["cell_formula"] != row["reduced_formula"]:
            formula += f"（{row['reduced_formula']}）"
        lines.append(
            f"| {row['material_id']} | {formula} | {row['N']} | {row['elements']} | "
            f"{row['n_e']} | {row['capacity_mAh_g']:.3f} / +{row['capacity_margin_mAh_g']:.3f} | "
            f"+{row['minimum_pair_clearance_A']:.4f} | {row['P_coordination']} | "
            f"{row['TM_coordination']} | {1000*row['data_relative_delta_e_hull_eV_atom']:.3f} / "
            f"+{row['provisional_hull_margin_meV_atom']:.3f} | {row['novelty_score']:.3f} |"
        )
    lines += [
        "",
        "## 对优化问题表的结论",
        "",
        "| 约束 | 001 | 002 | 003 | 004 | 核查精度/边界 |",
        "|---|---|---|---|---|---|",
        "| N 为正整数 | 通过 | 通过 | 通过 | 通过 | 整数精确 |",
        "| A 属于允许元素集 | 通过 | 通过 | 通过 | 通过 | 集合精确 |",
        "| X 属于 [0,1)³ | 通过 | 通过 | 通过 | 通过 | float64 |",
        "| det(L)>0 | 通过 | 通过 | 通过 | 通过 | float64 |",
        "| 电中性 | 通过 | 通过 | 通过 | 通过 | 整数价态动态规划，电荷残差严格为 0 |",
        "| PBC 元素对最小距离 | 通过 | 通过 | 通过 | 通过 | 27 周期像、float64 |",
        "| Q_th≥130 mAh g⁻¹ | 通过 | 通过 | 通过 | 通过 | n_e 整数精确、容量 float64 |",
        "| P/TM 全部成键邻居为 O | 通过 | 通过 | 通过 | 通过 | 固定元素对截断 |",
        "| P 配位数=4 | 通过 | 通过 | 通过 | 通过 | 整数计数 |",
        "| TM 配位数∈{4,5,6} | 通过 | 通过 | 通过 | 通过 | 整数计数 |",
        "| ΔE_hull≤0.150 eV atom⁻¹（当前数据相对口径） | 通过 | 通过 | 通过 | 通过 | CHGNet 候选松弛能量对 17,774 个给定参考结构单点能量 |",
        "| ΔE_hull≤0.150 eV atom⁻¹（完整一致 DFT 相图） | 未评估 | 未评估 | 未评估 | 未评估 | 需要完整竞争相与统一 DFT 口径 |",
        "",
        "因此，按当前数据相对 CHGNet hull 口径，四个候选均为 11/11 通过；按严格口径，四个候选均为 10/10 个已评估非 hull 检查通过，完整 DFT hull 仍未评估。新颖性是目标函数而非硬约束；四个组成均未出现在参考数据中，且四个最终结构均未被 pymatgen StructureMatcher 匹配到参考结构。",
        "",
        "## 精度与稳健性提醒",
        "",
        "“通过”是针对表中明确的离散/阈值定义。002 和 003 距离配位截断边界分别仅约 0.0011 Å 和 0.0062 Å，配位判定对微小坐标扰动较敏感；四个数据相对 hull 余量也都只有 1.98–6.50 meV atom⁻¹。因此不能把当前结果提升为严格热力学稳定性证明。完整逐项数值见 `constraint_audit.csv`。",
        "",
        "结构图由开源 pymatgen、ASE、NumPy 与 Matplotlib 生成；灰线按约束核查相同的 P–O/TM–O 截断绘制，黑线为周期晶胞。",
    ]
    (args.out / "CONSTRAINT_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""Independent default-UMA verification of the actual exported campaign CIFs."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
import networkx as nx
from networkx.algorithms.isomorphism import categorical_node_match
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nagen.inverse._io import sha256_file as sha256
from nagen.selection.pipeline import Crystal, Conditioning, hard_gates, robust_rank, diversity_select
from nagen.selection.uma import UMAEvaluator
from nagen.selection.diversity import distances
from run_uma_campaign import Collector, inputs, load_hull, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    out = Path(args.run).resolve()
    config = json.loads((out / "configuration.json").read_text())
    for name in ("flow", "profile", "condition", "references"):
        assert sha256(inputs()[name]) == config["hashes"][name], f"changed input: {name}"
    for split, digest in config["known_hashes"].items():
        assert sha256(inputs()["known"] / f"{split}.jsonl") == digest
    torch.set_num_threads(2)
    evaluator = UMAEvaluator(str(inputs()["uma"]), "cuda", "omat")
    assert evaluator.provenance["state_sha256"] == config["hashes"]["uma"]
    hull = load_hull(evaluator.provenance["state_sha256"])
    collector = Collector(out)
    condition = Conditioning(**json.loads(inputs()["condition"].read_text()))
    profile = json.loads(inputs()["profile"].read_text())
    rows, frameworks, features = [], [], []
    for entry in collector.selected:
        path = out / "selected" / entry["file"]
        loaded = Structure.from_file(path)
        c = Crystal(entry["id"], [str(s.specie) for s in loaded], np.asarray(loaded.frac_coords), np.asarray(loaded.lattice.matrix))
        label = evaluator.evaluate(c.elements, c.frac, c.lattice)
        gates = hard_gates(c, condition, profile, forces=label["forces_eV_A"], force_threshold=.03, motif_backend="native")
        hr = hull.evaluate(c.elements, label["energy_eV_atom"])
        eh = hr.get("e_above_reference_hull_eV_atom")
        stripped = collector.strip(c)
        framework = collector.structure(stripped)
        feature = collector.descriptor(stripped)
        graph = collector.graph(stripped)
        known_match = any(collector.matcher.fit(framework, s) for s in collector.ref_structures)
        topology_match = any(nx.is_isomorphic(graph, g, node_match=categorical_node_match("element", "")) for g in collector.ref_graphs)
        nearest = float(distances(feature[None], collector.ref_features).min())
        frameworks.append(framework)
        features.append(feature)
        row = {"id": c.id, "file": str(path), "sha256": sha256(path), "labels": label,
               "hard_gates": gates, "reference_hull": hr,
               "framework_novelty": {"known_structure_match": known_match, "known_topology_match": topology_match,
                                     "nearest_known_distance": nearest, "known_reference_count": len(collector.refs)},
               "passed": gates["passed"] and eh is not None and eh <= .15 and sha256(path) == entry["sha256"]
                         and not known_match and not topology_match and nearest >= 1e-4}
        rows.append(row)
        print(json.dumps({"id": c.id, "passed": row["passed"], "force": label["force_max_eV_A"], "hull": eh}), flush=True)
    pairs = [(collector.selected[i]["id"], collector.selected[j]["id"])
             for i in range(len(frameworks)) for j in range(i)
             if collector.matcher.fit(frameworks[i], frameworks[j])]
    d = distances(np.stack(features), np.stack(features)) if features else np.empty((0, 0))
    minimum = float(d[np.triu_indices(len(d), 1)].min()) if len(d) > 1 else None
    passed = len(rows) >= config["target"] and all(r["passed"] for r in rows) and not pairs and (minimum is None or minimum >= 1e-4)
    metrics = ["reference_hull_eV_atom"] + [f"{e}_{m}" for e in ("P", "Fe") for m in ("off_center", "bond_distortion", "angle_distortion", "coordination_rarity")]
    ranking_rows = [{"id": r["id"], "passed": r["passed"], "group": "Na6Fe6P8O32",
                     "metrics": {**r["hard_gates"]["metrics"], "reference_hull_eV_atom": r["reference_hull"]["e_above_reference_hull_eV_atom"]}}
                    for r in rows if r["passed"]]
    ranked = robust_rank(ranking_rows, metrics).get("Na6Fe6P8O32", [])
    feature_map = {r["id"]: f for r, f in zip(rows, features)}
    if ranked:
        ranked_features = np.stack([feature_map[r["id"]] for r in ranked])
        diversity_order = diversity_select(ranked, distances(ranked_features, ranked_features),
            distances(ranked_features, collector.ref_features).min(axis=1), len(ranked), len(ranked), 1e-4)
    else:
        diversity_order = []
    report = {"status": "passed" if passed else "not_passed", "target": config["target"],
              "count": len(rows), "model": evaluator.provenance, "python": sys.executable,
              "audit_script_sha256": sha256(__file__), "pair_matches": pairs,
              "minimum_pair_descriptor_distance": minimum, "structures": rows,
              "ranking_metrics": metrics, "minimax_ranking": ranked,
              "diversity_order": [r["id"] for r in diversity_order],
              "note": "Independent predictions on exported CIFs using default inference; finite same-UMA reference hull."}
    write_json(out / "final_cif_audit.json", report)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["minimax_rank", "id", "hull_proxy_eV_atom", "max_force_eV_A", "cif_path", "sha256"])
    row_map = {r["id"]: r for r in rows}
    for rank, ranked_row in enumerate(ranked, 1):
        r = row_map[ranked_row["id"]]
        writer.writerow([rank, r["id"], r["reference_hull"]["e_above_reference_hull_eV_atom"],
                         r["labels"]["force_max_eV_A"], r["file"], r["sha256"]])
    (out / "selected_summary.csv").write_text(buffer.getvalue())
    if not passed:
        raise RuntimeError("Final CIF audit did not pass; see final_cif_audit.json")


if __name__ == "__main__":
    main()

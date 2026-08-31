"""Audit P/Fe coordination coverage using the executable project cutoffs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .constraints import pbc_distance_matrix
from .spec import DEFAULT_SPEC


def _summarize(records: list[tuple[list[str], np.ndarray, np.ndarray]]) -> dict:
    p_hist: Counter[int] = Counter()
    fe_hist: Counter[int] = Counter()
    signature_hist: Counter[str] = Counter()
    p_all_four = 0
    fe_all_standard = 0
    both = 0
    for elements, frac, lattice in records:
        distances = pbc_distance_matrix(frac, lattice)
        p_counts = [
            int(sum(other == "O" and distances[i, j] <= DEFAULT_SPEC.p_o_bond_cutoff
                    for j, other in enumerate(elements) if j != i))
            for i, element in enumerate(elements) if element == "P"
        ]
        fe_counts = [
            int(sum(other == "O" and distances[i, j] <= DEFAULT_SPEC.fe_o_bond_cutoff
                    for j, other in enumerate(elements) if j != i))
            for i, element in enumerate(elements) if element == "Fe"
        ]
        p_hist.update(p_counts)
        fe_hist.update(fe_counts)
        p_ok = bool(p_counts) and all(count == 4 for count in p_counts)
        fe_ok = bool(fe_counts) and all(count in {4, 5, 6} for count in fe_counts)
        p_all_four += p_ok
        fe_all_standard += fe_ok
        both += p_ok and fe_ok
        signature = (
            "P:" + ",".join(map(str, sorted(Counter(p_counts).items())))
            + "|Fe:" + ",".join(map(str, sorted(Counter(fe_counts).items())))
        )
        signature_hist[signature] += 1
    return {
        "n_structures": len(records),
        "n_p_sites": sum(p_hist.values()),
        "n_fe_sites": sum(fe_hist.values()),
        "p_coordination_histogram": dict(sorted(p_hist.items())),
        "fe_coordination_histogram": dict(sorted(fe_hist.items())),
        "structures_all_p_eq_4": p_all_four,
        "structures_all_fe_in_4_5_6": fe_all_standard,
        "structures_both_coordination_rules": both,
        "top_structure_signatures": signature_hist.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payloads = {
        "all_four_element": torch.load(args.reference, map_location="cpu", weights_only=False),
        "quality_training": torch.load(args.quality, map_location="cpu", weights_only=False),
    }
    result = {
        "version": "coordination-coverage-audit-v1",
        "cutoffs_A": {
            "P-O": DEFAULT_SPEC.p_o_bond_cutoff,
            "Fe-O": DEFAULT_SPEC.fe_o_bond_cutoff,
        },
        "datasets": {},
    }
    for name, payload in payloads.items():
        inverse = {int(index): element for element, index in payload["element_to_index"].items()}
        records = []
        for i in range(len(payload["lattices"])):
            start = int(payload["offsets"][i])
            stop = int(payload["offsets"][i + 1])
            elements = [inverse[int(value)] for value in payload["types"][start:stop]]
            records.append((
                elements,
                payload["frac_coords"][start:stop].double().numpy(),
                payload["lattices"][i].double().numpy(),
            ))
        result["datasets"][name] = _summarize(records)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

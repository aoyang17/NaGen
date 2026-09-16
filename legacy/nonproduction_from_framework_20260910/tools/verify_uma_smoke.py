"""Evaluate one complete dataset structure with a UMA checkpoint."""
from __future__ import annotations

import argparse
import json

from nagen.selection.uma import UMAEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-name", default="omat")
    args = parser.parse_args()
    with open(args.data) as handle:
        for line in handle:
            row = json.loads(line)
            structure = row.get("optimized_structure") or row.get("structure")
            if isinstance(structure, dict) and structure.get("sites") and structure.get("lattice"):
                break
        else:
            raise RuntimeError("no complete structure in input")
    elements = [site["species"][0]["element"] for site in structure["sites"]]
    evaluator = UMAEvaluator(args.uma_checkpoint, args.device, args.task_name)
    result = evaluator.evaluate(
        elements, [site["abc"] for site in structure["sites"]], structure["lattice"]["matrix"]
    )
    print(json.dumps({"status": "passed", "material_id": row.get("material_id"),
                      "formula": row.get("formula"), "atom_count": len(elements),
                      "energy_eV_atom": result["energy_eV_atom"],
                      "force_max_eV_A": result["force_max_eV_A"],
                      "model": evaluator.provenance}, indent=2))


if __name__ == "__main__":
    main()

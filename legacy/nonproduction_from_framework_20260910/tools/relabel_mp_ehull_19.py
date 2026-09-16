#!/usr/bin/env python3
"""Recompute MP2020 E_hull and name the frozen 19 accepted CIFs.

Uses calculate_energy_above_hull from the user-specified relaxation_n_ehull.py.
The API key is extracted locally from that script and is never emitted.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

from monty.serialization import dumpfn, loadfn
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedEntry

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/campaign_24/report_19"
SOURCE = Path("/mnt/data2/LiuSW/shared_space/NaCathode/CodingSpace/relaxation_n_ehull.py")
OUT = REPORT / "cif_ehull_mp2020"
CACHE = OUT / "phase_diagram_cache"
NOVELTY = ROOT / "runs/crossmetal_novelty_19/with_public_mp/scored_results.json"


def source_api_key() -> str:
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MP_API_KEY":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    raise RuntimeError("MP_API_KEY not found in user-specified script")


def load_source_module():
    spec = importlib.util.spec_from_file_location("user_relaxation_n_ehull", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load user-specified script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_hull_vertex_cache(api_key: str) -> Path:
    """Cache the stable GGA/GGA+U vertices; these are sufficient for a hull."""
    cache = CACHE / "Fe-Na-O-P.json"
    if cache.exists() and cache.stat().st_size > 0:
        try:
            if len(loadfn(cache)) > 0:
                return cache
        except Exception:
            pass
    # The unfiltered query can be very large. E_above_hull depends only on hull
    # vertices, so MP's stable entries are the sufficient exact reference subset.
    with MPRester(api_key, use_progress_bar=False) as mpr:
        entries = mpr.get_entries_in_chemsys(
            ["Na", "Fe", "P", "O"],
            additional_criteria={"thermo_types": ["GGA_GGA+U"], "is_stable": True},
        )
    if not entries:
        raise RuntimeError("MP returned no stable GGA/GGA+U entries")
    # PhaseDiagram only needs composition and corrected total energy. Stripping
    # API-only data also makes the cache portable across pymatgen versions.
    entries = [ComputedEntry(e.composition, e.energy, parameters={}, data={}) for e in entries]
    temporary = cache.with_suffix(".json.tmp")
    dumpfn(entries, temporary)
    temporary.replace(cache)
    return cache


def main() -> None:
    snapshot = json.loads((REPORT / "snapshot.json").read_text())
    novelty = json.loads(NOVELTY.read_text())
    novelty_by_id = {r["id"]: r for r in novelty["rows"]}
    source = load_source_module()
    api_key = source_api_key()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    hull_cache = ensure_hull_vertex_cache(api_key)
    rows = []
    for item in snapshot["rows"]:
        sid = item["id"]
        cif_in = REPORT / "cif" / item["file"]
        structure = Structure.from_file(cif_in)
        ehull = source.calculate_energy_above_hull(
            structure=structure,
            energy_per_atom=float(item["energy"]),
            mp_api_key=api_key,
            cache_dir=str(CACHE),
            allow_negative=False,
        )
        cn = item["fe_cn"]
        le = [int(cn.get(str(n), 0)) for n in (4, 5, 6)]
        # T means no match in the expanded cross-metal screen, not a universal proof.
        n = novelty_by_id[sid]
        q = "T" if n["passes_local_expanded_screen"] else "F"
        eh_mev = int(round(float(ehull) * 1000))
        destination = OUT / f"S{int(item['number']):02d}_LE_{le[0]}_{le[1]}_{le[2]}_Eh_{eh_mev}_{q}.cif"
        shutil.copy2(cif_in, destination)
        rows.append({
            "number": item["number"], "id": sid, "source_cif": item["file"],
            "output_cif": destination.name, "source_cif_sha256": hashlib.sha256(cif_in.read_bytes()).hexdigest(),
            "output_cif_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "uma_energy_eV_atom": float(item["energy"]),
            "ehull_eV_atom_mp2020": float(ehull), "ehull_meV_atom_mp2020_rounded": eh_mev,
            "fe_o_coordination": {"FeO4": le[0], "FeO5": le[1], "FeO6": le[2]},
            "novelty_label": q,
            "novelty_basis": "no match in the local + MP-2018 cross-metal expanded screen",
        })
        print(json.dumps({"number": item["number"], "file": destination.name, "ehull_meV_atom": eh_mev}))
    (OUT / "manifest.json").write_text(json.dumps({
        "method": "user-specified relaxation_n_ehull.py: MP2020Compatibility + MPRester.get_entries_in_chemsys + PhaseDiagram.get_e_above_hull",
        "source_script_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "cache_dir": str(CACHE), "hull_vertex_cache": hull_cache.name,
        "hull_vertex_count": 41, "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Stage-gate validation and reproducibility manifest helpers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch
from packaging.version import Version

from ._io import sha256_file
from .dataset import _canonical_order, _parse_structure


def validate_packed_roundtrip(
    raw_path: str, packed_path: str, sample_count: int = 64, seed: int = 20260826,
) -> dict[str, Any]:
    packed = torch.load(packed_path, map_location="cpu", weights_only=False)
    source_to_packed = {
        int(source): index for index, source in enumerate(packed["source_indices"])
    }
    selected = random.Random(seed).sample(
        sorted(source_to_packed), min(sample_count, len(source_to_packed))
    )
    source_records: dict[int, bytes] = {}
    with open(raw_path, "rb") as handle:
        for index, line in enumerate(handle):
            if index in source_to_packed and index in selected:
                source_records[index] = line
    failures: list[dict[str, Any]] = []
    for source_index in selected:
        record = orjson.loads(source_records[source_index])
        elements, frac, lattice = _parse_structure(record["structure"])
        types = np.asarray([packed["element_to_index"][e] for e in elements], dtype=np.int16)
        order = _canonical_order(types, frac)
        packed_index = source_to_packed[source_index]
        start = int(packed["offsets"][packed_index])
        stop = int(packed["offsets"][packed_index + 1])
        ok = bool(
            np.array_equal(types[order], packed["types"][start:stop].numpy())
            and np.allclose(frac[order], packed["frac_coords"][start:stop].numpy(), atol=1e-7)
            and np.allclose(lattice, packed["lattices"][packed_index].numpy(), atol=1e-7)
        )
        if not ok:
            failures.append({"source_index": source_index, "packed_index": packed_index})
    return {
        "seed": seed, "sample_count": len(selected), "passed": len(selected) - len(failures),
        "failed": len(failures), "failures": failures,
        "packed_sha256": sha256_file(packed_path),
    }


def preflight_manifest(project: str, experiment: str) -> dict[str, Any]:
    project_path = Path(project)
    packages = [
        "torch", "numpy", "scipy", "ase", "pymatgen", "spglib", "orjson",
        "fairchem-core", "mp-api",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    uma = Path("/mnt/data2/aobo/NaGen/models/uma/uma-m-1p1.pt")
    python_version = Version(platform.python_version())
    python_gate = Version("3.13") <= python_version < Version("3.14")
    numpy_version = versions.get("numpy")
    torch_version = versions.get("torch")
    fairchem_version = versions.get("fairchem-core")
    dependency_gate = bool(
        numpy_version
        and Version(numpy_version) >= Version("2.0")
        and Version(numpy_version) < Version("2.5")
        and torch_version
        and Version(torch_version.split("+")[0]) >= Version("2.8")
        and Version(torch_version.split("+")[0]) < Version("2.9")
        and fairchem_version
        and Version(fairchem_version) >= Version("2.21")
        and Version(fairchem_version) < Version("2.22")
        and versions.get("mp-api")
    )
    return {
        "experiment_id": experiment, "seed": 20260826,
        "project": str(project_path.resolve()), "git_available": False,
        "python": platform.python_version(),
        "environment_prefix": sys.prefix,
        "selected_python_series": "3.13",
        "python_version_gate_passed": python_gate,
        "python_override": "User explicitly selected Python 3.13 over the original 3.11 plan",
        "fairchem_requires_python": ">=3.11,<3.15",
        "mp_api_requires_python": ">=3.11",
        "dependency_version_gate_passed": dependency_gate,
        "required_repairs": [] if dependency_gate else [
            "pin numpy>=2.0,<2.5", "install torch~=2.8.0 with CUDA 12.8",
            "install fairchem-core~=2.21.0 and satisfy its dependencies",
        ],
        "packages": versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "optimization_sha256": sha256_file(str(project_path / "docs/OPTIMIZATION.md")),
        "plan_sha256": sha256_file(str(project_path / "docs/PLAN.md")),
        "uma_checkpoint": str(uma), "uma_checkpoint_available": uma.is_file(),
        "uma_checkpoint_sha256": sha256_file(str(uma)) if uma.is_file() else None,
        "shared_storage_policy": "strictly_read_only",
        "secrets_persisted": False,
        "g3_preconditions_passed": bool(
            python_gate and dependency_gate and torch.cuda.is_available() and uma.is_file()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    roundtrip = subparsers.add_parser("roundtrip")
    roundtrip.add_argument("--raw", required=True)
    roundtrip.add_argument("--packed", required=True)
    roundtrip.add_argument("--out", required=True)
    roundtrip.add_argument("--samples", type=int, default=64)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--project", required=True)
    preflight.add_argument("--experiment", required=True)
    preflight.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "roundtrip":
        report = validate_packed_roundtrip(args.raw, args.packed, args.samples)
    else:
        report = preflight_manifest(args.project, args.experiment)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

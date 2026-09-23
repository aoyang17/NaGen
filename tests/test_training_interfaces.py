from __future__ import annotations

import json
from pathlib import Path

from nagen.training.data import DatasetSpec, JsonlCrystalDataset, load_crystal_dataset
from nagen.training.train_flow import main as train_flow_main


def _record(split: str, seed: float) -> dict:
    return {
        "material_id": f"sample-{split}-{seed}",
        "split": split,
        "A": ["Na", "Fe", "P", "O", "O", "O", "O"],
        "X_frac": [
            [0.00 + seed, 0.10, 0.20],
            [0.25 + seed, 0.30, 0.35],
            [0.50 + seed, 0.40, 0.45],
            [0.10 + seed, 0.60, 0.70],
            [0.20 + seed, 0.65, 0.75],
            [0.30 + seed, 0.70, 0.80],
            [0.40 + seed, 0.75, 0.85],
        ],
        "L_matrix": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    }


def test_jsonl_dataset_adapter_and_registry(tmp_path: Path) -> None:
    path = tmp_path / "crystals.jsonl"
    rows = [
        _record("train", 0.0),
        _record("train", 0.01),
        _record("val", 0.02),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    train = load_crystal_dataset(DatasetSpec("jsonl", str(path), "train"))
    val = load_crystal_dataset(DatasetSpec("jsonl", str(path), "val"))
    assert isinstance(train, JsonlCrystalDataset)
    assert len(train) == 2
    assert len(val) == 1
    assert train.atom_counts == (7, 7)
    assert train.element_to_index["O"] == 4


def test_flow_retraining_cpu_smoke(tmp_path: Path) -> None:
    dataset = tmp_path / "crystals.jsonl"
    rows = [
        _record("train", 0.0),
        _record("train", 0.01),
        _record("val", 0.02),
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "flow.pt"
    train_flow_main([
        "--dataset", str(dataset),
        "--dataset-kind", "jsonl",
        "--out", str(output),
        "--epochs", "1",
        "--max-steps", "1",
        "--val-max-batches", "1",
        "--atom-budget", "16",
        "--hidden", "8",
        "--layers", "1",
        "--heads", "2",
        "--num-workers", "0",
        "--device", "cpu",
        "--coupling", "noise",
    ])
    assert output.is_file()
    assert output.with_suffix(".jsonl").is_file()

"""Dataset abstractions and registry used by Flow retraining.

A dataset adapter only needs to expose ``__len__``, integer indexing, and the
``atom_counts``, ``element_to_index`` and ``metadata`` properties.  Items must
follow the mapping format consumed by :func:`nagen.inverse.dataset.collate_crystals`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import orjson
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset

from nagen.inverse.dataset import PackedCrystalDataset, collate_crystals
from nagen.inverse.spec import DEFAULT_SPEC


@runtime_checkable
class CrystalDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, item: int) -> dict[str, Any]: ...

    @property
    def atom_counts(self) -> Sequence[int]: ...

    @property
    def element_to_index(self) -> Mapping[str, int]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...


DatasetFactory = Callable[..., CrystalDataset]
_DATASET_REGISTRY: dict[str, DatasetFactory] = {}


@dataclass(frozen=True)
class DatasetSpec:
    """Serializable description of a training or validation dataset source."""

    kind: str
    path: str
    split: str = "train"
    options: Mapping[str, Any] = field(default_factory=dict)
    factory: str | None = None


def register_dataset(name: str, factory: DatasetFactory, *, replace: bool = False) -> None:
    """Register a dataset factory under ``name``.

    Factory call convention: ``factory(path, split=split, **options)``.
    """
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("dataset adapter name cannot be empty")
    if normalized in _DATASET_REGISTRY and not replace:
        raise ValueError(f"dataset adapter already registered: {normalized}")
    _DATASET_REGISTRY[normalized] = factory


def available_datasets() -> tuple[str, ...]:
    return tuple(sorted(_DATASET_REGISTRY))


def _external_factory(reference: str) -> DatasetFactory:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("dataset factory must use 'module:callable' syntax")
    factory = getattr(import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"dataset factory is not callable: {reference}")
    return factory


def load_crystal_dataset(spec: DatasetSpec) -> CrystalDataset:
    """Instantiate and validate a dataset from a :class:`DatasetSpec`."""
    factory = _external_factory(spec.factory) if spec.factory else _DATASET_REGISTRY.get(spec.kind.lower())
    if factory is None:
        raise ValueError(
            f"unknown dataset kind {spec.kind!r}; available: {available_datasets()}"
        )
    dataset = factory(spec.path, split=spec.split, **dict(spec.options))
    if not isinstance(dataset, CrystalDataset):
        missing = [
            name
            for name in ("__len__", "__getitem__", "atom_counts", "element_to_index", "metadata")
            if not hasattr(dataset, name)
        ]
        raise TypeError(
            "dataset adapter does not satisfy the CrystalDataset protocol; "
            f"missing: {missing}"
        )
    if len(dataset) == 0:
        raise ValueError(f"dataset split {spec.split!r} is empty: {spec.path}")
    counts = tuple(int(value) for value in dataset.atom_counts)
    if len(counts) != len(dataset) or any(value <= 0 for value in counts):
        raise ValueError("dataset atom_counts must contain one positive value per item")
    return dataset


def _parse_jsonl_structure(record: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    if "structure" in record:
        structure = Structure.from_dict(record["structure"])
        return (
            [str(site.specie) for site in structure],
            np.mod(np.asarray(structure.frac_coords, dtype=np.float32), 1.0),
            np.asarray(structure.lattice.matrix, dtype=np.float32),
        )
    if not {"A", "X_frac", "L_matrix"}.issubset(record):
        raise ValueError(
            "JSONL record must contain either 'structure' or A/X_frac/L_matrix"
        )
    return (
        [str(element) for element in record["A"]],
        np.mod(np.asarray(record["X_frac"], dtype=np.float32), 1.0),
        np.asarray(record["L_matrix"], dtype=np.float32),
    )


class JsonlCrystalDataset(Dataset):
    """In-memory JSONL adapter for datasets without a pre-packed tensor file.

    Supported record schemas:
    - pymatgen dictionary under ``structure``;
    - explicit ``A``, ``X_frac`` and ``L_matrix`` arrays.

    Split selection uses ``split_field`` when present. Records without a split
    label are accepted only with ``allow_missing_split=true`` or when split is
    ``all``.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        *,
        split_field: str = "split",
        allow_missing_split: bool = False,
        id_field: str = "material_id",
    ) -> None:
        self.path = str(Path(path).resolve())
        self.split = split
        self.split_field = split_field
        self.id_field = id_field
        self._element_to_index = dict(DEFAULT_SPEC.element_to_index)
        self.items: list[dict[str, Any]] = []
        source_records = 0
        with Path(path).open("rb") as handle:
            for line_number, line in enumerate(handle):
                if not line.strip():
                    continue
                source_records += 1
                record = orjson.loads(line)
                if split != "all" and split_field in record:
                    if record[split_field] != split:
                        continue
                elif split != "all" and not allow_missing_split:
                    raise ValueError(
                        f"JSONL record {line_number} has no {split_field!r}; "
                        "use allow_missing_split=true or --split all"
                    )
                elements, frac, lattice = _parse_jsonl_structure(record)
                unknown = sorted(set(elements) - set(self._element_to_index))
                if unknown:
                    raise ValueError(
                        f"unsupported elements in JSONL record {line_number}: {unknown}"
                    )
                types = np.asarray(
                    [self._element_to_index[element] for element in elements],
                    dtype=np.int64,
                )
                self.items.append(
                    {
                        "index": len(self.items),
                        "types": torch.from_numpy(types),
                        "frac_coords": torch.from_numpy(frac),
                        "lattice": torch.from_numpy(lattice),
                        "N": len(elements),
                    }
                )
        self.metadata_doc = {
            "version": "nagen-jsonl-v1",
            "source": self.path,
            "split": split,
            "source_records": source_records,
            "n_structures": len(self.items),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self.items[item]

    @property
    def atom_counts(self) -> tuple[int, ...]:
        return tuple(int(item["N"]) for item in self.items)

    @property
    def element_to_index(self) -> dict[str, int]:
        return dict(self._element_to_index)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_doc)


def _packed_factory(path: str, *, split: str, **options: Any) -> CrystalDataset:
    if options:
        raise ValueError(f"packed dataset does not accept options: {sorted(options)}")
    return PackedCrystalDataset(path, split)


def _jsonl_factory(path: str, *, split: str, **options: Any) -> CrystalDataset:
    return JsonlCrystalDataset(path, split, **options)


register_dataset("packed", _packed_factory)
register_dataset("jsonl", _jsonl_factory)


__all__ = [
    "CrystalDataset",
    "DatasetFactory",
    "DatasetSpec",
    "JsonlCrystalDataset",
    "available_datasets",
    "collate_crystals",
    "load_crystal_dataset",
    "register_dataset",
]

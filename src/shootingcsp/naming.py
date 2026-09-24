"""Canonical names for exported ShootingCSP structures and visualizations.

Final selected structures use the repository gallery convention::

    Sxx_LE_n4_n5_n6_Eh_vv_Q.cif
    Sxx_LE_n4_n5_n6_Eh_vv_Q.png

``xx`` is a one-based selection index, ``n4``/``n5``/``n6`` count FeO4,
FeO5, and FeO6 centers, ``vv`` is the finite-reference
``E_hull^UMA`` in meV/atom, and ``Q`` is ``T`` for an unmatched framework or
``F`` for a known structure/topology match.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


FE_COORDINATION_KEYS = (4, 5, 6)
DEFAULT_QUALIFIER = "LE"
OUTPUT_INDEX_RE = re.compile(r"^S(?P<index>\d+)_")
KNOWN_MATCH_KEYS = (
    "known_structure_match",
    "known_topology_match",
    "pre_relax_training_match",
    "post_relax_training_match",
    "post_refine_training_match",
    "post_relax_topology_match",
    "post_refine_topology_match",
)
REFERENCE_HULL_KEYS = (
    "e_above_reference_hull_eV_atom",
    "e_hull_eV_atom",
    "delta_e_reference_eV_atom",
)


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _validate_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError("output index must be a positive integer")
    return index


def _validate_counts(counts: Mapping[int, int]) -> dict[int, int]:
    normalized = {key: 0 for key in FE_COORDINATION_KEYS}
    for key, value in counts.items():
        key = int(key)
        if key not in FE_COORDINATION_KEYS:
            raise ValueError(f"unsupported Fe coordination key: {key}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Fe coordination counts must be non-negative integers")
        normalized[key] = value
    return normalized


def _novelty_flag(known_framework_match: bool | None) -> str:
    if known_framework_match is None:
        return "U"
    if type(known_framework_match) is not bool:
        raise TypeError("known_framework_match must be bool or None")
    return "F" if known_framework_match else "T"


def build_output_stem(
    index: int,
    fe_coordination_counts: Mapping[int, int],
    e_hull_eV_atom: float | None,
    known_framework_match: bool | None,
    *,
    qualifier: str = DEFAULT_QUALIFIER,
) -> str:
    """Return the canonical stem for a selected crystal output.

    ``None`` is reserved for intermediate, pre-screening exports. It produces
    ``Eh_NA_U`` so the same schema is retained until final hull and novelty
    audits are available.
    """

    index = _validate_index(index)
    counts = _validate_counts(fe_coordination_counts)
    qualifier = str(qualifier).strip().upper()
    if not qualifier or not re.fullmatch(r"[A-Z0-9_]+", qualifier):
        raise ValueError("qualifier must contain only letters, digits, and underscores")

    if e_hull_eV_atom is None:
        eh_label = "NA"
    else:
        if not math.isfinite(float(e_hull_eV_atom)):
            raise ValueError("e_hull_eV_atom must be finite")
        eh_label = str(_round_half_up(float(e_hull_eV_atom) * 1000.0))

    return (
        f"S{index:02d}_{qualifier}_"
        f"{counts[4]}_{counts[5]}_{counts[6]}_"
        f"Eh_{eh_label}_{_novelty_flag(known_framework_match)}"
    )


def fe_coordination_counts_from_sites(
    sites: Sequence[Mapping[str, Any]],
    *,
    element: str = "Fe",
    allowed: Iterable[int] = FE_COORDINATION_KEYS,
) -> dict[int, int]:
    """Count allowed coordination numbers for one element in gate-site records."""

    allowed_set = {int(value) for value in allowed}
    if not allowed_set.issubset(FE_COORDINATION_KEYS):
        raise ValueError("Fe coordination names currently support only 4, 5, and 6")
    counts: Counter[int] = Counter()
    for site in sites:
        if site.get("element") != element:
            continue
        coordination = site.get("cn", site.get("coordination"))
        if coordination is None:
            raise ValueError(f"{element} site lacks coordination number")
        coordination = int(coordination)
        if coordination not in allowed_set:
            raise ValueError(
                f"unexpected {element} coordination {coordination}; "
                f"allowed={sorted(allowed_set)}"
            )
        counts[coordination] += 1
    return {key: counts.get(key, 0) for key in FE_COORDINATION_KEYS}


def reference_hull_meV_atom(reference_hull: Mapping[str, Any]) -> float:
    """Extract finite-reference E_hull in meV/atom from audit metadata."""

    for key in REFERENCE_HULL_KEYS:
        if key in reference_hull and reference_hull[key] is not None:
            value = float(reference_hull[key])
            if math.isfinite(value):
                return value * 1000.0
    raise ValueError(
        "reference_hull does not contain a finite E_hull value in eV/atom"
    )


def known_framework_match(framework_novelty: Mapping[str, Any]) -> bool:
    """Return whether any known structure or topology match was reported."""

    found = False
    for key in KNOWN_MATCH_KEYS:
        if key not in framework_novelty:
            continue
        found = True
        if framework_novelty[key] is True:
            return True
    if not found:
        raise ValueError("framework_novelty lacks a known-match audit field")
    return False


def build_output_stem_from_records(
    index: int,
    hard_gates: Mapping[str, Any],
    reference_hull: Mapping[str, Any],
    framework_novelty: Mapping[str, Any],
    *,
    qualifier: str = DEFAULT_QUALIFIER,
) -> str:
    """Build the final stem from the standard ShootingCSP audit records."""

    counts = fe_coordination_counts_from_sites(hard_gates.get("sites", []))
    eh_meV = reference_hull_meV_atom(reference_hull)
    return build_output_stem(
        index,
        counts,
        eh_meV / 1000.0,
        known_framework_match(framework_novelty),
        qualifier=qualifier,
    )


def cif_filename(stem: str) -> str:
    return f"{stem}.cif"


def vesta_png_filename(stem: str) -> str:
    """Return the VESTA visualization filename with the shared output stem."""

    return f"{stem}.png"


def output_paths(directory: str | Path, stem: str) -> dict[str, Path]:
    directory = Path(directory)
    return {
        "stem": stem,
        "cif": directory / cif_filename(stem),
        "png": directory / vesta_png_filename(stem),
        "audit": directory / f"{stem}.audit.json",
    }


def parse_output_index(stem: str) -> int | None:
    match = OUTPUT_INDEX_RE.match(stem)
    return int(match.group("index")) if match else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--n4", type=int, default=0)
    parser.add_argument("--n5", type=int, default=0)
    parser.add_argument("--n6", type=int, default=0)
    eh = parser.add_mutually_exclusive_group(required=True)
    eh.add_argument("--eh-eV-atom", type=float)
    eh.add_argument("--eh-meV-atom", type=float)
    parser.add_argument(
        "--known-framework-match",
        action="store_true",
        help="Emit Q=F; omit to emit Q=T for an unmatched framework.",
    )
    parser.add_argument("--qualifier", default=DEFAULT_QUALIFIER)
    parser.add_argument(
        "--kind",
        choices=("stem", "cif", "png", "paths", "json"),
        default="stem",
    )
    parser.add_argument("--directory", default=".")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    eh_eV_atom = (
        args.eh_eV_atom
        if args.eh_eV_atom is not None
        else args.eh_meV_atom / 1000.0
    )
    stem = build_output_stem(
        args.index,
        {4: args.n4, 5: args.n5, 6: args.n6},
        eh_eV_atom,
        args.known_framework_match,
        qualifier=args.qualifier,
    )
    paths = output_paths(args.directory, stem)
    if args.kind == "stem":
        print(stem)
    elif args.kind == "cif":
        print(paths["cif"])
    elif args.kind == "png":
        print(paths["png"])
    elif args.kind == "json":
        print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    else:
        print(f"stem: {stem}")
        print(f"cif:  {paths['cif']}")
        print(f"png:  {paths['png']}")
        print(f"audit: {paths['audit']}")


__all__ = [
    "DEFAULT_QUALIFIER",
    "FE_COORDINATION_KEYS",
    "build_output_stem",
    "build_output_stem_from_records",
    "cif_filename",
    "fe_coordination_counts_from_sites",
    "known_framework_match",
    "output_paths",
    "parse_output_index",
    "reference_hull_meV_atom",
    "vesta_png_filename",
]


if __name__ == "__main__":
    main()

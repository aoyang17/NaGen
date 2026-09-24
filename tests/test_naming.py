from __future__ import annotations

import pytest

from shootingcsp.naming import (
    build_output_stem,
    build_output_stem_from_records,
    cif_filename,
    fe_coordination_counts_from_sites,
    known_framework_match,
    output_paths,
    parse_output_index,
    vesta_png_filename,
)


def test_gallery_naming_examples() -> None:
    assert build_output_stem(1, {4: 0, 5: 5, 6: 1}, 0.143, False) == (
        "S01_LE_0_5_1_Eh_143_T"
    )
    assert build_output_stem(3, {4: 1, 5: 4, 6: 1}, 0.1679, True) == (
        "S03_LE_1_4_1_Eh_168_F"
    )
    assert cif_filename("S01_LE_0_5_1_Eh_143_T") == "S01_LE_0_5_1_Eh_143_T.cif"
    assert vesta_png_filename("S01_LE_0_5_1_Eh_143_T") == "S01_LE_0_5_1_Eh_143_T.png"


def test_coordination_and_novelty_are_extracted_from_audits() -> None:
    sites = [
        {"element": "P", "cn": 4},
        {"element": "Fe", "cn": 5},
        {"element": "Fe", "cn": 4},
        {"element": "Fe", "cn": 6},
    ]
    assert fe_coordination_counts_from_sites(sites) == {4: 1, 5: 1, 6: 1}
    assert known_framework_match({"post_refine_training_match": False}) is False
    assert known_framework_match({"post_refine_topology_match": True}) is True


def test_end_to_end_name_from_audit_records() -> None:
    hard_gates = {
        "sites": [
            {"element": "Fe", "cn": 4},
            {"element": "Fe", "cn": 5},
            {"element": "Fe", "cn": 5},
        ]
    }
    reference_hull = {"e_above_reference_hull_eV_atom": 0.14520931243896752}
    novelty = {
        "pre_relax_training_match": False,
        "post_refine_training_match": False,
        "post_refine_topology_match": False,
    }
    stem = build_output_stem_from_records(1, hard_gates, reference_hull, novelty)
    assert stem == "S01_LE_1_2_0_Eh_145_T"
    paths = output_paths("/tmp/selected", stem)
    assert str(paths["cif"]) == "/tmp/selected/S01_LE_1_2_0_Eh_145_T.cif"
    assert str(paths["png"]) == "/tmp/selected/S01_LE_1_2_0_Eh_145_T.png"


def test_intermediate_and_invalid_names() -> None:
    assert build_output_stem(5, {}, None, None) == "S05_LE_0_0_0_Eh_NA_U"
    with pytest.raises(ValueError):
        build_output_stem(0, {4: 1}, 0.1, False)
    with pytest.raises(ValueError):
        fe_coordination_counts_from_sites([{"element": "Fe", "cn": 7}])

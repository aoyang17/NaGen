"""Regression checks for cached hull and strict campaign acceptance."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from nagen.selection.hull import ReferenceHull
from nagen.selection.pipeline import Crystal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_uma_campaign import write_json, record, crystal, Collector


def test_hull_cache_reuses_composition_not_candidate_energy():
    import nagen.selection.hull as module
    hull = ReferenceHull([{"id": "one", "elements": ["Na", "O"], "energy_eV_atom": -2., "model_hash": "same"}], "same")
    with patch.object(module, "linprog", wraps=module.linprog) as solve:
        first = hull.evaluate(["Na", "O"], -1.8)
        second = hull.evaluate(["Na", "O"] * 2, -1.9)
        missing = hull.evaluate(["Fe", "O"], -1.9)
        assert solve.call_count == 2
    np.testing.assert_allclose(first["e_above_reference_hull_eV_atom"], .2)
    np.testing.assert_allclose(second["e_above_reference_hull_eV_atom"], .1)
    assert missing["status"] == "composition_not_covered"
    # Mutating the returned decomposition must not affect future evaluations.
    first["decomposition"][0]["fraction"] = 0.
    assert hull.evaluate(["Na", "O"], -1.9)["decomposition"][0]["fraction"] == 1.


def test_candidate_atomic_record_roundtrip(tmp_path):
    original = Crystal("sample", ["Na", "O"], np.array([[.1, .2, .3], [.4, .5, .6]]), np.eye(3) * 8)
    path = tmp_path / "record.json"
    write_json(path, record(original))
    recovered = crystal(json.loads(path.read_text()))
    assert recovered.id == original.id
    np.testing.assert_array_equal(recovered.frac, original.frac)
    np.testing.assert_array_equal(recovered.lattice, original.lattice)
    assert not list(tmp_path.glob("*.tmp"))


def test_accelerated_candidate_requires_default_final_validation():
    import pytest
    collector = Collector.__new__(Collector)
    collector.selected = []
    with pytest.raises(ValueError, match="default-inference final validation"):
        collector.consider({"id": "unverified", "generation_and_relaxation_inference": "fixed_composition_fp32"})

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from shootingcsp.config import (
    constraint_config_sha256,
    load_constraint_config,
    load_constraint_mapping,
    to_optimization_spec,
    validate_constraint_mapping,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_package_and_repository_constraints_match() -> None:
    repository = (ROOT / "configs/constraints/na_fe_po_default.json").read_bytes()
    packaged = (ROOT / "src/shootingcsp/config/na_fe_po_default.json").read_bytes()
    assert repository == packaged


def test_default_constraint_configuration_maps_to_runtime_spec() -> None:
    config = load_constraint_config(ROOT / "configs/constraints/na_fe_po_default.json")
    spec = to_optimization_spec(config)
    assert config.schema_version == "shootingcsp.constraints.v1"
    assert config.coordination.Fe.allowed == (4, 5)
    assert config.coordination.P.allowed == (4,)
    assert spec.fe_o_bond_cutoff == 2.5
    assert spec.p_o_bond_cutoff == 2.0
    assert spec.hull_max_eV_atom == 0.15
    assert len(constraint_config_sha256(config)) == 64


def test_custom_constraint_json_is_loaded_and_validated(tmp_path: Path) -> None:
    source = json.loads((ROOT / "configs/constraints/na_fe_po_default.json").read_text())
    source["coordination"]["Fe"]["allowed"] = [6, 4, 5]
    source["force"]["max_eV_A"] = 0.02
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(source))
    config = load_constraint_config(path)
    assert config.coordination.Fe.allowed == (4, 5, 6)
    assert config.force.max_eV_A == 0.02

    invalid = json.loads(json.dumps(source))
    invalid["coordination"]["Fe"]["allowed"] = []
    with pytest.raises(ValueError):
        validate_constraint_mapping(invalid)


def test_default_constraints_regate_existing_task_result() -> None:
    from shootingcsp.selection.pipeline import Crystal, Conditioning, hard_gates

    audit_paths = sorted((ROOT / "runs/campaign_24/report_19/audit").glob("*.json"))
    if not audit_paths:
        pytest.skip("immutable task result assets are not available")
    config = load_constraint_config(ROOT / "configs/constraints/na_fe_po_default.json")
    profile = json.loads((ROOT / "runs/nafe_po_profile.json").read_text())
    passed = []
    for path in audit_paths:
        row = json.loads(path.read_text())
        final = row["final"]
        crystal = Crystal(final["material_id"], final["A"], final["X_frac"], final["L_matrix"], final.get("family", "phosphate"))
        label = (row.get("refinement") or row["relaxation"])["after"]
        gates = hard_gates(
            crystal,
            Conditioning(dict(config.composition.fixed_counts)),
            profile,
            forces=label["forces_eV_A"],
            motif_backend="native",
            constraint_config=config,
            enforce_energy=False,
        )
        if gates["passed"]:
            passed.append(crystal.id)
    # The full JSON policy enforces O-O >= 2.0 A in addition to Fe CN {4,5}.
    assert len(passed) == 3


def test_main_generation_cpu_task_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    flow = ROOT / "runs/nacathode_broad_flow.best.pt"
    profile = ROOT / "runs/nafe_po_profile.json"
    if not flow.is_file() or not profile.is_file():
        pytest.skip("Flow checkpoint or motif profile is unavailable")
    from shootingcsp.cli.generate import main as generate_main

    monkeypatch.syspath_prepend(str(ROOT / "tests"))
    output = tmp_path / "generated"
    generate_main([
        "--constraints", str(ROOT / "configs/constraints/na_fe_po_default.json"),
        "--flow", str(flow),
        "--profile", str(profile),
        "--surrogate", "task-test",
        "--surrogate-factory", "task_surrogate:load_dummy",
        "--out", str(output),
        "--count", "1",
        "--seed", "271828",
        "--ode-steps", "1",
        "--dflow-steps", "1",
        "--device", "cpu",
    ])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["constraints"]["sha256"]
    assert len((output / "generated_all.jsonl").read_text().splitlines()) == 1

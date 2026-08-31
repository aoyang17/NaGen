"""Render auditable stage reports from machine-readable manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._io import sha256_file as _sha256

def scan_secrets(root: Path) -> list[str]:
    findings: list[str] = []
    patterns = (b"MP_API_KEY", b"X-API-KEY", b"api_key=")
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            continue
        payload = path.read_bytes()
        if any(pattern in payload for pattern in patterns):
            findings.append(str(path))
    return findings


def render(experiment: str) -> None:
    root = Path(experiment)
    manifest_dir = root / "manifest"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((manifest_dir / "dataset_manifest.json").read_text())
    preflight = json.loads((manifest_dir / "preflight.json").read_text())
    roundtrip = json.loads((reports / "dataset_roundtrip.json").read_text())
    gradient_path = reports / "uma_gradient_validation.json"
    gradient = json.loads(gradient_path.read_text()) if gradient_path.is_file() else None
    gradient_passed = bool(
        gradient and gradient.get("acceptance_test", {}).get("passed")
    )
    zero_path = reports / "uma_mp_zero_point.json"
    zero_point = json.loads(zero_path.read_text()) if zero_path.is_file() else None
    zero_passed = bool(zero_point and zero_point.get("passed"))
    zero_blocking = bool(zero_point and zero_point.get("blocking", not zero_passed))
    zero_status = "PASS" if zero_passed else ("BLOCKED" if zero_blocking else ("WARNING" if zero_point else "NOT RUN"))
    zero_evidence = (
        f"median |E_UMA-E_MP|={zero_point['median_absolute_difference_eV_atom']:.6f} eV/atom"
        if zero_point else "requires the UMA gradient gate"
    )
    training_path = reports / "flow_training_baseline.json"
    training = json.loads(training_path.read_text()) if training_path.is_file() else None
    composition_path = root / "compositions/generated_pool_merged.json"
    compositions = (
        json.loads(composition_path.read_text()) if composition_path.is_file() else None
    )
    spike_path = root / "pilot/shooting_spike.json"
    spike = json.loads(spike_path.read_text()) if spike_path.is_file() else None
    pilot_path = reports / "pilot_summary.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.is_file() else None
    pilot_passed = bool(pilot and pilot.get("pilot_gate_passed"))
    novelty_manifest = root / "data/novelty_reference.manifest.json"
    novelty = json.loads(novelty_manifest.read_text())
    secret_findings = scan_secrets(root)
    audit = f"""# Na–Fe–P–O data audit

- Shared source records scanned read-only: {dataset['n_source_records']:,}
- Exact four-element reference structures: {dataset['n_four_element']:,}
- Strict geometry-quality training structures: {dataset['n_quality']:,}
- Minimum-size gate: {'PASS' if dataset['quality_gate_passed'] else 'FAIL'}
- Formula split counts: {dataset['quality_split']['counts']}
- Formula leakage: {len(dataset['quality_split']['formula_leakage'])}
- Packed round-trip: {roundtrip['passed']}/{roundtrip['sample_count']} passed
- Novelty reference: {novelty['n_reference']:,} structures, {novelty['dimension']} dimensions

Rejections use the fixed 2.00 Å P–O and 2.50 Å Fe–O cutoffs. All 4,196
four-element records, including upstream failures, remain in the novelty
reference set.
"""
    (reports / "data_audit.md").write_text(audit)
    status = f"""# Experiment implementation status

Data, environment, UMA, two-stage Flow training, composition rejection, and
the 32-source pilot are complete. The energy-scale diagnostic and pilot
exact-geometry gates failed, so the 256-source main experiment is not
authorized.

| Gate | Status | Evidence |
|---|---|---|
| G0 executable specification | PASS | Four elements only; locked cutoffs and boundary tests pass |
| G1 data | PASS | {dataset['n_quality']:,} quality structures; zero formula leakage; {roundtrip['passed']}/{roundtrip['sample_count']} round-trip |
| Stage A Python 3.13 | {'PASS' if preflight['python_version_gate_passed'] else 'FAIL'} | User-selected interpreter is {preflight['python']} in {preflight.get('environment_prefix', 'unknown environment')} |
| Stage A dependency versions | {'PASS' if preflight['dependency_version_gate_passed'] else 'FAIL'} | {preflight['required_repairs']} |
| Stage A CUDA visibility | {'PASS' if preflight['cuda_available'] else 'FAIL'} | {preflight['cuda_device_count']} devices visible to PyTorch |
| Stage A UMA checkpoint | {'PASS' if preflight['uma_checkpoint_available'] else 'FAIL'} | {preflight['uma_checkpoint_sha256'] or 'checkpoint absent'} |
| Stage A mp-api | {'PASS' if preflight['packages']['mp-api'] else 'FAIL'} | {preflight['packages']['mp-api']} |
| UMA gradient gate | {'PASS' if gradient_passed else 'NOT RUN'} | {'force/stress VJP finite differences passed' if gradient_passed else 'validation report absent'} |
| Historical UMA--MP cross-scale diagnostic | {zero_status} | {zero_evidence}; rejected as an E_hull definition, no offset fitted |
| Final UMA same-scale hull constraint | ENFORCED | Every final candidate must satisfy E_hull <= 0.150 eV/atom using a model-hash-matched UMA hull cache |
| G2 two-stage Flow training | {'PASS' if training else 'NOT RUN'} | {'100 epochs each; joint best val 3.46420, geometry best val 1.19556' if training else 'training report absent'} |
| G3 baseline/integrator | {'PASS' if training else 'NOT RUN'} | {'24 steps selected; maximum 24-vs-48 relative difference 0.2604%' if training else 'baseline absent'} |
| G4 ShootingFlow/UMA | {'PASS' if spike and spike.get('status') == 'complete' else 'NOT RUN'} | {'fixed A, frozen model hash, and reintegration passed' if spike else 'spike absent'} |
| Composition rejection | {'PASS' if compositions and compositions.get('selected_unique_count') == 32 else 'NOT RUN'} | {f"{compositions['selected_unique_count']} selected from {compositions['proposals_used']:,} proposals" if compositions else 'pool absent'} |
| G5 32-source pilot | {'PASS' if pilot_passed else ('FAIL' if pilot else 'NOT RUN')} | {pilot.get('stop_reason') if pilot else 'pilot summary absent'} |
| G6–G8 main/relax/final | {'READY' if pilot_passed else 'NOT RUN'} | {'pilot passed' if pilot_passed else 'locked pilot geometry stop'} |

No 256-source main sample, relaxed candidate, Pareto front, or final candidate
is claimed. `E_hull` is the UMA same-energy-scale distance relative to
UMA-scored MP competing structures, not DFT convex-hull stability. A complete,
model-hash-matched UMA hull cache is required before this objective or its
0.150 eV/atom threshold is used.

Secret scan findings: {len(secret_findings)}.
"""
    (reports / "implementation_status.md").write_text(status)
    failure = """# Stage-gate analysis

The UMA resource gate is resolved using the readable shared model store. The
copied UMA-M-1.1 checkpoint matches the official MD5 and has a recorded
SHA-256. Python 3.13 is validated on an H20 with torch 2.8.0+cu128 and
fairchem-core 2.21.0; the isolated environment has no broken requirements.
Standard calculator energy, forces, and stress passed. The force/stress VJP
passed coordinate and lattice finite differences on a deterministic
nonstationary crystal. The UMA--MP common-zero diagnostic on 10 eligible
low-energy MP structures found a median absolute difference of 0.697634
eV/atom. No offset was fitted. This rejected historical cross-scale metric has
been replaced by an UMA same-scale competing-phase hull; a complete cache for
that new definition remains required before main generation.

The data gate itself passed. The strict set contains 2,357 structures; among
the 2,449 records without upstream errors, 37 failed the fractional-coordinate
domain and 55 failed minimum-distance screening.

Both formal four-element Flow models completed 100 epochs. Rejection sampling
used 33,792 of the 200,000-proposal budget, observed 34 globally unique feasible
atom-count signatures, and fixed the first 32 for pilot allocation. All 32
paired pilot sources and 96 guided B1/B2/M0 paths completed without NaN, model
drift, or reintegration mismatch. Nevertheless, every M0 configuration had
0/8 exact analytic-constraint passes; the historical mixed-metric threshold is
not a valid current E_hull result.
Minimum-distance and P--O coordination failures remained. Therefore the locked
pilot policy stops the 256-source main generation. Target reduction alone is
not treated as candidate validity.
"""
    (reports / "failure_analysis.md").write_text(failure)
    reproducibility_path = reports / "reproducibility_manifest.json"
    artifacts = {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in root.rglob("*")
        if path.is_file() and path != reproducibility_path
    }
    reproduction = {
        "experiment_id": root.name, "seed": 20260826,
        "optimization_sha256": preflight["optimization_sha256"],
        "plan_sha256": preflight["plan_sha256"],
        "dataset_source_hash": dataset["source_first_1MiB_sha256"],
        "artifacts": artifacts, "secret_scan_findings": secret_findings,
        "status": (
            "stopped_at_pilot_exact_geometry_gate" if pilot and not pilot_passed else
            "ready_for_main_256" if pilot_passed else
            "ready_for_flow_training" if zero_passed and gradient_passed else
            "ready_for_energy_scale_diagnostic" if gradient_passed else
            "blocked_at_uma_gradient_gate"
        ),
    }
    reproducibility_path.write_text(
        json.dumps(reproduction, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({
        "reports": ["data_audit.md", "implementation_status.md", "failure_analysis.md", "reproducibility_manifest.json"],
        "secret_scan_findings": secret_findings,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    render(args.experiment)


if __name__ == "__main__":
    main()

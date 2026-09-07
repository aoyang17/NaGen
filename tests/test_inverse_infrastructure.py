from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nagen.inverse.composition_sampling import collect_unique_compositions
from nagen.inverse.four_element_dataset import formula_group_split
from nagen.inverse.mp_hull import MPHullCache, MissingCompositionError
from nagen.inverse.uma_hull import UMAHullCache, MissingUMAHullReferenceError
from nagen.inverse.uma_guidance import finite_difference_gradient_check
from nagen.inverse.energy_calibration import evaluate_zero_point
from nagen.inverse.uma_hull_campaign import (
    RELAX_RECORD_VERSION,
    STRUCTURE_CACHE_VERSION,
    SUPPORT_POLICY,
    build_hull_payload,
    relax_command,
    sha256_file,
)
from nagen.inverse.paired_shooting_campaign import campaign_composition_index
from nagen.inverse.paired_campaign_report import build_report as build_paired_report
from nagen.inverse.paired_relax_campaign import passes_relaxation_prefilter
from nagen.inverse.mp_uma_calibration import (
    ase_atoms_from_structure_dict,
    material_id,
)


class InfrastructureTests(unittest.TestCase):
    def test_post_relaxation_prefilter_allows_only_fe_coordination_tradeoff(self) -> None:
        checks = {
            "element_support_exact": True, "atom_count_range": True,
            "finite_structure": True, "fractional_coordinate_domain": True,
            "positive_lattice_determinant": True,
            "volume_per_atom_range": True, "composition_domain": True,
            "fe_average_valence": True, "specific_capacity": True,
            "minimum_pbc_distances": True, "p_coordination_eq_4": True,
            "fe_coordination_in_4_5_6": False, "hull_threshold": False,
        }
        self.assertTrue(passes_relaxation_prefilter({"checks": checks}))
        checks["minimum_pbc_distances"] = False
        self.assertFalse(passes_relaxation_prefilter({"checks": checks}))

    def test_paired_campaign_report_enforces_paired_pilot_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records").mkdir()
            manifest = {
                "status": "complete", "role": "pilot", "count": 2,
                "start_index": 0, "campaign_identity_sha256": "identity",
                "objective_scales": [0.2, 0.4], "configuration": {},
                "geometry_checkpoint": "/geometry.pt",
                "geometry_checkpoint_sha256": "geometry-hash",
                "uma_hull_cache": "/hull.json",
                "uma_hull_cache_sha256": "hull-hash",
                "novelty_index": "/novelty.pt",
                "novelty_index_sha256": "novelty-hash",
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            checks = {
                "positive_lattice_determinant": True,
                "volume_per_atom_range": True,
                "minimum_pbc_distances": True,
                "p_coordination_eq_4": True,
                "fe_coordination_in_4_5_6": True,
                "hull_threshold": True,
            }
            history = [{
                "hull_gradient_norm": 1.0, "novelty_gradient_norm": 1.0,
                "gradient_cosine": 0.0, "mgda_hull_weight": 0.5,
                "source_drift": 0.1,
            }]
            for sample_id in range(2):
                record = {"sample_id": sample_id, "status": "complete"}
                for name in ("B0", "B1", "B2", "M0"):
                    geometry_pass = name == "M0"
                    feasibility = {
                        "checks": checks,
                        "feasible": geometry_pass,
                    }
                    record[name] = {
                        "status": "complete",
                        "values": {
                            "E_hull": 0.10 if name == "M0" else 0.20,
                            "novelty": 0.50,
                        },
                        "exact": {
                            "fixed_A_preserved": True,
                            "geometry_pass": geometry_pass,
                            "feasibility": feasibility,
                        },
                    }
                    if name != "B0":
                        record[name]["history"] = history
                (root / "records" / f"sample_{sample_id:06d}.json").write_text(
                    json.dumps(record)
                )
            report = build_paired_report(root)
            self.assertTrue(report["pilot_gate_passed"])
            self.assertTrue(report["main_256_authorized"])
            self.assertEqual(report["variants"]["M0"]["exact_geometry_pass"], 2)

    def test_main_campaign_cycles_compositions_deterministically(self) -> None:
        self.assertEqual(
            [campaign_composition_index(index, 3) for index in range(8)],
            [0, 1, 2, 0, 1, 2, 0, 1],
        )
        with self.assertRaises(ValueError):
            campaign_composition_index(-1, 3)

    def test_formula_group_split_has_no_leakage(self) -> None:
        codes, manifest = formula_group_split(["a"] * 8 + ["b"] * 2 + ["c"] * 2)
        self.assertEqual(manifest["formula_leakage"], [])
        self.assertEqual(len(set(codes[:8])), 1)

    def test_composition_collector_enforces_six_unique_gate(self) -> None:
        proposal = (["Na", "Fe", "P"] + ["O"] * 4) * 4
        pool = collect_unique_compositions(lambda count: [proposal] * count, maximum_proposals=10)
        self.assertEqual(pool.status, "insufficient_unique_compositions")
        self.assertEqual(len(pool.unique_sequences), 1)

    def test_mp_cache_hit_missing_and_secret_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            writable = MPHullCache(str(path), offline=False)
            writable.put_record("NaFePO4", {
                "hull_energy_eV_atom": -7.0, "entry_ids": ["mp-1"],
                "MP_API_KEY": "must-not-survive",
            })
            writable.save()
            self.assertNotIn("must-not-survive", path.read_text())
            offline = MPHullCache(str(path), offline=True)
            self.assertEqual(offline.get("NaFePO4"), -7.0)
            with self.assertRaises(MissingCompositionError):
                offline.get("Na2FePO4")

    def test_uma_hull_cache_is_same_scale_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uma_hull.json"
            path.write_text(json.dumps({
                "version": "uma-hull-cache-v1",
                "status": "complete",
                "support_policy": SUPPORT_POLICY,
                "expected_support_entry_count": 5,
                "relaxed_support_entry_count": 5,
                "uma_model_sha256": "abc",
                "task_name": "omat",
                "compositions": {
                    "NaFePO4": {"reference_energy_eV_atom": -6.5},
                },
            }))
            cache = UMAHullCache(
                str(path), expected_model_sha256="abc", expected_task_name="omat"
            )
            self.assertEqual(cache.get("NaFePO4"), -6.5)
            with self.assertRaises(MissingUMAHullReferenceError):
                cache.get("Na2FePO4")
            with self.assertRaises(ValueError):
                UMAHullCache(str(path), expected_model_sha256="different")

    def test_uma_hull_build_requires_complete_relaxed_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase_path = root / "phase.json"
            pool_path = root / "pool.json"
            records_root = root / "relax"
            records_dir = records_root / "records"
            records_dir.mkdir(parents=True)
            entries = [
                ("mp-1-GGA", {"Na": 1}, 0.0),
                ("mp-2-GGA", {"Fe": 1}, 0.0),
                ("mp-3-GGA", {"P": 1}, 0.0),
                ("mp-4-GGA", {"O": 2}, 0.0),
                ("mp-5-GGA+U", {"Na": 1, "Fe": 1, "P": 1, "O": 4}, -1.0),
            ]
            phase_path.write_text(json.dumps({
                "version": "mp-phase-entries-v1",
                "compatibility": "MaterialsProject2020Compatibility",
                "queried_at": "2026-08-27T00:00:00+00:00",
                "n_compatible_entries": len(entries),
                "entries": [
                    {"entry_id": entry_id, "composition": composition,
                     "energy_eV": energy_per_atom * sum(composition.values())}
                    for entry_id, composition, energy_per_atom in entries
                ],
            }))
            pool_path.write_text(json.dumps({
                "unique_compositions": [{
                    "counts": {"Na": 1, "Fe": 1, "P": 1, "O": 4}
                }]
            }))
            phase_hash = sha256_file(phase_path)
            protocol = {"fire_steps": 10, "bfgs_steps": 10}
            for entry_id, composition, energy_per_atom in entries:
                (records_dir / f"{entry_id}.json").write_text(json.dumps({
                    "version": RELAX_RECORD_VERSION,
                    "entry_id": entry_id,
                    "returned_material_id": material_id(entry_id),
                    "composition": composition,
                    "n_atoms": int(sum(composition.values())),
                    "converged": True,
                    "final_energy_eV_atom": energy_per_atom,
                    "uma_model_sha256": "uma-hash",
                    "task_name": "omat",
                    "protocol": protocol,
                    "protocol_sha256": "protocol-hash",
                    "phase_cache_sha256": phase_hash,
                    "structure_cache_sha256": "structure-hash",
                }))
            payload = build_hull_payload(phase_path, pool_path, records_root)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["expected_support_entry_count"], 5)
            self.assertAlmostEqual(
                payload["compositions"]["NaFePO4"]["reference_energy_eV_atom"],
                -1.0,
            )
            (records_dir / "mp-5-GGA+U.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing=1"):
                build_hull_payload(phase_path, pool_path, records_root)

    def test_uma_hull_relax_shard_is_resumable(self) -> None:
        from ase.calculators.lj import LennardJones

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase_path = root / "phase.json"
            structure_path = root / "structures.json"
            checkpoint_path = root / "uma.pt"
            output = root / "relax"
            checkpoint_path.write_bytes(b"synthetic checkpoint")
            phase_path.write_text(json.dumps({
                "version": "mp-phase-entries-v1",
                "compatibility": "MaterialsProject2020Compatibility",
                "n_compatible_entries": 1,
                "entries": [{
                    "entry_id": "mp-1-GGA", "composition": {"Na": 1},
                    "energy_eV": 0.0,
                }],
            }))
            structure_path.write_text(json.dumps({
                "version": STRUCTURE_CACHE_VERSION,
                "status": "complete",
                "phase_cache_sha256": sha256_file(phase_path),
                "structures": {
                    "mp-1-GGA": {
                        "requested_material_id": "mp-1",
                        "returned_material_id": "mp-1",
                        "structure": {
                            "lattice": {
                                "matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
                                "pbc": [True, True, True],
                            },
                            "sites": [{
                                "species": [{"element": "Na", "occu": 1}],
                                "abc": [0, 0, 0],
                            }],
                        },
                    }
                },
            }))
            args = Namespace(
                phase_cache=str(phase_path), structure_cache=str(structure_path),
                uma_checkpoint=str(checkpoint_path), out_directory=str(output),
                device="cpu", task_name="omat", shard_index=0, shard_count=1,
                fire_fmax=0.05, fire_steps=2, bfgs_fmax=0.01, bfgs_steps=2,
                resume=False,
            )

            output_directories = []

            def fake_relax(atoms, calculator, output_directory, config):
                output_directories.append(output_directory)
                atoms.calc = calculator
                return SimpleNamespace(
                    atoms=atoms, converged=True, status="converged",
                    final_fmax_eV_A=0.0, fire_converged=True,
                    bfgs_converged=True,
                )

            with patch(
                "nagen.inverse.uma_hull_campaign._load_calculator",
                return_value=LennardJones(),
            ), patch(
                "nagen.inverse.uma_hull_campaign.relax_structure",
                side_effect=fake_relax,
            ):
                relax_command(args)
                args.resume = True
                relax_command(args)
            record = json.loads(
                (output / "records" / "mp-1-GGA.json").read_text()
            )
            manifest = json.loads(
                (output / "manifest_shard_000.json").read_text()
            )
            self.assertTrue(record["converged"])
            self.assertEqual(record["attempt"], 1)
            self.assertTrue(output_directories[0].endswith("attempt_001"))
            self.assertEqual(manifest["counts"]["skipped"], 1)

    def test_uma_hull_failed_relaxation_retries_in_new_log_directory(self) -> None:
        from ase.calculators.lj import LennardJones

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase_path = root / "phase.json"
            structure_path = root / "structures.json"
            checkpoint_path = root / "uma.pt"
            output = root / "relax"
            checkpoint_path.write_bytes(b"synthetic checkpoint")
            phase_path.write_text(json.dumps({
                "version": "mp-phase-entries-v1",
                "compatibility": "MaterialsProject2020Compatibility",
                "n_compatible_entries": 1,
                "entries": [{
                    "entry_id": "mp-1-GGA", "composition": {"Na": 1},
                    "energy_eV": 0.0,
                }],
            }))
            structure_path.write_text(json.dumps({
                "version": STRUCTURE_CACHE_VERSION,
                "status": "complete",
                "phase_cache_sha256": sha256_file(phase_path),
                "structures": {
                    "mp-1-GGA": {
                        "requested_material_id": "mp-1",
                        "returned_material_id": "mp-1",
                        "structure": {
                            "lattice": {
                                "matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
                                "pbc": [True, True, True],
                            },
                            "sites": [{
                                "species": [{"element": "Na", "occu": 1}],
                                "abc": [0, 0, 0],
                            }],
                        },
                    }
                },
            }))
            args = Namespace(
                phase_cache=str(phase_path), structure_cache=str(structure_path),
                uma_checkpoint=str(checkpoint_path), out_directory=str(output),
                device="cpu", task_name="omat", shard_index=0, shard_count=1,
                fire_fmax=0.05, fire_steps=2, bfgs_fmax=0.01, bfgs_steps=2,
                resume=True,
            )
            output_directories = []

            def fake_relax(atoms, calculator, output_directory, config):
                output_directories.append(output_directory)
                atoms.calc = calculator
                if len(output_directories) == 1:
                    raise OSError(5, "Input/output error")
                return SimpleNamespace(
                    atoms=atoms, converged=True, status="converged",
                    final_fmax_eV_A=0.0, fire_converged=True,
                    bfgs_converged=True,
                )

            with patch(
                "nagen.inverse.uma_hull_campaign._load_calculator",
                return_value=LennardJones(),
            ), patch(
                "nagen.inverse.uma_hull_campaign.relax_structure",
                side_effect=fake_relax,
            ):
                relax_command(args)
                relax_command(args)
            record = json.loads(
                (output / "records" / "mp-1-GGA.json").read_text()
            )
            self.assertTrue(record["converged"])
            self.assertEqual(record["attempt"], 2)
            self.assertTrue(output_directories[0].endswith("attempt_001"))
            self.assertTrue(output_directories[1].endswith("attempt_002"))

    def test_coordinate_and_lattice_finite_difference_gate(self) -> None:
        torch.manual_seed(3)
        frac = torch.randn(1, 2, 3, dtype=torch.float64)
        lattice = torch.randn(1, 3, 3, dtype=torch.float64)
        result = finite_difference_gradient_check(
            lambda x, cell: x.square().sum((1, 2)) + 0.3 * cell.square().sum((1, 2)),
            frac, lattice, epsilon=1e-5,
        )
        self.assertTrue(result.passed)

    def test_energy_scale_diagnostic_blocks_without_offset_fit(self) -> None:
        records = [
            {"composition": "NaFePO4", "mp_entry_id": f"mp-{index}",
             "E_UMA_eV_atom": -6.7, "E_MP_eV_atom": -7.0}
            for index in range(10)
        ]
        report = evaluate_zero_point(records, "abc")
        self.assertFalse(report["passed"])
        self.assertTrue(report["blocking"])
        self.assertFalse(report["experiment_may_continue"])
        self.assertFalse(report["offset_fit"])

    def test_mp_thermo_suffix_is_removed_from_material_id(self) -> None:
        self.assertEqual(material_id("mp-19028-GGA+U"), "mp-19028")
        self.assertEqual(material_id("mp-1-GGA"), "mp-1")
        self.assertEqual(material_id("mp-2"), "mp-2")

    def test_mp_structure_dictionary_converts_without_pymatgen(self) -> None:
        atoms = ase_atoms_from_structure_dict({
            "lattice": {
                "matrix": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                "pbc": [True, True, True],
            },
            "sites": [
                {"species": [{"element": "Na", "occu": 1}], "abc": [0, 0, 0]},
                {"species": [{"element": "O", "occu": 1}], "abc": [0.5, 0.5, 0.5]},
            ],
        })
        self.assertEqual(atoms.get_chemical_symbols(), ["Na", "O"])
        self.assertTrue(atoms.pbc.all())


if __name__ == "__main__":
    unittest.main()

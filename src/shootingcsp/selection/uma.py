"""UMA energy/force evaluator for auditable ASE relaxation.

This adapter deliberately exposes only detached ASE quantities.  D-Flow source
optimization uses the separate first-order VJP bridge in
``shootingcsp.inverse.uma_guidance``; final relaxation must be an independent,
reproducible calculation.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np


class UMAEvaluator:
    """Single-checkpoint UMA calculator with immutable provenance."""

    def __init__(self, checkpoint: str, device: str = "cuda", task_name: str = "omat",
                 inference_settings: str = "default"):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            from fairchem.core import FAIRChemCalculator
            from fairchem.core.units.mlip_unit import load_predict_unit
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("fairchem-core is required for UMA evaluation") from error
        settings = inference_settings
        if inference_settings in ("fixed_composition_fp32", "fixed_composition_fp32_compiled", "batch_fp32", "batch_tf32"):
            from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
            settings = InferenceSettings(tf32=inference_settings == "batch_tf32", activation_checkpointing=False,
                                         merge_mole=not inference_settings.startswith("batch_"), compile=inference_settings.endswith("_compiled"),
                                         external_graph_gen=False, internal_graph_gen_version=2)
        self.predictor = load_predict_unit(str(path), inference_settings=settings, device=device)
        self.calculator = FAIRChemCalculator(self.predictor, task_name=task_name)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.provenance = {
            "backend": "UMA",
            "fairchem_core_version": version("fairchem-core"),
            "checkpoint": str(path.resolve()),
            "state_sha256": digest.hexdigest(),
            "task_name": task_name,
            "device": device,
            "energy_unit": "eV/atom",
            "force_unit": "eV/Angstrom",
            "stress_unit": "eV/Angstrom^3",
            "inference_settings": inference_settings,
        }
        self.calls = 0

    def evaluate_batch(self, structures):
        """Full-precision batched E/F/S; inputs are independent ASE Atoms."""
        if not structures:
            return []
        if self.provenance["inference_settings"].startswith("fixed_composition"):
            raise ValueError("merged UMA does not support multi-system batches")
        from fairchem.core.datasets import data_list_collater
        data = []
        for atoms in structures:
            self.calculator._check_atoms_pbc(atoms)
            self.calculator._validate_charge_and_spin(atoms)
            data.append(self.calculator.a2g(atoms))
        prediction = self.predictor.predict(data_list_collater(data, otf_graph=True))
        energies = prediction["energy"].detach().cpu().numpy().reshape(-1)
        forces = prediction["forces"].detach().cpu().numpy()
        stresses = prediction["stress"].detach().cpu().numpy().reshape(-1, 3, 3)
        if len(energies) != len(structures) or len(forces) != sum(map(len, structures)):
            raise ValueError("invalid batched UMA output dimensions")
        results, offset = [], 0
        for atoms, energy, stress in zip(structures, energies, stresses):
            force = np.asarray(forces[offset:offset + len(atoms)], float)
            offset += len(atoms)
            if not (np.isfinite(energy) and np.isfinite(force).all() and np.isfinite(stress).all()):
                raise ValueError("nonfinite batched UMA output")
            results.append({"energy_eV_atom": float(energy) / len(atoms), "forces_eV_A": force,
                            "stress_eV_A3": np.asarray(stress, float),
                            "force_max_eV_A": float(np.linalg.norm(force, axis=1).max())})
        self.calls += 1
        return results

    def evaluate(self, elements, frac, cell):
        from ase import Atoms

        frac = np.asarray(frac, dtype=float)
        cell = np.asarray(cell, dtype=float)
        if (not elements or frac.shape != (len(elements), 3) or cell.shape != (3, 3)
                or not np.isfinite(frac).all() or not np.isfinite(cell).all()
                or abs(np.linalg.det(cell)) < 1e-6):
            raise ValueError("invalid periodic structure")
        atoms = Atoms(symbols=elements, scaled_positions=frac, cell=cell, pbc=True)
        atoms.calc = self.calculator
        return self.evaluate_atoms(atoms)

    def evaluate_atoms(self, atoms):
        """Read an attached calculator's cached outputs without rebuilding Atoms.

        Reusing FIRE's terminal Atoms avoids a redundant model evaluation and
        preserves the exact terminal coordinates used for the forces.
        """
        if atoms.calc is not self.calculator:
            raise ValueError("Atoms must use this evaluator's calculator")
        self.calls += 1
        energy = float(atoms.get_potential_energy()) / len(atoms)
        forces = np.asarray(atoms.get_forces(), dtype=float)
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
        if not (np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()):
            raise ValueError("nonfinite UMA output")
        return {
            "energy_eV_atom": energy,
            "forces_eV_A": forces,
            "stress_eV_A3": stress,
            "force_max_eV_A": float(np.linalg.norm(forces, axis=1).max()),
        }

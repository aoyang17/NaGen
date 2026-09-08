"""CHGNet NAXL evaluator and first-order force/stress VJP.

Graph conversion is internal to the evaluator; NaGen state remains N,A,X,L.
No historical dataset energy labels are consumed by this module.
"""
import hashlib
from importlib.metadata import version

import numpy as np
import torch


class _EnergyVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frac, cell, energy, grad_frac, grad_cell):
        ctx.save_for_backward(grad_frac, grad_cell)
        return energy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        grad_frac, grad_cell = ctx.saved_tensors
        return upstream * grad_frac, upstream * grad_cell, None, None, None


class CHGNetEvaluator:
    def __init__(self, device="cpu", model_name="0.3.0"):
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        self.model = CHGNet.load(model_name=model_name, use_device=device)
        self.model.eval()
        self.calculator = CHGNetCalculator(model=self.model, use_device=device,
                                          on_isolated_atoms="error")
        digest = hashlib.sha256()
        for key, tensor in sorted(self.model.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.provenance = {
            "backend": "CHGNet", "package_version": version("chgnet"),
            "model_name": model_name, "state_sha256": digest.hexdigest(),
            "device": device, "energy_unit": "eV/atom", "force_unit": "eV/Angstrom",
            "stress_unit": "eV/Angstrom^3", "higher_derivatives": False,
        }
        self.calls = 0

    @torch.enable_grad()
    def evaluate(self, elements, frac, cell):
        from ase import Atoms
        frac, cell = np.asarray(frac, float), np.asarray(cell, float)
        if (not elements or frac.shape != (len(elements), 3) or cell.shape != (3, 3)
            or not np.isfinite(frac).all() or not np.isfinite(cell).all()
            or abs(np.linalg.det(cell)) < 1e-6):
            raise ValueError("invalid NAXL structure")
        atoms = Atoms(symbols=elements, scaled_positions=frac, cell=cell, pbc=True)
        atoms.calc = self.calculator
        self.calls += 1
        # ASE returns total eV and stress in eV/A^3, unlike direct CHGNet (GPa).
        energy = float(atoms.get_potential_energy()) / len(atoms)
        forces = np.asarray(atoms.get_forces(), float)
        stress = np.asarray(atoms.get_stress(voigt=False), float)
        if not all(np.isfinite(x).all() for x in (energy, forces, stress)):
            raise ValueError("nonfinite CHGNet output")
        return {"energy_eV_atom": energy, "forces_eV_A": forces,
                "stress_eV_A3": stress,
                "force_max_eV_A": float(np.linalg.norm(forces, axis=1).max())}

    def energy_function(self, elements):
        """Return callable for one structure, fixed discrete A, batched X/L."""
        def energy(frac, cell):
            if frac.shape != (1, len(elements), 3) or cell.shape != (1, 3, 3):
                raise ValueError("expected one fixed-composition NAXL structure")
            result = self.evaluate(elements, frac[0].detach().cpu().numpy(),
                                   cell[0].detach().cpu().numpy())
            force = torch.as_tensor(result["forces_eV_A"], dtype=frac.dtype, device=frac.device)
            stress = torch.as_tensor(result["stress_eV_A3"], dtype=cell.dtype, device=cell.device)
            h = cell[0].detach()
            n = len(elements)
            grad_frac = (-force @ h.T / n).unsqueeze(0)
            grad_cell = (torch.linalg.det(h).abs() * torch.linalg.inv(h).T @ stress / n).unsqueeze(0)
            value = frac.new_tensor(result["energy_eV_atom"])
            return _EnergyVJP.apply(frac, cell, value, grad_frac, grad_cell)
        return energy

    def gradient_check(self, crystal, epsilon=2e-3, directions=3, seed=17):
        """Directional checks for both coordinates and general cell changes."""
        rng = np.random.default_rng(seed)
        f = torch.tensor(crystal.frac[None], dtype=torch.float64, requires_grad=True)
        h = torch.tensor(crystal.lattice[None], dtype=torch.float64, requires_grad=True)
        fn = self.energy_function(crystal.elements)
        gf, gh = torch.autograd.grad(fn(f, h), (f, h))
        checks = []
        for kind, tensor, grad in (("frac", f, gf), ("cell", h, gh)):
            for _ in range(directions):
                direction = torch.tensor(rng.normal(size=tensor.shape), dtype=tensor.dtype)
                direction /= direction.norm()
                with torch.no_grad():
                    if kind == "frac":
                        fd = (fn(f + epsilon*direction, h) - fn(f - epsilon*direction, h)) / (2*epsilon)
                    else:
                        fd = (fn(f, h + epsilon*direction) - fn(f, h - epsilon*direction)) / (2*epsilon)
                automatic = float((grad * direction).sum())
                finite = float(fd)
                error = abs(automatic - finite)
                checks.append({"variable": kind, "automatic": automatic, "finite_difference": finite,
                               "abs_error": error, "passed": error <= .002 + .05 * abs(finite)})
        return {"passed": all(x["passed"] for x in checks), "epsilon": epsilon,
                "checks": checks, "model": self.provenance}

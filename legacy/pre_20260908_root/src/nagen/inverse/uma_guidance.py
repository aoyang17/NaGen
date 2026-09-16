"""Differentiable UMA adapter and finite-difference acceptance gate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from ._io import sha256_file as checkpoint_sha256

EnergyFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EnergyFactory = Callable[[torch.Tensor], EnergyFunction]
SecondOrderEvaluator = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class _AttachEnergyVJP(torch.autograd.Function):
    """Attach externally evaluated first derivatives to a scalar energy."""

    @staticmethod
    def forward(
        ctx: object, frac: torch.Tensor, lattice: torch.Tensor,
        energy: torch.Tensor, grad_frac: torch.Tensor, grad_lattice: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(grad_frac, grad_lattice)  # type: ignore[attr-defined]
        return energy

    @staticmethod
    def backward(ctx: object, upstream: torch.Tensor) -> tuple[torch.Tensor, ...]:
        grad_frac, grad_lattice = ctx.saved_tensors  # type: ignore[attr-defined]
        return upstream * grad_frac, upstream * grad_lattice, None, None, None


@dataclass(frozen=True)
class GradientCheckResult:
    coordinate_cosine: float
    lattice_cosine: float
    coordinate_median_relative_error: float
    lattice_median_relative_error: float
    passed: bool


def _comparison(automatic: torch.Tensor, finite: torch.Tensor) -> tuple[float, float]:
    flat_a = automatic.detach().double().flatten()
    flat_f = finite.detach().double().flatten()
    denominator = flat_a.norm() * flat_f.norm()
    cosine = float(torch.dot(flat_a, flat_f) / denominator) if denominator > 0 else 1.0
    active = flat_f.abs() > 1e-7
    relative = (flat_a - flat_f).abs() / flat_f.abs().clamp_min(1e-7)
    median = float(relative[active].median()) if bool(active.any()) else 0.0
    return cosine, median


def finite_difference_gradient_check(
    energy_fn: EnergyFunction, frac: torch.Tensor, lattice: torch.Tensor,
    epsilon: float = 1e-4,
) -> GradientCheckResult:
    frac_var = frac.detach().clone().requires_grad_(True)
    lattice_var = lattice.detach().clone().requires_grad_(True)
    energy = energy_fn(frac_var, lattice_var).sum()
    grad_frac, grad_lattice = torch.autograd.grad(energy, (frac_var, lattice_var))
    finite_frac = torch.zeros_like(frac)
    finite_lattice = torch.zeros_like(lattice)
    for target, finite, is_frac in (
        (frac, finite_frac, True), (lattice, finite_lattice, False)
    ):
        for index in range(target.numel()):
            plus_frac, minus_frac = frac.clone(), frac.clone()
            plus_lattice, minus_lattice = lattice.clone(), lattice.clone()
            plus = plus_frac if is_frac else plus_lattice
            minus = minus_frac if is_frac else minus_lattice
            plus.flatten()[index] += epsilon
            minus.flatten()[index] -= epsilon
            finite.flatten()[index] = (
                energy_fn(plus_frac, plus_lattice).sum()
                - energy_fn(minus_frac, minus_lattice).sum()
            ) / (2.0 * epsilon)
    coordinate_cosine, coordinate_error = _comparison(grad_frac, finite_frac)
    lattice_cosine, lattice_error = _comparison(grad_lattice, finite_lattice)
    passed = bool(
        coordinate_cosine >= 0.95 and lattice_cosine >= 0.95
        and coordinate_error <= 0.10 and lattice_error <= 0.10
    )
    return GradientCheckResult(
        coordinate_cosine, lattice_cosine, coordinate_error, lattice_error, passed
    )


def load_uma_tensor_energy(
    checkpoint: str, atomic_numbers: torch.Tensor, device: str = "cuda",
    task_name: str = "omat",
) -> tuple[EnergyFunction, dict]:
    """Load the native tensor path, failing closed when fairchem is unavailable.

    Fairchem APIs have changed across releases, so the returned callable is
    intentionally constructed only for the supported ``pretrained_mlip`` API.
    No ASE/NumPy conversion is accepted for guidance.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"UMA checkpoint not found: {path}")
    try:
        from fairchem.core import pretrained_mlip  # type: ignore
        from fairchem.core.datasets.atomic_data import AtomicData  # type: ignore
    except ImportError as error:
        raise RuntimeError("fairchem-core with native tensor prediction is required") from error
    predictor = pretrained_mlip.load_predict_unit(
        str(path), inference_settings="default", device=device
    )
    numbers = atomic_numbers.detach().long().flatten().to(device)

    def energy_per_atom(frac: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
        if frac.shape[0] != 1 or lattice.shape != (1, 3, 3):
            raise ValueError("the validated native UMA bridge accepts one structure per call")
        cell = lattice.to(device)
        positions = torch.einsum("bnd,bdk->bnk", frac.to(device), cell).squeeze(0)
        atom_count = numbers.numel()
        data = AtomicData(
            pos=positions, atomic_numbers=numbers, cell=cell,
            pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
            natoms=torch.tensor([atom_count], dtype=torch.long, device=device),
            edge_index=torch.empty(2, 0, dtype=torch.long, device=device),
            cell_offsets=torch.empty(0, 3, dtype=positions.dtype, device=device),
            nedges=torch.zeros(1, dtype=torch.long, device=device),
            charge=torch.zeros(1, dtype=torch.long, device=device),
            spin=torch.zeros(1, dtype=torch.long, device=device),
            fixed=torch.zeros(atom_count, dtype=torch.long, device=device),
            tags=torch.zeros(atom_count, dtype=torch.long, device=device),
            batch=torch.zeros(atom_count, dtype=torch.long, device=device),
            dataset=task_name,
        )
        prediction = predictor.predict(data)
        energy = prediction.get("energy")
        if energy is None or not energy.requires_grad:
            raise RuntimeError("fairchem prediction severed the energy graph; G3 fails closed")
        return energy.reshape(-1) / atom_count

    return energy_per_atom, {
        "checkpoint": str(path.resolve()), "checkpoint_sha256": checkpoint_sha256(str(path)),
        "device": device, "task_name": task_name,
        "gradient_path": "fairchem-AtomicData/predictor.predict",
    }


def load_uma_calculator_vjp_energy(
    checkpoint: str, atomic_numbers: torch.Tensor, device: str = "cuda",
    task_name: str = "omat",
) -> tuple[EnergyFunction, dict]:
    """Load UMA through ASE and attach force/stress first derivatives as a VJP.

    ``FAIRChemCalculator`` intentionally returns detached NumPy values.  Forces
    supply ``-dE/dr`` and stress supplies the symmetric strain derivative.  For
    row-vector cells ``H`` and fractional coordinates ``f``, the gradients are
    ``dE/df = -F H^T`` and ``dE/dH = V H^{-T} sigma``.  Values and derivatives
    are divided by atom count to expose energy per atom.  This path supports the
    first derivative needed by source-noise optimization, not higher derivatives.
    """
    factory, metadata = load_uma_calculator_vjp_factory(
        checkpoint, device=device, task_name=task_name
    )
    return factory(atomic_numbers), metadata


def load_uma_calculator_vjp_factory(
    checkpoint: str, device: str = "cuda", task_name: str = "omat",
) -> tuple[EnergyFactory, dict]:
    """Load UMA once and construct validated VJP energy functions per composition."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"UMA checkpoint not found: {path}")
    try:
        from ase import Atoms
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
    except ImportError as error:
        raise RuntimeError("fairchem-core and ASE are required for UMA VJP") from error
    predictor = load_predict_unit(
        str(path), inference_settings="default", device=device
    )
    calculator = FAIRChemCalculator(predictor, task_name=task_name)

    def factory(atomic_numbers: torch.Tensor) -> EnergyFunction:
        numbers = atomic_numbers.detach().long().flatten().cpu().tolist()
        atom_count = len(numbers)

        def energy_per_atom(frac: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
            if frac.shape[0] != 1 or lattice.shape != (1, 3, 3):
                raise ValueError("the validated UMA VJP bridge accepts one structure per call")
            cell = lattice[0]
            positions = frac[0] @ cell
            atoms = Atoms(
                numbers=numbers,
                positions=positions.detach().double().cpu().numpy(),
                cell=cell.detach().double().cpu().numpy(),
                pbc=True,
            )
            atoms.calc = calculator
            energy = torch.as_tensor(
                atoms.get_potential_energy() / atom_count,
                dtype=frac.dtype, device=frac.device,
            )
            forces = torch.as_tensor(
                atoms.get_forces(), dtype=frac.dtype, device=frac.device,
            )
            stress = torch.as_tensor(
                atoms.get_stress(voigt=False), dtype=frac.dtype, device=frac.device,
            )
            grad_positions = -forces / atom_count
            grad_frac = (grad_positions @ cell.mT).unsqueeze(0)
            volume = torch.linalg.det(cell).abs()
            grad_lattice = (
                volume * torch.linalg.inv(cell).mT @ stress / atom_count
            ).unsqueeze(0)
            return _AttachEnergyVJP.apply(
                frac, lattice, energy, grad_frac, grad_lattice
            )

        return energy_per_atom

    return factory, {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": checkpoint_sha256(str(path)),
        "device": device,
        "task_name": task_name,
        "gradient_path": "FAIRChemCalculator force/stress VJP",
        "supports_higher_derivatives": False,
    }


def load_uma_native_second_order_evaluator(
    checkpoint: str, device: str = "cuda", task_name: str = "omat",
) -> tuple[SecondOrderEvaluator, dict]:
    """Load UMA with force/stress outputs retaining their higher-order graph.

    UMA's standard inference head computes forces with ``create_graph=False``.
    Setting only the output heads to training mode keeps the backbone in
    deterministic evaluation mode while enabling the head's documented
    higher-order derivative path.  This computes Hessian-vector products on
    demand; it deliberately does not materialize the full Hessian.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"UMA checkpoint not found: {path}")
    try:
        from fairchem.core.datasets.atomic_data import AtomicData
        from fairchem.core.units.mlip_unit import load_predict_unit
    except ImportError as error:
        raise RuntimeError("fairchem-core native prediction is required") from error
    # The default fast path uses torch.compile/AOTAutograd, which does not
    # support the double backward needed for force/stress constraint
    # Jacobians.  ``traineval`` is deliberately non-compiled and supports
    # changing structures between optimizer evaluations (older fairchem
    # releases called the corresponding preset ``batch``).
    predictor = load_predict_unit(
        str(path), inference_settings="traineval", device=str(device)
    )
    initialized = False

    def atomic_data(
        numbers: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor,
        target_device: torch.device | str,
    ):
        numbers = numbers.to(target_device)
        positions = positions.to(target_device)
        cell = cell.to(target_device)
        atom_count = numbers.numel()
        return AtomicData(
            pos=positions, atomic_numbers=numbers, cell=cell,
            pbc=torch.ones(1, 3, dtype=torch.bool, device=target_device),
            natoms=torch.tensor(
                [atom_count], dtype=torch.long, device=target_device
            ),
            edge_index=torch.empty(
                2, 0, dtype=torch.long, device=target_device
            ),
            cell_offsets=torch.empty(
                0, 3, dtype=positions.dtype, device=target_device
            ),
            nedges=torch.zeros(1, dtype=torch.long, device=target_device),
            charge=torch.zeros(1, dtype=torch.long, device=target_device),
            spin=torch.zeros(1, dtype=torch.long, device=target_device),
            fixed=torch.zeros(
                atom_count, dtype=torch.long, device=target_device
            ),
            tags=torch.zeros(
                atom_count, dtype=torch.long, device=target_device
            ),
            batch=torch.zeros(
                atom_count, dtype=torch.long, device=target_device
            ),
            dataset=task_name,
        )

    def evaluate(
        atomic_numbers: torch.Tensor, frac: torch.Tensor, lattice: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal initialized
        if frac.shape[0] != 1 or lattice.shape != (1, 3, 3):
            raise ValueError("second-order UMA evaluator accepts one structure")
        numbers = atomic_numbers.detach().long().flatten().to(device)
        cell = lattice.to(device)
        positions = torch.einsum(
            "bnd,bdk->bnk", frac.to(device), cell
        ).squeeze(0)
        atom_count = numbers.numel()
        if not initialized:
            # fairchem prepares its as-loaded CPU model before moving it to the
            # requested accelerator.  Initialize with detached CPU data, then
            # enable create_graph on the prepared output heads.
            predictor.predict(atomic_data(
                numbers.detach().cpu(), positions.detach().cpu(),
                cell.detach().cpu(), "cpu",
            ))
            initialized = True
        # The UMA EFS head uses ``self.training`` only to select create_graph
        # for force/stress autograd.  Keep the prepared backbone in eval mode.
        for head in predictor.model.module.output_heads.values():
            head.train(True)
        data = atomic_data(numbers, positions, cell, device)
        prediction = predictor.predict(data)
        energy = prediction.get("energy")
        forces = prediction.get("forces")
        stress = prediction.get("stress")
        if energy is None or forces is None or stress is None:
            raise RuntimeError("UMA prediction lacks energy, forces, or stress")
        if not forces.requires_grad or not stress.requires_grad:
            raise RuntimeError("UMA higher-order force/stress graph is detached")
        return energy.reshape(-1) / atom_count, forces, stress.reshape(-1, 3, 3)

    return evaluate, {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": checkpoint_sha256(str(path)),
        "device": device,
        "task_name": task_name,
        "inference_settings": "batch (non-compiled)",
        "gradient_path": "prepared native UMA EFS head with create_graph",
        "supports_higher_derivatives": True,
        "full_hessian_materialized": False,
    }

"""Independent ASE FIRE states sharing batched force evaluations.

Only evaluation is batched. Velocities, timestep adaptation, convergence and
step budgets remain independent for every structure; converged systems retire.
"""
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE


def fire_batch(evaluate_batch, structures, fmax=.03, steps=400, maxstep=.1):
    if fmax <= 0 or steps < 1 or maxstep <= 0:
        raise ValueError("positive relaxation parameters required")
    optimizers = [FIRE(atoms, logfile=None, maxstep=maxstep) for atoms in structures]
    first, last = [None] * len(structures), [None] * len(structures)
    active = list(range(len(structures)))
    while active:
        labels = evaluate_batch([structures[i] for i in active])
        if len(labels) != len(active):
            raise ValueError("one prediction required per active structure")
        remaining = []
        for i, label in zip(active, labels):
            atoms, optimizer = structures[i], optimizers[i]
            if first[i] is None:
                first[i] = label
            last[i] = label
            atoms.calc = SinglePointCalculator(atoms, energy=label["energy_eV_atom"] * len(atoms),
                                               forces=label["forces_eV_A"], stress=label["stress_eV_A3"])
            if label["force_max_eV_A"] <= fmax or optimizer.nsteps >= steps:
                continue
            optimizer.step()
            optimizer.nsteps += 1
            remaining.append(i)
        active = remaining
    return [{"before": before, "after": after, "steps": optimizer.nsteps,
             "converged": after["force_max_eV_A"] <= fmax}
            for before, after, optimizer in zip(first, last, optimizers)]

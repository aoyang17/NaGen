"""Paired fixed-cell relaxation diagnostic; not a generation benchmark."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from ase import Atoms
from ase.optimize import FIRE

from .audit import load_crystals
from .energy import CHGNetEvaluator
from .experiment import json_default, sha256
from .pipeline import Crystal, Conditioning, hard_gates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--fmax', type=float, default=.03)
    args = p.parse_args()
    if args.steps < 1 or args.fmax <= 0:
        p.error('positive steps and fmax required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    evaluator = CHGNetEvaluator(args.device)
    profile = json.loads(Path(args.profile).read_text())
    start = time.monotonic()
    results = []
    with (out/'pairs.jsonl').open('x') as handle, (out/'relaxed.jsonl').open('x') as structures:
        for c in load_crystals(args.input):
            row = {'id': c.id, 'status': 'failed'}
            try:
                condition = Conditioning(dict(Counter(c.elements)), c.family)
                before = evaluator.evaluate(c.elements, c.frac, c.lattice)
                atoms = Atoms(symbols=c.elements, scaled_positions=c.frac,
                              cell=c.lattice, pbc=True)
                atoms.calc = evaluator.calculator
                opt = FIRE(atoms, logfile=None, maxstep=.1)
                opt.run(fmax=args.fmax, steps=args.steps)
                relaxed = Crystal(c.id, c.elements, atoms.get_scaled_positions(),
                                  np.asarray(atoms.cell), c.family)
                after = evaluator.evaluate(relaxed.elements, relaxed.frac, relaxed.lattice)
                checks = {}
                for name, crystal, label in [('before', c, before), ('after', relaxed, after)]:
                    checks[name] = hard_gates(crystal, condition, profile,
                                             forces=label['forces_eV_A'], motif_backend='native')
                row.update(status='completed', steps=opt.nsteps,
                           converged=after['force_max_eV_A'] <= args.fmax,
                           before=before, after=after, gates=checks,
                           delta_energy_eV_atom=after['energy_eV_atom']-before['energy_eV_atom'])
                structures.write(json.dumps({'material_id': c.id, 'A': c.elements,
                    'X_frac': relaxed.frac, 'L_matrix': relaxed.lattice,
                    'family': c.family}, default=json_default, allow_nan=False)+'\n')
                structures.flush()
            except Exception as error:
                row['error'] = f'{type(error).__name__}: {error}'
            results.append(row)
            handle.write(json.dumps(row, default=json_default, allow_nan=False)+'\n')
            handle.flush()
            print(json.dumps({'id': c.id, 'status': row['status'], 'steps': row.get('steps')}), flush=True)
    good = [r for r in results if r['status'] == 'completed']
    summary = {'configuration': vars(args), 'input_sha256': sha256(args.input),
               'profile_sha256': sha256(args.profile), 'model': evaluator.provenance,
               'completed': len(good), 'failed': len(results)-len(good),
               'converged': sum(r['converged'] for r in good),
               'elapsed_seconds': time.monotonic()-start,
               'note': 'Fixed-cell diagnostic on existing structures; no hull or generation claim. '
                       'Evaluator calls exclude ASE optimizer calculator calls.',
               'force_threshold_counts': {phase: {str(t): sum(
                   r[phase]['force_max_eV_A'] <= t for r in good)
                   for t in [.03, .05, .1]} for phase in ['before', 'after']},
               'all_gates_passed': {phase: sum(r['gates'][phase]['passed'] for r in good)
                                    for phase in ['before', 'after']}}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    if summary['failed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

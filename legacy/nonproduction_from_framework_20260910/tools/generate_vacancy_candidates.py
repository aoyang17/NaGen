"""Generate unseen fixed-composition Na-vacancy orderings from a TRAIN prototype."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import random

from pymatgen.core import Structure

from nagen.selection.audit import load_crystals
from nagen.selection.experiment import json_default, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--parent', required=True)
    p.add_argument('--target-na', type=int, required=True)
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.target_na < 0 or args.count < 1:
        p.error('invalid target/count')
    raw = {r['material_id']: r for r in map(json.loads, Path(args.train).read_text().splitlines())}
    rows = [c for c in load_crystals(args.train) if raw[c.id].get('parent_group') == args.parent]
    if not rows:
        raise ValueError('parent absent from training split')
    prototypes = sorted(rows, key=lambda c: (-c.elements.count('Na'), c.id))
    prototype = prototypes[0]
    na_sites = [i for i,e in enumerate(prototype.elements) if e == 'Na']
    if args.target_na >= len(na_sites):
        raise ValueError('target Na must be below fully occupied prototype')
    choices = list(combinations(na_sites, len(na_sites)-args.target_na))
    random.Random(args.seed).shuffle(choices)
    accepted = []
    for removed in choices:
        keep = [i for i in range(len(prototype.elements)) if i not in removed]
        candidate = Structure(prototype.lattice, [prototype.elements[i] for i in keep],
                              prototype.frac[keep])
        accepted.append({'material_id': f'vacancy_seed{args.seed}_{len(accepted):04d}',
                         'A': [s.specie.symbol for s in candidate],
                         'X_frac': candidate.frac_coords,
                         'L_matrix': candidate.lattice.matrix, 'family': 'phosphate',
                         'generation': {'method': 'train_prototype_Na_vacancy_enumeration',
                            'prototype_id': prototype.id, 'removed_prototype_site_indices': list(removed)}})
        if len(accepted) == args.count:
            break
    if not accepted:
        raise RuntimeError('no unseen vacancy configurations found')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with out.open('x') as handle:
        for row in accepted:
            handle.write(json.dumps(row, default=json_default)+'\n')
    out.with_suffix('.manifest.json').write_text(json.dumps({'status': 'completed',
        'count': len(accepted), 'requested': args.count, 'configuration': vars(args),
        'train_sha256': sha256(args.train), 'prototype_id': prototype.id,
        'prototype_Na': len(na_sites), 'enumerated_combinations': len(choices),
        'novelty_check': 'Deferred until after CHGNet relaxation; only distinct discrete vacancy index sets here.',
        'note': 'Generated from a training prototype by discrete Na vacancy ordering. '
                'No test/validation structure, energy label, or relaxed candidate used. '
                'Post-relax StructureMatcher novelty is mandatory before selection.'}, indent=2)+'\n')
    print(json.dumps({'accepted': len(accepted), 'prototype': prototype.id,
                      'possible_combinations': len(choices)}))


if __name__ == '__main__':
    main()

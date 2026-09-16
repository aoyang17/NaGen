"""Exhaustive same-composition StructureMatcher checks against earlier splits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core import Structure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    refs = defaultdict(list)
    matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5,
                               primitive_cell=True, scale=True, attempt_supercell=True,
                               comparator=ElementComparator())
    blocked, attempts = [], 0
    started = time.monotonic()
    with (out/'matches.jsonl').open('x') as log:
        for split in ['train', 'val', 'test']:
            rows = [json.loads(line) for line in (Path(args.data)/f'{split}.jsonl').open()]
            staged = []
            for i, row in enumerate(rows):
                structure = Structure.from_dict(row['structure'])
                key = structure.composition.reduced_formula
                if split != 'train':
                    for ref_id, ref in refs[key]:
                        attempts += 1
                        if matcher.fit(structure, ref):
                            blocked.append(row['material_id'])
                            log.write(json.dumps({'id': row['material_id'], 'split': split,
                                                   'matched_earlier_id': ref_id})+'\n')
                            log.flush()
                            break
                staged.append((key, row['material_id'], structure))
                if (i+1) % 100 == 0:
                    print(f'{split} {i+1}/{len(rows)}, matches={len(blocked)}, comparisons={attempts}', flush=True)
            # Test is checked against all original validation structures, including
            # excluded ones, conservatively preserving the tuning/test separation.
            for key, identifier, structure in staged:
                refs[key].append((identifier, structure))
    result = {'status': 'completed', 'excluded_heldout_ids': blocked,
              'comparisons': attempts, 'elapsed_seconds': time.monotonic()-started,
              'matcher': matcher.as_dict(),
              'note': 'Training unchanged; matched val/test IDs must be excluded from benchmark. '
                      'This is configured near-equivalence, not mathematical structure identity.'}
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'excluded': len(blocked), 'comparisons': attempts}))


if __name__ == '__main__':
    main()

"""Validate a complete raw dataset and split target records by parent identifier.

Parent grouping is a conservative ID heuristic, NOT crystallographic equivalence.
Unknown ID formats are quarantined from the benchmark rather than split randomly.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def parent_group(identifier):
    match = re.match(r'^(.*?)_Fe_Na\d+_', identifier)
    return match.group(1) if match else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--expected-bytes', type=int, default=529741128)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--split-method', choices=['hash', 'balanced-parent'], default='balanced-parent')
    args = p.parse_args()
    source = Path(args.input)
    if source.stat().st_size != args.expected_bytes:
        p.error('input size does not match full dataset; refusing a partial-data benchmark')
    rows, seen = [], set()
    total = 0
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for line in handle:
            digest.update(line)
            r = json.loads(line)
            total += 1
            sites = r['structure']['sites']
            elements = {s['element'] for site in sites for s in site['species']}
            if not ({'Fe', 'P', 'O'} <= elements <= {'Na', 'Fe', 'P', 'O'}):
                continue
            identifier = r['material_id']
            if identifier in seen:
                raise ValueError(f'duplicate material ID: {identifier}')
            seen.add(identifier)
            group = parent_group(identifier)
            ordered = all(len(s['species']) == 1 and s['species'][0].get('occu', 1) == 1 for s in sites)
            if not group or not ordered:
                split = 'quarantine'
            else:
                value = int(hashlib.sha256(f'{args.seed}:{group}'.encode()).hexdigest()[:8], 16) / 2**32
                split = 'train' if value < .7 else 'val' if value < .85 else 'test'
            rows.append({'material_id': identifier, 'structure': r['structure'],
                         'parent_group': group, 'split': split})
    if args.split_method == 'balanced-parent':
        groups = Counter(r['parent_group'] for r in rows if r['split'] != 'quarantine')
        targets = dict(zip(['train', 'val', 'test'], [sum(groups.values())*f for f in [.7, .15, .15]]))
        allocated = Counter()
        assignment = {}
        for group in sorted(groups, key=lambda g: (-groups[g], hashlib.sha256(f'{args.seed}:{g}'.encode()).hexdigest())):
            split = max(targets, key=lambda s: targets[s]-allocated[s])
            assignment[group] = split
            allocated[split] += groups[group]
        for r in rows:
            if r['split'] != 'quarantine':
                r['split'] = assignment[r['parent_group']]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for split in ['train', 'val', 'test', 'quarantine']:
        with (out/f'{split}.jsonl').open('x') as handle:
            for r in rows:
                if r['split'] == split:
                    handle.write(json.dumps(r)+'\n')
    with (out/'target.jsonl').open('x') as handle:
        for r in rows:
            if all(len(s['species']) == 1 and s['species'][0].get('occu', 1) == 1
                   for s in r['structure']['sites']):
                handle.write(json.dumps(r)+'\n')
    manifest = {'input_sha256': digest.hexdigest(), 'input_bytes': source.stat().st_size,
                'raw_records': total, 'target_records': len(rows), 'seed': args.seed,
                'split_method': args.split_method,
                'counts': dict(Counter(r['split'] for r in rows)),
                'parent_groups': {split: len({r['parent_group'] for r in rows
                                 if r['split'] == split}) for split in ['train', 'val', 'test']},
                'note': 'Historical energies discarded. Parent-ID split still requires '
                        'cross-split StructureMatcher audit before scientific claims.'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

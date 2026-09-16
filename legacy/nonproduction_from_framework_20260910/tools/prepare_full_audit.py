"""Train-only motif calibration, all-target geometry audit, balanced GPU shards."""
import argparse
from collections import Counter
import json
from pathlib import Path

from nagen.selection.audit import load_crystals
from nagen.selection.experiment import sha256
from nagen.selection.pipeline import Conditioning, calibrate, hard_gates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--shards', type=int, default=8)
    args = p.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    train = list(load_crystals(data/'train.jsonl'))
    profile = calibrate(train, min_structures=20)
    profile['source_sha256'] = sha256(data/'train.jsonl')
    (out/'profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    print(json.dumps({'status': 'profile_ready', 'train': len(train), 'profiles': profile['profiles']}), flush=True)
    raw = {r['material_id']: r for r in map(json.loads, (data/'target.jsonl').read_text().splitlines())}
    checks, fe_cn, p_cn, valences = Counter(), Counter(), Counter(), Counter()
    shard_rows = [[] for _ in range(args.shards)]
    shard_atoms = [0]*args.shards
    crystals = list(load_crystals(data/'target.jsonl'))
    # Largest-first balancing by N^2, not input order or predicted energy.
    for c in sorted(crystals, key=lambda c: (-len(c.elements), c.id)):
        shard = min(range(args.shards), key=lambda i: shard_atoms[i])
        shard_rows[shard].append(raw[c.id])
        shard_atoms[shard] += len(c.elements)**2
    for i, rows in enumerate(shard_rows):
        with (out/f'shard_{i:02d}.jsonl').open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row)+'\n')
    relax_ids = set()
    with (out/'geometry.jsonl').open('x') as handle:
        for i, c in enumerate(crystals):
            row = hard_gates(c, Conditioning(dict(Counter(c.elements)), c.family),
                             profile, motif_backend='native')
            row['split'] = raw[c.id]['split']
            row['parent_group'] = raw[c.id]['parent_group']
            handle.write(json.dumps(row, allow_nan=False)+'\n')
            checks.update(k for k, v in row['checks'].items() if v)
            if all(v for k, v in row['checks'].items() if k != 'final_force'):
                relax_ids.add(c.id)
            valences[str(row['charge_model']['required_Fe_average'])] += 1
            fe_cn.update(str(s['cn']) for s in row['sites'] if s['element'] == 'Fe')
            p_cn.update(str(s['cn']) for s in row['sites'] if s['element'] == 'P')
            if (i+1) % 200 == 0:
                handle.flush()
                print(f'geometry {i+1}/{len(crystals)}', flush=True)
    for i, rows in enumerate(shard_rows):
        with (out/f'relax_shard_{i:02d}.jsonl').open('x') as handle:
            for row in rows:
                if row['material_id'] in relax_ids:
                    handle.write(json.dumps(row)+'\n')
    summary = {'count': len(crystals), 'training_count': len(train),
               'passed_counts': dict(checks), 'all_target_Fe_CN_sites': dict(fe_cn),
               'all_target_P_CN_sites': dict(p_cn), 'required_Fe_average_counts': dict(valences),
               'shard_counts': [len(r) for r in shard_rows],
               'relax_eligible': len(relax_ids),
               'profile_sha256': sha256(out/'profile.json'),
               'note': 'No force labels in geometry audit. Only training structures calibrate gates. '
                       'Distinct IDs are not independent parent structures; cross-split matching pending.'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

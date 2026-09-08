"""Verify and merge retained plus newly completed single-point shards only."""
import argparse
import json
from pathlib import Path

from nagen.selection.experiment import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first-run', required=True)
    p.add_argument('--remaining-run', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rows, manifests, seen, models = [], [], set(), set()
    for i in range(8):
        directory = Path(args.first_run if i < 4 else args.remaining_run)/f'shard_{i:02d}'
        source = Path(args.audit)/f'shard_{i:02d}.jsonl'
        manifest = json.loads((directory/'labels.manifest.json').read_text())
        expected = {json.loads(line)['material_id'] for line in source.open()}
        shard = [json.loads(line) for line in (directory/'labels.jsonl').open()]
        ids = {r['id'] for r in shard}
        if (manifest['failed'] or manifest['input_sha256'] != sha256(source)
            or ids != expected or len(ids) != len(shard) or seen & ids
            or manifest['scored'] != len(shard)):
            raise ValueError(f'incomplete/inconsistent shard {i}')
        if any(r['status'] != 'scored' or r['model_hash'] != manifest['model']['state_sha256'] for r in shard):
            raise ValueError(f'unverified labels in shard {i}')
        seen.update(ids)
        models.add(manifest['model']['state_sha256'])
        rows.extend(shard)
        manifests.append(manifest)
    if len(models) != 1:
        raise ValueError('mixed potential models')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    with (out/'labels.jsonl').open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
    result = {'status': 'completed', 'count': len(rows), 'model': manifests[0]['model'],
              'force_threshold_counts': {str(t): sum(r['force_max_eV_A'] <= t for r in rows) for t in [.03, .05, .1]},
              'shard_manifests': manifests,
              'note': 'Single-point evaluation of unchanged input geometry only. '
                      'No relaxed geometries or historical energies used; energy is not E_hull.'}
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'shard_manifests'}, indent=2))


if __name__ == '__main__':
    main()

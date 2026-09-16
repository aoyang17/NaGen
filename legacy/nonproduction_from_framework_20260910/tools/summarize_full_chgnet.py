"""Aggregate a complete all-target experiment and paired same-pool ranking ablations."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from nagen.selection.audit import load_crystals
from nagen.selection.diversity import descriptor, distances
from nagen.selection.experiment import json_default, sha256
from nagen.selection.pipeline import robust_rank, diversity_select


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def unique_map(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate IDs in aggregation')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--matching', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--shards', type=int, default=8)
    args = p.parse_args()
    root, out = Path(args.run), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    data, audit = Path(args.data), Path(args.audit)
    metadata = unique_map(read_rows(data/'target.jsonl'), 'material_id')
    geometry = unique_map(read_rows(audit/'geometry.jsonl'), 'id')
    matching = json.loads(Path(args.matching).read_text())
    if matching['status'] != 'completed':
        raise ValueError('split matching is incomplete')
    excluded = set(matching['excluded_heldout_ids'])
    labels, pairs, structures, hashes, manifests = [], [], [], set(), []
    for i in range(args.shards):
        shard = root/f'shard_{i:02d}'
        manifest = json.loads((shard/'labels.manifest.json').read_text())
        relaxation = json.loads((shard/'relax/summary.json').read_text())
        if manifest['failed'] or relaxation['failed']:
            raise ValueError('resolve shard evaluation failures before aggregation')
        if manifest['input_sha256'] != sha256(audit/f'shard_{i:02d}.jsonl'):
            raise ValueError('single-point input hash mismatch')
        if relaxation['input_sha256'] != sha256(audit/f'relax_shard_{i:02d}.jsonl'):
            raise ValueError('relaxation input hash mismatch')
        hashes.update([manifest['model']['state_sha256'], relaxation['model']['state_sha256']])
        if relaxation['profile_sha256'] != sha256(audit/'profile.json'):
            raise ValueError('mismatched frozen motif profile')
        labels.extend(read_rows(shard/'labels.jsonl'))
        pairs.extend(read_rows(shard/'relax/pairs.jsonl'))
        structures.extend(read_rows(shard/'relax/relaxed.jsonl'))
        manifests.append({'score': manifest, 'relax': relaxation})
    labeled = unique_map(labels, 'id')
    paired = unique_map(pairs, 'id')
    if len(hashes) != 1 or set(labeled) != set(metadata):
        raise ValueError('model mismatch or incomplete target labeling')
    if any(r['status'] != 'scored' or r['model_hash'] not in hashes for r in labels):
        raise ValueError('unverified labels in shard files')
    eligible = {key for key, row in geometry.items()
                if all(v for k, v in row['checks'].items() if k != 'final_force')}
    if set(paired) != eligible:
        raise ValueError('relaxation coverage does not match frozen eligibility')
    before_checks = {}
    for key, row in labeled.items():
        checks = dict(geometry[key]['checks'])
        checks['final_force'] = row['force_max_eV_A'] <= .05
        before_checks[key] = checks
    out.mkdir(parents=True)
    for name, rows in [('pairs.jsonl', pairs), ('relaxed.jsonl', structures)]:
        with (out/name).open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
    summary = {'status': 'aggregated', 'model': manifests[0]['score']['model'],
               'target_count': len(metadata), 'scored': len(labeled),
               'relaxed': len(paired), 'converged_to_003': sum(r['converged'] for r in pairs),
               'hard_pass_before': sum(all(c.values()) for c in before_checks.values()),
               'hard_pass_after': sum(r['gates']['after']['passed'] for r in pairs),
               'before_passed_counts': dict(Counter(k for c in before_checks.values() for k,v in c.items() if v)),
               'after_passed_counts': dict(Counter(k for r in pairs for k,v in r['gates']['after']['checks'].items() if v)),
               'excluded_leakage_ids': sorted(excluded), 'shard_manifests': manifests,
               'data_sha256': sha256(data/'target.jsonl'),
               'note': 'Existing-structure screening, not NaGen generation. CHGNet energy is NOT E_hull. '
                       'After statistics cover only cheap-gate-eligible structures. '
                       'No stability claim; no thresholds tuned against test outcomes.'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    crystals = {c.id: c for c in load_crystals(out/'relaxed.jsonl')}
    metrics = ['energy_eV_atom']+[f'{e}_{m}' for e in ['P', 'Fe'] for m in
               ['off_center', 'bond_distortion', 'angle_distortion', 'coordination_rarity']]
    ranked_rows = []
    for r in pairs:
        key = r['id']
        if not r['gates']['after']['passed'] or metadata[key]['split'] not in ['val', 'test'] or key in excluded:
            continue
        c = crystals[key]
        ranked_rows.append({'id': key, 'passed': True,
            'group': repr((metadata[key]['split'], c.family, c.composition)),
            'metrics': {**r['gates']['after']['metrics'], 'energy_eV_atom': r['after']['energy_eV_atom']}})
    groups = robust_rank(ranked_rows, metrics)
    comparisons = {}
    if groups:
        train = list(load_crystals(data/'train.jsonl'))
        features = []
        for i, c in enumerate(train):
            features.append(descriptor(c))
            if (i+1) % 200 == 0:
                print(f'train descriptors {i+1}/{len(train)}', flush=True)
        train_features = np.stack(features)
        for group, ranked in groups.items():
            strategies = {'minimax': ranked,
                          'energy': sorted(ranked, key=lambda r: (r['metrics']['energy_eV_atom'], r['id']))}
            comparisons[group] = {'pool_count': len(ranked), 'strategies': {}}
            for name, ordering in strategies.items():
                pool = ordering[:32]
                feature = np.stack([descriptor(crystals[r['id']]) for r in pool])
                nearest = distances(feature, train_features).min(axis=1)
                pair_distance = distances(feature, feature)
                chosen = diversity_select(pool, pair_distance, nearest, 8, len(pool), .02)
                top = ordering[:8]
                comparisons[group]['strategies'][name] = {
                    'ranked_ids': [r['id'] for r in ordering],
                    'top8_without_diversity': [r['id'] for r in top],
                    'selected_with_diversity': [r['id'] for r in chosen],
                    'quality_pool_nearest_train': dict(zip([r['id'] for r in pool], nearest.tolist())),
                    'top8_metric_medians': {m: float(np.median([r['metrics'][m] for r in top])) for m in metrics},
                    'top8_worst_percentile_median': float(np.median([max(r['percentiles'].values()) for r in top]))}
    with (out/'ranking_ablation.json').open('x') as handle:
        json.dump({'groups': comparisons, 'metrics': metrics, 'k': 8, 'quality_pool': 32,
                   'duplicate_threshold': .02, 'same_feasible_pool_per_group': True,
                   'note': 'Radial-distance selection is not a strict uniqueness proof; '
                           'percentiles are empirical within the identical feasible pool.'}, handle, indent=2)
    summary['status'] = 'completed'
    summary['ranking_groups'] = len(groups)
    summary['ranking_candidates'] = len(ranked_rows)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k,v in summary.items() if k not in ['shard_manifests', 'excluded_leakage_ids']}, default=json_default))


if __name__ == '__main__':
    main()

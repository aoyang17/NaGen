"""Four-layer selection of newly generated endpoints, never training relaxation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher

from nagen.selection.audit import load_crystals
from nagen.selection.diversity import descriptor, distances
from nagen.selection.pipeline import robust_rank, diversity_select
from nagen.selection.experiment import json_default, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--expected-runs', type=int, default=9)
    args = p.parse_args()
    directories = sorted(p.parent for p in Path(args.runs).glob('*/summary.json'))
    if len(directories) != args.expected_runs:
        raise ValueError('incomplete generation experiment')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    reports, crystals, summaries = [], {}, []
    hashes, profile_hashes = set(), set()
    counts = Counter()
    for directory in directories:
        summary = json.loads((directory/'summary.json').read_text())
        if summary['status'] != 'completed':
            raise ValueError('unfinished pilot run')
        hashes.add((summary['flow_sha256'], summary['model']['state_sha256']))
        profile_hashes.add(summary['profile_sha256'])
        summaries.append(summary)
        for c in load_crystals(directory/'candidates.jsonl'):
            if c.id in crystals:
                raise ValueError('candidate IDs collide across conditions/seeds')
            crystals[c.id] = c
        for row in map(json.loads, (directory/'evaluations.jsonl').read_text().splitlines()):
            counts[f'{row["arm"]}_attempted_endpoints'] += 1
            if row['status'] == 'evaluated':
                counts[f'{row["arm"]}_evaluated'] += 1
                for check, passed in row['gate']['checks'].items():
                    counts[f'{row["arm"]}_pass_{check}'] += int(passed)
                if row['gate']['passed']:
                    c = crystals[row['id']]
                    reports.append({**row['gate'], 'arm': row['arm'],
                        'group': repr((row['arm'], c.family, c.composition))})
                    counts[f'{row["arm"]}_hard_pass'] += 1
    if len(hashes) != 1 or len(profile_hashes) != 1:
        raise ValueError('mixed generator/potential/profile provenance')
    metrics = ['energy_eV_atom']+[f'{e}_{m}' for e in ['P','Fe'] for m in
        ['off_center','bond_distortion','angle_distortion','coordination_rarity']]
    groups = robust_rank(reports, metrics)
    selections, selected = {}, []
    if groups:
        train = list(load_crystals(args.train))
        train_features = np.stack([descriptor(c) for c in train])
        matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5, attempt_supercell=True)
        def structure(c):
            return Structure(c.lattice, c.elements, c.frac)
        train_structures = {}
        for c in train:
            train_structures.setdefault(c.composition, []).append(structure(c))
        for group, ranked in groups.items():
            pool = ranked[:32]
            features = np.stack([descriptor(crystals[r['id']]) for r in pool])
            nearest = distances(features, train_features).min(axis=1)
            chosen = diversity_select(pool, distances(features, features), nearest, 8, len(pool), .02)
            accepted, accepted_structures = [], []
            for row in chosen:
                c = crystals[row['id']]
                candidate = structure(c)
                if any(matcher.fit(candidate, ref) for ref in train_structures.get(c.composition, [])):
                    continue
                if any(matcher.fit(candidate, ref) for ref in accepted_structures):
                    continue
                accepted.append(row['id'])
                accepted_structures.append(candidate)
                selected.append(row)
            energy_order = sorted(ranked, key=lambda r: (r['metrics']['energy_eV_atom'], r['id']))
            selections[group] = {'pool_count': len(ranked),
                'minimax_order': [r['id'] for r in ranked], 'energy_order': [r['id'] for r in energy_order],
                'selected_after_radial_and_matcher': accepted,
                'nearest_training_distances': dict(zip([r['id'] for r in pool], nearest.tolist()))}
    cif_dir = out/'selected_CIF'
    cif_dir.mkdir()
    cif_manifest = []
    for row in selected:
        c = crystals[row['id']]
        path = cif_dir/f'{c.id}.cif'
        original = Structure(c.lattice, c.elements, c.frac)
        CifWriter(original, symprec=None, significant_figures=10).write_file(path)
        if not StructureMatcher(ltol=1e-6, stol=1e-5, angle_tol=1e-5,
                                primitive_cell=False, scale=False).fit(original, Structure.from_file(path)):
            raise ValueError('CIF roundtrip verification failed')
        cif_manifest.append({'id': c.id, 'file': path.name, 'sha256': sha256(path), 'metrics': row['metrics'], 'arm': row['arm']})
    (cif_dir/'manifest.json').write_text(json.dumps(cif_manifest, indent=2)+'\n')
    (cif_dir/'README.txt').write_text('New generated candidates passing frozen hard gates, minimax quality pool,\n'
        'radial diversity selection and configured StructureMatcher checks.\n'
        'No terminal relaxation; CHGNet energy is not a complete thermodynamic E_hull.\n')
    result = {'status': 'completed', 'counts': dict(counts), 'selected_count': len(selected),
        'selection': selections, 'run_summaries': summaries, 'train_sha256': sha256(args.train),
        'metrics': metrics, 'note': 'Same feasible pool for energy/minimax ordering within each arm/composition. '
        'Report failures and actual potential calls; trajectory endpoints are not independent samples. '
        'One generator seed; three source seeds do not establish training-seed robustness.'}
    (out/'summary.json').write_text(json.dumps(result, indent=2, default=json_default)+'\n')
    print(json.dumps({'counts': dict(counts), 'selected_count': len(selected)}))


if __name__ == '__main__':
    main()

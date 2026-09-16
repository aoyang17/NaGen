"""Rank and diversity-select relaxed NEW candidates with post-relax novelty checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from nagen.selection.audit import load_crystals
from nagen.selection.diversity import descriptor, distances
from nagen.selection.pipeline import robust_rank, diversity_select
from nagen.selection.experiment import sha256
from nagen.selection.hull import ReferenceHull


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--generated', required=True, help='original Flow-generated JSONL with provenance')
    p.add_argument('--hull-labels', required=True, help='frozen same-model finite reference labels')
    p.add_argument('--max-reference-hull', type=float, default=.15)
    args = p.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    pairs = [json.loads(l) for l in (run/'pairs.jsonl').read_text().splitlines()]
    generated = {r['material_id']:r for r in map(json.loads,Path(args.generated).read_text().splitlines())}
    if {r['id'] for r in pairs} != set(generated):
        raise ValueError('relaxation records do not exactly match generated candidates')
    reference_records = [json.loads(l) for l in Path(args.hull_labels).read_text().splitlines()]
    run_summary=json.loads((run/'summary.json').read_text())
    model_hash=run_summary['model']['state_sha256']
    if model_hash != reference_records[0]['model_hash']:
        raise ValueError('candidate and finite-reference energies use different model checkpoints')
    hull = ReferenceHull(reference_records, model_hash)
    crystals = {c.id:c for c in load_crystals(run/'relaxed.jsonl')}
    feasible = []
    for r in pairs:
        if r['status'] == 'completed' and r['gates']['after']['passed']:
            hull_result=hull.evaluate(crystals[r['id']].elements,r['after']['energy_eV_atom'])
            ehull=hull_result['e_above_reference_hull_eV_atom']
            if ehull is None or ehull > args.max_reference_hull: continue
            metrics = {**r['gates']['after']['metrics'], 'reference_hull_eV_atom':ehull}
            feasible.append({**r['gates']['after'], 'group': repr(crystals[r['id']].composition),
                             'metrics': metrics, 'reference_hull':hull_result})
    metric_names = ['reference_hull_eV_atom']+[f'{e}_{m}' for e in ['P','Fe'] for m in
        ['off_center','bond_distortion','angle_distortion','coordination_rarity']]
    ranked = robust_rank(feasible, metric_names)
    train = list(load_crystals(args.train))
    matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5, primitive_cell=True,
                               scale=True, attempt_supercell=True)
    train_by_comp = {}
    for c in train:
        train_by_comp.setdefault(c.composition, []).append(Structure(c.lattice,c.elements,c.frac))
    chosen_rows, report = [], {}
    for group, rows in ranked.items():
        # Ranking is computed over every physically feasible, threshold-passing
        # candidate before novelty is consulted. Diversity then operates on the
        # full quality-ordered set; the hull threshold, rather than an arbitrary
        # top-n truncation, defines acceptable continuous quality.
        composition = crystals[rows[0]['id']].composition
        relevant_train = [c for c in train if c.composition == composition]
        if not relevant_train:
            raise ValueError('no same-composition training reference for novelty')
        train_features = np.stack([descriptor(c) for c in relevant_train])
        train_structures=train_by_comp.get(composition,[])
        eligible=[]; rejected={'exact_training_pattern':0,'radial_near_duplicate':0,'structure_matcher':0}
        eligible_features=[]
        for row in rows:
            if generated[row['id']]['generation']['exact_training_pattern']:
                rejected['exact_training_pattern']+=1; continue
            feature=descriptor(crystals[row['id']])
            nearest=float(distances(np.stack([feature]),train_features).min())
            if nearest < 1e-4:
                rejected['radial_near_duplicate']+=1; continue
            c=crystals[row['id']]; s=Structure(c.lattice,c.elements,c.frac)
            if any(matcher.fit(s,t) for t in train_structures):
                rejected['structure_matcher']+=1; continue
            eligible.append(row); eligible_features.append(feature)
        # 1e-4 is frozen from TRAIN same-composition nearest-neighbour distances:
        # numerical/symmetry duplicates cluster near zero, while the median is
        # 2.42e-3. The old 0.02 exceeded most legitimate train variations.
        if eligible:
            features=np.stack(eligible_features)
            nearest=distances(features,train_features).min(axis=1)
            radial=diversity_select(eligible,distances(features,features),nearest,8,len(eligible),1e-4)
        else:
            radial=[]
        accepted, accepted_structures = [], []
        for row in radial:
            c = crystals[row['id']]; s = Structure(c.lattice,c.elements,c.frac)
            if any(matcher.fit(s,t) for t in accepted_structures):
                continue
            accepted.append(row); accepted_structures.append(s)
        chosen_rows.extend(accepted)
        report[group] = {'feasible_and_threshold_passing':len(rows),
                         'novelty_eligible':len(eligible),'novelty_rejected':rejected,
                         'radial_survivors':len(radial),
                         'selected_ids':[r['id'] for r in accepted],
                         'reference_hull_order':[r['id'] for r in sorted(rows,key=lambda r:r['metrics']['reference_hull_eV_atom'])],
                         'minimax_order':[r['id'] for r in rows]}
    cif = out/'CIF'; cif.mkdir()
    manifest=[]
    for row in chosen_rows:
        c=crystals[row['id']]; path=cif/f'{c.id}.cif'
        s=Structure(c.lattice,c.elements,c.frac); CifWriter(s,significant_figures=10).write_file(path)
        if not StructureMatcher(ltol=1e-6,stol=1e-5,angle_tol=1e-5,primitive_cell=False,scale=False).fit(s,Structure.from_file(path)):
            raise ValueError('CIF roundtrip failed')
        manifest.append({'id':c.id,'file':path.name,'sha256':sha256(path),'metrics':row['metrics'],
                         'reference_hull':row['reference_hull'],'generation':generated[c.id]['generation']})
    (cif/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result={'status':'completed','attempted':len(pairs),'relax_completed':sum(r['status']=='completed' for r in pairs),
            'hard_feasible':len(feasible),'selected':len(chosen_rows),'selection':report,
            'radial_duplicate_threshold':1e-4,
            'max_reference_hull_eV_atom':args.max_reference_hull,
            'finite_reference_sha256':hull.hash,
            'note':'Random-source conditional Flow candidates only. Post-relax hard gates, finite-reference '
                   'CHGNet hull threshold, minimax ranking, then novelty/diversity. Reference hull is not a complete phase diagram.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='selection'}))

if __name__=='__main__': main()

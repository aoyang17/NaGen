"""Train a Na-site rectified flow and emit generated, fixed-composition crystals."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from pymatgen.core import Structure

from nagen.selection.audit import load_crystals
from nagen.selection.experiment import json_default, sha256
from nagen.selection.occupancy_flow import (OccupancyFlow, OccupancyFlowConfig,
    flow_matching_loss, sample_logits, topk_occupancy)


def occupancy_patterns(train, parent, target_na, mapping_threshold):
    raw = {r['material_id']: r for r in map(json.loads, Path(train).read_text().splitlines())}
    family = [c for c in load_crystals(train) if raw[c.id].get('parent_group') == parent]
    if not family:
        raise ValueError('parent absent from declared training split')
    prototype = max(family, key=lambda c: c.elements.count('Na'))
    na_indices = np.flatnonzero(np.asarray(prototype.elements) == 'Na')
    sites = prototype.frac[na_indices]
    patterns, mapping = [], []
    for crystal in family:
        if crystal.elements.count('Na') != target_na:
            continue
        coords = crystal.frac[np.asarray(crystal.elements) == 'Na']
        delta = coords[:, None] - sites[None]
        delta -= np.floor(delta+.5)
        distance = np.linalg.norm(delta @ prototype.lattice, axis=-1)
        rows, columns = linear_sum_assignment(distance)
        error = float(distance[rows, columns].max())
        if error <= mapping_threshold:
            pattern = tuple(sorted(map(int, columns)))
            patterns.append(pattern); mapping.append({'id': crystal.id, 'max_A': error})
    patterns = sorted(set(patterns))
    if len(patterns) < 20:
        raise ValueError('insufficient unique high-quality occupancy labels')
    return prototype, na_indices, patterns, mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True); p.add_argument('--parent', required=True)
    p.add_argument('--target-na', type=int, required=True); p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=314159); p.add_argument('--epochs', type=int, default=4000)
    p.add_argument('--samples', type=int, default=256); p.add_argument('--steps', type=int, default=64)
    p.add_argument('--mapping-threshold', type=float, default=1.0); p.add_argument('--device', default='cpu')
    p.add_argument('--emit-unique-only', action='store_true',
                   help='write one representative per identical decoded pattern (compute cache only)')
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    prototype, na_indices, patterns, mapping = occupancy_patterns(
        args.train, args.parent, args.target_na, args.mapping_threshold)
    labels = torch.zeros(len(patterns), len(na_indices), device=args.device)
    for i, pattern in enumerate(patterns): labels[i, list(pattern)] = 1
    model = OccupancyFlow(OccupancyFlowConfig(len(na_indices))).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    history=[]
    for epoch in range(args.epochs):
        optimizer.zero_grad(); loss=flow_matching_loss(model, labels, args.target_na)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if epoch % 100 == 0 or epoch+1 == args.epochs: history.append([epoch, float(loss)])
    checkpoint=out/'flow.pt'
    torch.save({'state_dict':model.state_dict(), 'config':model.metadata(), 'args':vars(args),
                'patterns':[list(x) for x in patterns]}, checkpoint)
    model.eval(); generator=torch.Generator(device=args.device).manual_seed(args.seed+1)
    logits=sample_logits(model,args.samples,args.target_na,args.steps,generator,args.device)
    generated=topk_occupancy(logits,args.target_na).cpu().numpy()
    known=set(patterns); seen=set(); rows=[]
    base_keep=[i for i,e in enumerate(prototype.elements) if e!='Na']
    for source_index, occupancy in enumerate(generated):
        pattern=tuple(np.flatnonzero(occupancy).tolist())
        duplicate_sample=pattern in seen; seen.add(pattern)
        keep=base_keep+[int(na_indices[i]) for i in pattern]
        crystal=Structure(prototype.lattice,[prototype.elements[i] for i in keep],prototype.frac[keep])
        row={'material_id':f'occflow_seed{args.seed+1}_{source_index:04d}',
             'A':[s.specie.symbol for s in crystal], 'X_frac':crystal.frac_coords,
             'L_matrix':crystal.lattice.matrix, 'family':'phosphate',
             'generation':{'method':'conditional_rectified_flow_Na_occupancy',
              'source_seed':args.seed+1,'source_index':source_index,'ode_steps':args.steps,
              'target_Na':args.target_na,'parent_family':args.parent,'prototype_id':prototype.id,
              'occupied_Na_site_indices':list(pattern),'exact_training_pattern':pattern in known,
              'duplicate_generated_pattern':duplicate_sample}}
        rows.append(row)
    emitted = ([row for row in rows if not row['generation']['duplicate_generated_pattern']]
               if args.emit_unique_only else rows)
    generated_path=out/'generated.jsonl'
    with generated_path.open('x') as handle:
        for row in emitted: handle.write(json.dumps(row,default=json_default)+'\n')
    unique=set(tuple(np.flatnonzero(x).tolist()) for x in generated)
    manifest={'status':'completed','method':'conditional_rectified_flow_Na_occupancy',
      'conditioning':{'N':len(prototype.elements)-len(na_indices)+args.target_na,
       'composition':dict(zip(*np.unique(rows[0]['A'],return_counts=True))),
       'target_Na':args.target_na,'family':'phosphate','parent_family':args.parent},
      'training':{'unique_patterns':len(patterns),'accepted_mappings':len(mapping),
       'mapping_threshold_A':args.mapping_threshold,'history':history,
       'loss':'standard rectified-flow velocity MSE only; no physical or novelty terms'},
      'sampling':{'requested':args.samples,'unique_patterns':len(unique),
       'emitted_for_evaluation':len(emitted),'identical_decode_cache':args.emit_unique_only,
       'exact_training_patterns':sum(x in known for x in unique),
       'novel_patterns':sum(x not in known for x in unique)},
      'provenance':{'train_sha256':sha256(args.train),'checkpoint_sha256':sha256(checkpoint),
       'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
      'note':'All samples are retained here. Training-set and generated duplicates are handled only in downstream diversity selection.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=json_default)+'\n')
    print(json.dumps(manifest['sampling']))

if __name__=='__main__': main()

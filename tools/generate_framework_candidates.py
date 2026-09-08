"""Generate complete fixed-composition crystal geometries from random Flow sources."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from nagen.inverse.generate import generate_conditioned_geometry_batch, load_model
from nagen.selection.experiment import json_default, sha256
from nagen.selection.pipeline import Crystal, Conditioning, hard_gates, neighbors


def catastrophic_reason(crystal):
    volume_per_atom=abs(np.linalg.det(crystal.lattice))/len(crystal.elements)
    if not np.isfinite(crystal.frac).all() or not np.isfinite(crystal.lattice).all():
        return 'nonfinite'
    if not 2 <= volume_per_atom <= 60 or np.linalg.cond(crystal.lattice)>40:
        return 'unsafe_cell'
    if any(d<.4 for i in range(len(crystal.elements)) for _,_,d in neighbors(crystal,i,.4)):
        return 'overlap_below_0.4A'
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flow',required=True); p.add_argument('--condition',required=True)
    p.add_argument('--profile',required=True); p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=128); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--seed',type=int,default=271828); p.add_argument('--ode-steps',type=int,default=48)
    p.add_argument('--source-mode',choices=('uniform','polyhedral'),default='uniform')
    p.add_argument('--flow-end-time',type=float,default=1.0)
    p.add_argument('--device',default='cuda'); args=p.parse_args()
    if args.count<1 or args.batch_size<1: p.error('positive count/batch-size required')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    model,checkpoint=load_model(args.flow,torch.device(args.device),True)
    if checkpoint.get('training_mode') not in ('geometry','fixed_NA_geometry'):
        raise ValueError('checkpoint is not an explicitly fixed-composition geometry Flow')
    condition=Conditioning(**json.loads(Path(args.condition).read_text()))
    elements=[e for e in ('Na','Fe','P','O') for _ in range(condition.counts.get(e,0))]
    indices=torch.tensor([checkpoint['element_to_index'][e] for e in elements],dtype=torch.long)
    profile=json.loads(Path(args.profile).read_text()); records=[]; counts=Counter()
    torch.manual_seed(args.seed)
    with (out/'generated_all.jsonl').open('x') as all_handle, (out/'generated_safe.jsonl').open('x') as safe_handle:
        for start in range(0,args.count,args.batch_size):
            size=min(args.batch_size,args.count-start)
            types,frac,lattice,mask=generate_conditioned_geometry_batch(
                model,[indices.clone() for _ in range(size)],args.seed+start,args.ode_steps,
                'midpoint',0,source_mode=args.source_mode,end_time=args.flow_end_time)
            for local in range(size):
                index=start+local; crystal=Crystal(f'framework_flow_seed{args.seed}_{index:04d}',elements,
                    frac[local,mask[local]].cpu().numpy(),lattice[local].cpu().numpy(),condition.family)
                reason=catastrophic_reason(crystal)
                # Force is intentionally unknown here; record motif/geometry checks
                # without pretending the final force gate has been evaluated.
                gate=hard_gates(crystal,condition,profile,forces=None,motif_backend='native')
                row={'material_id':crystal.id,'A':elements,'X_frac':crystal.frac,'L_matrix':crystal.lattice,
                     'family':condition.family,'generation':{'method':'fixed_NA_complete_XL_rectified_flow',
                     'source_seed':args.seed+start,'source_batch_index':local,'global_index':index,
                     'ode_steps':args.ode_steps,'generated_variables':'all Na/Fe/P/O coordinates and lattice',
                     'source_mode':args.source_mode,
                     'flow_end_time':args.flow_end_time,
                     'copied_parent_coordinates':False},'pre_relax':{'catastrophic_reason':reason,'gates':gate}}
                all_handle.write(json.dumps(row,default=json_default,allow_nan=False)+'\n'); records.append(row)
                counts['attempted']+=1
                if reason is None:
                    safe_handle.write(json.dumps(row,default=json_default,allow_nan=False)+'\n'); counts['safe_for_potential']+=1
                else: counts[reason]+=1
            all_handle.flush(); safe_handle.flush(); print(json.dumps({'generated':min(start+size,args.count),'counts':counts}),flush=True)
    manifest={'status':'completed','configuration':vars(args),'counts':counts,
      'conditioning':{'N':len(elements),'counts':condition.counts,'family':condition.family},
      'flow_sha256':sha256(args.flow),'train_dataset_hash':checkpoint.get('dataset_source_hash'),
      'training_objective':checkpoint.get('training_objective'),
      'note':'Every candidate starts from a fresh random X,L source. No parent coordinates, energy, motif, force, or novelty objective is used in generation. Unsafe endpoints remain in generated_all.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=json_default)+'\n')

if __name__=='__main__': main()

"""Replay every random-source Flow batch and verify exported raw coordinates."""
import argparse,json
from pathlib import Path
import numpy as np
import torch

from nagen.inverse.generate import generate_conditioned_geometry_batch,load_model
from nagen.selection.audit import load_crystals


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',required=True); p.add_argument('--flow',required=True); p.add_argument('--condition',required=True); p.add_argument('--out',required=True); p.add_argument('--device',default='cuda'); args=p.parse_args()
    run=Path(args.run); manifest=json.loads((run/'generation/manifest.json').read_text()); config=manifest['configuration']; condition=json.loads(Path(args.condition).read_text()); elements=[e for e in ('Na','Fe','P','O') for _ in range(condition['counts'].get(e,0))]
    model,checkpoint=load_model(args.flow,torch.device(args.device),True); sequence=torch.tensor([checkpoint['element_to_index'][e] for e in elements]); exported={c.id:c for c in load_crystals(run/'generation/generated_all.jsonl')}; rows=[]
    for start in range(0,config['count'],config['batch_size']):
        size=min(config['batch_size'],config['count']-start); types,frac,lattice,mask=generate_conditioned_geometry_batch(model,[sequence.clone() for _ in range(size)],config['seed']+start,config['ode_steps'],'midpoint',0,source_mode=config['source_mode'],end_time=config.get('flow_end_time',1.0))
        for local in range(size):
            id=f'framework_flow_seed{config["seed"]}_{start+local:04d}'; target=exported[id]; delta=frac[local,mask[local]].cpu().numpy()-target.frac; delta-=np.floor(delta+.5); rows.append({'id':id,'max_fractional_error':float(np.abs(delta).max()),'max_lattice_error_A':float(np.abs(lattice[local].cpu().numpy()-target.lattice).max())})
    result={'status':'verified' if max(max(x['max_fractional_error'],x['max_lattice_error_A']) for x in rows)<=1e-6 else 'failed','count':len(rows),'max_fractional_error':max(x['max_fractional_error'] for x in rows),'max_lattice_error_A':max(x['max_lattice_error_A'] for x in rows),'flow':args.flow,'rows':rows}; Path(args.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if result['status']!='verified': raise SystemExit(2)

if __name__=='__main__':main()

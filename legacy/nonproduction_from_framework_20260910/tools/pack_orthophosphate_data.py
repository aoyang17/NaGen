"""Pack all Na/Fe/P/O structures with O=4P into stratified geometry splits."""
import argparse
from collections import Counter,defaultdict
import json,random
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure

from nagen.inverse.spec import DEFAULT_SPEC
from nagen.selection.audit import load_crystals
from nagen.selection.experiment import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=20260909); args=p.parse_args()
    out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    raw_meta={}
    for split in ('train','val','test'):
        for line in open(Path(args.data)/f'{split}.jsonl'):
            r=json.loads(line); raw_meta[r['material_id']]=r.get('parent_group','unknown')
    selected=[]
    for source_split in ('train','val','test'):
        for c in load_crystals(Path(args.data)/f'{source_split}.jsonl'):
            counts=Counter(c.elements)
            if counts['P'] and counts['O']==4*counts['P']:
                selected.append((c,raw_meta[c.id],source_split))
    by_parent=defaultdict(list)
    for row in selected: by_parent[row[1]].append(row)
    rng=random.Random(args.seed); split_for={}
    for parent,rows in sorted(by_parent.items()):
        rng.shuffle(rows); n=len(rows); nv=max(1,round(.1*n)) if n>=10 else 1; nt=max(1,round(.1*n)) if n>=20 else 0
        for c,_,_ in rows[:nv]: split_for[c.id]=1
        for c,_,_ in rows[nv:nv+nt]: split_for[c.id]=2
        for c,_,_ in rows[nv+nt:]: split_for[c.id]=0
    types=[]; frac=[]; lattices=[]; ids=[]; splits=[]; offsets=[0]; parents=[]
    for c,parent,_ in selected:
        reduced=Structure(c.lattice,c.elements,c.frac).get_reduced_structure(reduction_algo='niggli')
        if [s.specie.symbol for s in reduced]!=c.elements: raise ValueError('Niggli ordering changed')
        types.extend(DEFAULT_SPEC.element_to_index[e] for e in c.elements); frac.extend(reduced.frac_coords%1); lattices.append(reduced.lattice.matrix); ids.append(c.id); parents.append(parent); splits.append(split_for[c.id]); offsets.append(len(types))
    payload={'version':'orthophosphate-polyhedral-flow-v1','element_to_index':DEFAULT_SPEC.element_to_index,
      'types':torch.tensor(types,dtype=torch.int16),'frac_coords':torch.tensor(np.asarray(frac),dtype=torch.float32),
      'lattices':torch.tensor(np.asarray(lattices),dtype=torch.float32),'ids':ids,'parents':parents,
      'split':torch.tensor(splits,dtype=torch.uint8),'offsets':torch.tensor(offsets),
      'filter_definition':'Na/Fe/P/O original optimized geometry with O=4P; Niggli basis only; no energy labels; parent-stratified random split.',
      'source_hashes':{s:sha256(Path(args.data)/f'{s}.jsonl') for s in ('train','val','test')},'split_seed':args.seed}
    out.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,out)
    comps=Counter(tuple(sorted(Counter(c.elements).items())) for c,_,_ in selected if split_for[c.id]==0)
    manifest={'count':len(selected),'split_counts':dict(Counter(splits)),'parent_counts':dict(Counter(parents)),
      'training_compositions':[(dict(k),v) for k,v in comps.most_common(12)],'sha256':sha256(out),'contains_energy_labels':False}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

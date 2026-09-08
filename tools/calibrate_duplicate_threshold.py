"""Freeze radial duplicate threshold from training data only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

from nagen.selection.audit import load_crystals
from nagen.selection.diversity import descriptor, distances, DESCRIPTOR_VERSION
from nagen.selection.experiment import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True); p.add_argument('--condition',required=True); p.add_argument('--out',required=True)
    args=p.parse_args(); condition=json.loads(Path(args.condition).read_text())['counts']
    rows=[c for c in load_crystals(args.train) if Counter(c.elements)==Counter(condition)]
    if len(rows)<20: raise ValueError('insufficient training structures for distance calibration')
    x=np.stack([descriptor(c) for c in rows]); d=distances(x,x); d[d<=1e-15]=np.inf
    nearest=d.min(1); threshold=1e-4
    result={'status':'completed','training_count':len(rows),'train_sha256':sha256(args.train),
            'descriptor':DESCRIPTOR_VERSION,'nearest_distance_quantiles':dict(zip(
              ['min','p05','p25','median','p75','p95','max'],map(float,np.quantile(nearest,[0,.05,.25,.5,.75,.95,1])))),
            'frozen_duplicate_threshold':threshold,
            'fraction_training_nearest_below_threshold':float((nearest<threshold).mean()),
            'note':'Threshold chosen before candidate ranking, between the 5th and 25th percentiles of TRAIN nearest distances. '
                   'Finalists additionally require StructureMatcher non-equivalence.'}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

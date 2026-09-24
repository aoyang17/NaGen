"""Four-layer selection with novelty defined on the Fe-P-O framework only."""
import argparse,hashlib,itertools,json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import categorical_node_match
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from shootingcsp.selection.audit import load_crystals
from shootingcsp.selection.diversity import descriptor,distances,DESCRIPTOR_VERSION
from shootingcsp.inverse._io import sha256_file as sha256
from shootingcsp.naming import build_output_stem_from_records
from shootingcsp.selection.hull import ReferenceHull
from shootingcsp.selection.pipeline import Crystal,neighbors,robust_rank,diversity_select


def strip_na(c):
    keep=[i for i,e in enumerate(c.elements) if e!='Na']
    return Crystal(c.id,[c.elements[i] for i in keep],c.frac[keep],c.lattice,c.family)


def structure(c): return Structure(c.lattice,c.elements,c.frac)


def coordination_graph(c):
    graph=nx.Graph()
    for i,e in enumerate(c.elements):
        if e!='Na': graph.add_node(i,element=e)
    for i,e in enumerate(c.elements):
        if e not in ('P','Fe'): continue
        for j,_,_ in neighbors(c,i,2.0 if e=='P' else 2.5):
            if c.elements[j]=='O': graph.add_edge(i,j)
    return graph


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',required=True); p.add_argument('--known',required=True); p.add_argument('--hull-labels',required=True); p.add_argument('--out',required=True); p.add_argument('--k',type=int,default=6); p.add_argument('--max-reference-hull',type=float,default=.15); args=p.parse_args()
    run,out=Path(args.run),Path(args.out); out.mkdir(parents=True,exist_ok=False)
    generated={r['material_id']:r for r in map(json.loads,(run/'generation/generated_all.jsonl').open())}
    pairs={r['id']:r for r in map(json.loads,(run/'relax/pairs.jsonl').open())}; raw={c.id:c for c in load_crystals(run/'generation/generated_all.jsonl')}; final={c.id:c for c in load_crystals(run/'relax/relaxed.jsonl')}
    known=[]
    for split in ('train','val','test'): known.extend(load_crystals(Path(args.known)/f'{split}.jsonl'))
    labels=[json.loads(x) for x in Path(args.hull_labels).read_text().splitlines()]; model_hash=json.loads((run/'relax/summary.json').read_text())['model']['state_sha256']; hull=ReferenceHull(labels,model_hash)
    feasible=[]
    for id,row in pairs.items():
        if not row['gates']['after']['passed']: continue
        hr=hull.evaluate(final[id].elements,row['after']['energy_eV_atom']); eh=hr['e_above_reference_hull_eV_atom']
        if eh is None or eh>args.max_reference_hull: continue
        metrics={**row['gates']['after']['metrics'],'reference_hull_eV_atom':eh}
        feasible.append({**row['gates']['after'],'passed':True,'group':repr(final[id].composition),'metrics':metrics,'reference_hull':hr})
    metrics=['reference_hull_eV_atom']+[f'{e}_{m}' for e in ('P','Fe') for m in ('off_center','bond_distortion','angle_distortion','coordination_rarity')]
    ranked=next(iter(robust_rank(feasible,metrics).values()),[])
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True); node_match=categorical_node_match('element','')
    target_count=Counter(e for e in final[ranked[0]['id']].elements if e!='Na') if ranked else Counter()
    refs=[strip_na(c) for c in known if Counter(e for e in c.elements if e!='Na')==target_count]; ref_struct=[structure(c) for c in refs]; ref_graph=[coordination_graph(c) for c in refs]; ref_features=np.stack([descriptor(c) for c in refs])
    eligible=[]; features=[]; novelty=[]
    for row in ranked:
        id=row['id']; before,after=strip_na(raw[id]),strip_na(final[id]); bs,fs=structure(before),structure(after); fg=coordination_graph(after)
        pre_match=any(matcher.fit(bs,x) for x in ref_struct); post_match=any(matcher.fit(fs,x) for x in ref_struct); topology_match=any(nx.is_isomorphic(fg,x,node_match=node_match) for x in ref_graph)
        d=float(distances(np.stack([descriptor(after)]),ref_features).min())
        record={'id':id,'pre_relax_training_match':pre_match,'post_relax_training_match':post_match,'post_relax_topology_match':topology_match,'nearest_known_framework_distance':d,'pre_edges':coordination_graph(before).number_of_edges(),'post_edges':fg.number_of_edges()}
        novelty.append(record)
        provenance=generated[id]['generation']
        if (not pre_match and not post_match and not topology_match and d>=1e-4
            and provenance['generated_variables']=='all Na/Fe/P/O coordinates and lattice'
            and not provenance['copied_parent_coordinates']):
            eligible.append(row); features.append(descriptor(after))
    if eligible:
        features=np.stack(features); nearest=distances(features,ref_features).min(1)
        chosen=diversity_select(eligible,distances(features,features),nearest,args.k,len(eligible),1e-4)
    else: chosen=[]
    # Final exact pairwise identity audit is independent of radial selection.
    chosen_struct=[structure(strip_na(final[x['id']])) for x in chosen]
    pair_matches=[(chosen[i]['id'],chosen[j]['id']) for i,j in itertools.combinations(range(len(chosen)),2) if matcher.fit(chosen_struct[i],chosen_struct[j])]
    if pair_matches: raise RuntimeError(f'selected framework duplicates: {pair_matches}')
    cif=out/'CIF'; cif.mkdir(); manifest=[]
    for output_index,row in enumerate(chosen,1):
        id=row['id']; novelty_audit=next(x for x in novelty if x['id']==id)
        output_name=build_output_stem_from_records(output_index,row,row['reference_hull'],novelty_audit)
        path=cif/f'{output_name}.cif'; CifWriter(structure(final[id]),significant_figures=10).write_file(path)
        if not StructureMatcher(ltol=1e-6,stol=1e-5,angle_tol=1e-5,primitive_cell=False,scale=False).fit(structure(final[id]),Structure.from_file(path)):
            raise RuntimeError(f'CIF round-trip failed: {id}')
        manifest.append({'id':id,'output_index':output_index,'output_name':output_name,'file':path.name,'vesta_png':f'{output_name}.png','sha256':sha256(path),'metrics':row['metrics'],'reference_hull':row['reference_hull'],'generation':generated[id]['generation'],'framework_novelty':novelty_audit})
    (cif/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary={'status':'completed','attempted':len(pairs),'hard_feasible':sum(r['gates']['after']['passed'] for r in pairs.values()),'hull_threshold_passing':len(feasible),'framework_novelty_eligible':len(eligible),'selected':len(chosen),'selected_ids':[x['id'] for x in chosen],'pair_matches':pair_matches,'known_framework_references':len(refs),'framework_descriptor':DESCRIPTOR_VERSION,'ranking_metrics':metrics,'minimax_order':[x['id'] for x in ranked],'novelty_audit':novelty,'finite_reference_sha256':hull.hash,'known_hashes':{s:sha256(Path(args.known)/f'{s}.jsonl') for s in ('train','val','test')},'note':'Novelty uses Fe-P-O only; Na is removed. Requires both StructureMatcher and coordination-graph non-equivalence before farthest-first diversity.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps({k:v for k,v in summary.items() if k not in ('novelty_audit','minimax_order')},indent=2))

if __name__=='__main__':main()

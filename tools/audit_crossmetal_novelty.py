"""Expand frozen-candidate novelty checks to metal-substituted phosphate frameworks."""
import argparse
import ast
import gzip
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from functools import reduce
from math import gcd
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/crossmetal_novelty_19'
PUBLIC_PO=OUT/'mp_public_po.json'


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2))
    tmp.replace(path)


def api_key():
    key=os.environ.get('MP_API_KEY')
    if key:return key
    # Reuse the project's existing MP credential without importing its runnable script.
    path=Path('/home/aobo/NaGen/tools/relaxation_n_ehull.py')
    tree=ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MP_API_KEY' for t in n.targets):
            if isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):return n.value.value
    raise RuntimeError('No configured MP credential')


def download():
    OUT.mkdir(parents=True,exist_ok=True)
    headers={'X-API-KEY':api_key(),'Accept':'application/json'}
    params={'elements':'P,O','deprecated':'false','_fields':'material_id,formula_pretty,structure,elements','_all_fields':'false','_limit':500}
    received=0;total=None;pages=[]
    while total is None or received<total:
        query={**params,'_skip':received}
        url='https://api.materialsproject.org/materials/summary/?'+urllib.parse.urlencode(query)
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as response:
                payload=json.load(response)
        except urllib.error.HTTPError as e:
            write(OUT/'download_status.json',{'status':'failed','http_status':e.code,'received':received,'query':params})
            raise RuntimeError(f'MP API HTTP {e.code}; credential not logged') from None
        except urllib.error.URLError:
            write(OUT/'download_status.json',{'status':'network_failed','received':received,'query':params})
            raise RuntimeError('MP API network request failed; credential not logged') from None
        data=payload.get('data',[])
        total=int(payload['meta']['total_doc'])
        if not data:raise RuntimeError('Empty MP page before declared total')
        page=OUT/f'mp_page_{received:06d}.json'
        write(page,payload)
        pages.append({'file':page.name,'sha256':hashlib.sha256(page.read_bytes()).hexdigest(),'count':len(data)})
        received+=len(data)
        print(json.dumps({'received':received,'total':total}),flush=True)
    write(OUT/'download_status.json',{'status':'complete','utc':datetime.now(timezone.utc).isoformat(),'query':params,'received':received,'total':total,'pages':pages})


def public_download():
    url='https://ndownloader.figshare.com/files/13309250'
    expected='1f3de2dc7c68959647240921b841293fce918ea1708e6f3760ba35a9cdfe0500'
    path=OUT/'mp_20181018.json.gz'
    if not path.exists():
        temp=path.with_suffix('.gz.partial')
        with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as handle:
            size=0
            while chunk:=response.read(1024*1024):
                handle.write(chunk);size+=len(chunk)
                if size>512*1024*1024:raise RuntimeError('Public snapshot exceeded 512 MiB safety cap')
                if size%(20*1024*1024)==0:print('downloaded MiB',size//(1024*1024),flush=True)
        temp.replace(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==expected,'Public dataset hash differs from published matminer metadata'
    with gzip.open(path,'rt') as handle:data=json.load(handle)
    column=data['columns'].index('structure')
    id_column=data['columns'].index('mpid')
    selected=[]
    for index,row in enumerate(data['data']):
        raw=row[column]
        elements={sp['element'] for site in raw['sites'] for sp in site['species']}
        if {'P','O'}<=elements:
            selected.append({'id':row[id_column],'structure':raw})
    write(PUBLIC_PO,{'source_url':url,'sha256':digest,'retrieved_snapshot':'2018-10-18','total_structures':len(data['data']),
        'exclusions':'Historical complete MP snapshot per published metadata; not full current MP.',
        'id_note':'Original MP IDs retained.',
        'structures':selected})
    print(json.dumps({'public_total':len(data['data']),'P_O_structures':len(selected)}),flush=True)


def framework(structure):
    """Strip alkali guests; merge metal identities only, preserving P and O."""
    from pymatgen.core import Element, Structure
    alkali={'Li','Na','K','Rb','Cs','Fr'}
    species=[];coords=[];metals=set()
    for site in structure:
        elements={str(e.symbol) for e in site.species}
        occupancy=sum(site.species.values())
        if abs(occupancy-1)>1e-8:
            return None, 'partial_occupancy', sorted(metals)
        if elements<=alkali:
            continue
        if elements=={'P'}:label='P'
        elif elements=={'O'}:label='O'
        elif elements.isdisjoint(alkali) and all(Element(e).is_metal for e in elements):
            label='Fe';metals.update(elements)
        else:
            return None,'other_anion_or_mixed_role',sorted(metals)
        species.append(label);coords.append(site.frac_coords)
    if set(species)!={'Fe','P','O'}:
        return None,'not_metal_phosphate',sorted(metals)
    return Structure(structure.lattice,species,coords),None,sorted(metals)


def ratio(structure):
    counts=Counter(str(s.specie) for s in structure)
    div=reduce(gcd,counts.values())
    return tuple(sorted((e,n//div) for e,n in counts.items()))


def prepare():
    from pymatgen.core import Structure
    sources=[Path('/home/aobo/NaGen/parent_RDF_opt.jsonl'),
        Path('/mnt/data2/aobo/NaGen/experiments/exp_2026-08-27_hull_distance_calibration/cache/mp_reference_structures.json'),
        Path('/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26_02/mp_cache/mp_competing_structures.json')]
    if OUT.name=='with_public_mp':sources.append(PUBLIC_PO)
    target=(('Fe',3),('O',16),('P',4))
    refs=[];seen={};source_stats=[]
    for source in sources:
        if source.suffix=='.jsonl':
            def records():
                with source.open() as handle:
                    for line in handle:
                        r=json.loads(line)
                        yield r['material_id'],r['structure'],'local_derived'
        elif source==PUBLIC_PO:
            def records():
                for r in json.loads(source.read_text())['structures']:
                    yield r['id'],r['structure'],'public_MP_2018_snapshot'
        else:
            def records():
                for ident,r in json.loads(source.read_text())['structures'].items():
                    yield r.get('returned_material_id',ident),r.get('structure',r),'cached_mp'
        stat={'file':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scanned':0,'eligible_before_exact_hash_dedup':0,
              'excluded':Counter(),'metal_coverage':Counter(),'eligible_metal_coverage':Counter()}
        for ident,raw,origin in records():
            stat['scanned']+=1
            s=Structure.from_dict(raw)
            f,reason,metals=framework(s)
            stat['metal_coverage'].update(metals)
            if reason:
                stat['excluded'][reason]+=1;continue
            if ratio(f)!=target:
                stat['excluded']['different_reduced_M_P_O_ratio']+=1;continue
            stat['eligible_before_exact_hash_dedup']+=1
            stat['eligible_metal_coverage'].update(metals)
            # Deduplicate only exactly identical normalized arrays, not approximately similar references.
            key=hashlib.sha256(json.dumps({'L':f.lattice.matrix.tolist(),'A':[str(x.specie) for x in f],'X':f.frac_coords.tolist()},sort_keys=True).encode()).hexdigest()
            alias={'id':str(ident),'origin':origin,'metals':metals,'formula':s.composition.reduced_formula}
            if key in seen:
                refs[seen[key]]['aliases'].append(alias)
            else:
                seen[key]=len(refs)
                refs.append({'id':str(ident),'structure':f.as_dict(),'aliases':[alias]})
            if stat['scanned']%2000==0:print('scanned',stat['scanned'],'eligible',len(refs),flush=True)
        for key in ('excluded','metal_coverage','eligible_metal_coverage'):stat[key]=dict(stat[key])
        source_stats.append(stat)
        print(json.dumps({k:stat[k] for k in ('file','scanned','eligible_before_exact_hash_dedup','eligible_metal_coverage')}),flush=True)
    write(OUT/'local_references.json',{'scope':'Local derived Na-cathode dataset plus frozen MP caches, NOT online full MP query',
        'target_ratio':target,'sources':source_stats,'unique_exact_arrays':len(refs),'references':refs})


def evaluate():
    import csv
    import io
    import numpy as np
    import networkx as nx
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from networkx.algorithms.isomorphism import categorical_node_match
    sys.path.insert(0,str(ROOT/'src'))
    from shootingcsp.selection.pipeline import Crystal
    from shootingcsp.selection.diversity import descriptor, distances
    from select_framework_candidates import coordination_graph
    pool=json.loads((OUT/'local_references.json').read_text())
    refs=pool['references']
    rs=[Structure.from_dict(r['structure']) for r in refs]
    def crystal(s,ident):return Crystal(ident,[str(x.specie) for x in s],np.asarray(s.frac_coords),np.asarray(s.lattice.matrix))
    standard=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    loose=StructureMatcher(ltol=.3,stol=.5,angle_tol=10,primitive_cell=True,scale=True,attempt_supercell=True)
    ref_feats=[];ref_graphs=[];ref_hashes=[]
    for i,s in enumerate(rs):
        ref_feats.append(descriptor(crystal(s,refs[i]['id'])))
        g=coordination_graph(crystal(s.get_primitive_structure(),refs[i]['id']))
        ref_graphs.append(g)
        ref_hashes.append(nx.weisfeiler_lehman_graph_hash(g,node_attr='element'))
    ref_feats=np.stack(ref_feats)
    snap=ROOT/'runs/campaign_24/report_19'
    selected=json.loads((snap/'snapshot.json').read_text())['rows']
    reused=0;cache_hash=None
    if OUT.name=='with_public_mp':
        cached_path=OUT.parent/'results.json'
        cached=json.loads(cached_path.read_text())
        cached_refs=json.loads((OUT.parent/'local_references.json').read_text())['references']
        # Reuse only the completed all-negative screening against an unchanged prefix.
        assert [r['id'] for r in cached['rows']]==[r['id'] for r in selected]
        assert all(r['passes_local_expanded_screen'] for r in cached['rows'])
        assert [r['structure'] for r in refs[:len(cached_refs)]]==[r['structure'] for r in cached_refs]
        reused=len(cached_refs);cache_hash=hashlib.sha256(cached_path.read_bytes()).hexdigest()
    rows=[]
    for entry in selected:
        ident=entry['id']
        audit=json.loads((snap/'audit'/(ident+'.json')).read_text())
        assert hashlib.sha256((snap/'cif'/entry['file']).read_bytes()).hexdigest()==entry['sha256']
        original=Structure.from_file(snap/'cif'/entry['file'])
        final,_,_=framework(original)
        raw=audit['raw']
        before,_,_=framework(Structure(raw['L_matrix'],raw['A'],raw['X_frac']))
        matches=[];loose_matches=[];pre_matches=[]
        for i,reference in enumerate(rs):
            if i<reused:continue
            if standard.fit(final,reference):matches.append(i)
            elif loose.fit(final,reference):loose_matches.append(i)
            if standard.fit(before,reference):pre_matches.append(i)
        graph=coordination_graph(crystal(final.get_primitive_structure(),ident))
        gh=nx.weisfeiler_lehman_graph_hash(graph,node_attr='element')
        topology=[i for i,g in enumerate(ref_graphs) if ref_hashes[i]==gh and nx.is_isomorphic(graph,g,node_match=categorical_node_match('element',''))]
        dist=distances(descriptor(crystal(final,ident))[None],ref_feats)[0]
        nearest=np.argsort(dist)[:5]
        def details(indices):return [{'reference_id':refs[i]['id'],'aliases':refs[i]['aliases']} for i in indices]
        row={'id':ident,'number':entry['number'],'cif_sha256':entry['sha256'],'standard_matches':details(matches),'loose_only_matches':details(loose_matches),
             'pre_relax_standard_matches':details(pre_matches),'equal_primitive_quotient_graph_matches':details(topology),
             'graph_comparable_count':sum(g.number_of_nodes()==graph.number_of_nodes() for g in ref_graphs),
             'nearest_normalized_metal_references':[{'distance':float(dist[i]),**details([i])[0]} for i in nearest],
             'passes_local_expanded_screen':not bool(matches or loose_matches or pre_matches or topology)}
        rows.append(row)
        write(OUT/'results.partial.json',{'rows':rows})
        print(json.dumps({'candidate':entry['number'],'standard':len(matches),'loose':len(loose_matches),'pre':len(pre_matches),'graph':len(topology),'nearest_distance':float(dist.min())}),flush=True)
    coverage=Counter()
    for ref in refs:
        for alias in ref['aliases']:coverage.update(alias['metals'])
    result={'status':'historical_mp_and_local_complete_current_mp_blocked' if OUT.name=='with_public_mp' else 'local_audit_complete_mp_expansion_blocked','count':19,'reference_count':len(refs),'coverage':dict(coverage),
        'reused_negative_reference_count':reused,'reused_result_sha256':cache_hash,
        'sources':pool['sources'],'rows':rows,'matching_tolerances':{'standard':{'ltol':.2,'stol':.3,'angle_tol':5},'loose':{'ltol':.3,'stol':.5,'angle_tol':10}},
        'normalization':'Remove Li/Na/K/Rb/Cs/Fr; all other metal sites -> Fe label; P/O retain identity. Normalize composition ratio, allow primitive cells and supercells.',
        'limitations':['MP online API returned HTTP 403; no full-MP novelty conclusion.',
        'Local data are derived structures, not automatically experimental or MP entries.',
        'Other anions and partial vacancies excluded; alkali-only guest removal.',
        'Graph check is an equal-size primitive-cell quotient graph, not proof of infinite periodic net equivalence.',
        'No-match at geometric tolerances does not establish universal topological novelty.']}
    write(OUT/'results.json',result)
    buffer=io.StringIO();writer=csv.writer(buffer)
    writer.writerow(['number','id','standard_match_count','loose_only_count','pre_match_count','quotient_graph_match_count','nearest_distance','local_screen_passed','MP_complete'])
    for r in rows:writer.writerow([r['number'],r['id'],len(r['standard_matches']),len(r['loose_only_matches']),len(r['pre_relax_standard_matches']),len(r['equal_primitive_quotient_graph_matches']),r['nearest_normalized_metal_references'][0]['distance'],r['passes_local_expanded_screen'],False])
    (OUT/'summary.csv').write_text(buffer.getvalue())


def summarize():
    import csv
    import io
    import numpy as np
    from pymatgen.core import Structure
    sys.path.insert(0,str(ROOT/'src'))
    from shootingcsp.selection.pipeline import Crystal
    from shootingcsp.selection.diversity import descriptor,distances
    result=json.loads((OUT/'results.json').read_text())
    pool=json.loads((OUT/'local_references.json').read_text())
    features=[]
    for ref in pool['references']:
        s=Structure.from_dict(ref['structure'])
        features.append(descriptor(Crystal(ref['id'],[str(x.specie) for x in s],np.array(s.frac_coords),np.array(s.lattice.matrix))))
    pair=distances(np.stack(features),np.stack(features))
    positive=pair[np.triu_indices(len(pair),1)];positive=positive[positive>1e-4]
    scale=float(np.median(positive))
    rows=result['rows']
    for row in rows:
        nearest=row['nearest_normalized_metal_references'][0]['distance']
        known_match=bool(row['standard_matches'] or row['pre_relax_standard_matches'] or row['equal_primitive_quotient_graph_matches'])
        row['crossmetal_score_0_10']=0. if known_match else 10*nearest/(nearest+scale)
    result['score_definition']={'formula':'0 if a standard pre/post or quotient-graph match; otherwise 10*d/(d+d0)',
        'd0':scale,'scale_reference_pairs':len(positive),'note':'Reference-relative descriptor score, not probability of global novelty; recalibrated scale differs from old Fe-only score.'}
    write(OUT/'scored_results.json',result)
    public_source=next((s for s in pool['sources'] if s['file']==str(PUBLIC_PO)),None)
    lines=['# 19 条结构的跨金属骨架新颖性复核','',
        f'- 扩展参考：{result["reference_count"]} 份金属统一标记后的不同结构；候选固定为原 19 条。',
        f'- 标准空间匹配命中：{sum(bool(r["standard_matches"]) for r in rows)}/19；宽松容差新增命中：{sum(bool(r["loose_only_matches"]) for r in rows)}/19。',
        f'- 前态匹配命中：{sum(bool(r["pre_relax_standard_matches"]) for r in rows)}/19；相同大小 primitive 单胞商图同构命中：{sum(bool(r["equal_primitive_quotient_graph_matches"]) for r in rows)}/19。',
        '- **结论限于已覆盖的历史 MP 快照和本地参考，不是当前 MP 全库的新颖性证明。**','',
        '## 参考覆盖','',
        '公开数据采用 MP 2018-10-18 历史快照，共 83,989 条结构，含 P/O 的 8,210 条全部进入化学空间筛选；无额外能量或稳定性过滤。来源与 SHA256 由 [matminer 官方元数据](https://github.com/hackingmaterials/matminer/blob/master/matminer/datasets/dataset_metadata.json) 核验。',
        '另合并本地 17,774 条衍生结构及 572 条 MP 缓存记录；衍生结构不冒充 MP 原始条目。']
    if public_source:
        lines += [f'历史 MP 中有 {public_source["eligible_before_exact_hash_dedup"]} 条满足约化 M:P:O=3:4:16；按此比例过滤不会排除仅金属替换、原胞/超胞变换可得到的等价骨架。',
                  '', '| 历史 MP 参考金属 | 比例相容记录数（混合金属可重复计数） |','|---|---:|']
        lines += [f'| {metal} | {n} |' for metal,n in sorted(public_source['eligible_metal_coverage'].items())]
    lines += ['', '## 比较规则','',
        '删除 Li/Na/K/Rb/Cs/Fr，保留 P、O 身份，将其他金属统一为 M（实现中用 Fe 标签）；支持同位点不同金属的全占位混合，不允许 P/O 互换。',
        '标准 StructureMatcher：ltol=0.20、stol=0.30、angle_tol=5°；敏感性检查：0.30、0.50、10°。两者均允许体积缩放、原胞规约和超胞匹配。',
        '商图同构只比较相同 primitive 位点数，Fe/M–O 截断 2.5 Å、P–O 2.0 Å；不等同于对所有周期无限网完成严格拓扑分类。',
        f'0–10 分延续距离映射 S=10d/(d+d₀)，但参考尺度重标定为 d₀={scale:.8f}。匹配已知者为0；该分数不是发现概率，也不能直接与旧 Fe-only 分数作同尺度比较。',
        '', '## 逐结构结果','',
        '| 编号 | 结构 ID | 标准匹配 | 宽松新增 | 前态匹配 | 商图匹配 | 最近距离 | 跨金属评分 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| S{r["number"]:02d} | {r["id"]} | {len(r["standard_matches"])} | {len(r["loose_only_matches"])} | {len(r["pre_relax_standard_matches"])} | {len(r["equal_primitive_quotient_graph_matches"])} | {r["nearest_normalized_metal_references"][0]["distance"]:.6f} | {r["crossmetal_score_0_10"]:.2f} |')
    lines += ['', '## 未覆盖边界','',
        '当前 MP 在线 API 返回 HTTP 403；历史快照不包含后续新增结构。其他阴离子、部分空位及需移除碱土金属客体的框架不在本轮规范化范围内。需要可用的当前 MP 访问或全量 P–O 导出，才能完成最新全库检索。',
        '可复算明细：`scored_results.json`；全部参考的原始 ID、化学式、金属元素和源文件 SHA256 见 `local_references.json`。原 19 条晶体、弛豫记录及现有 HTML 未修改。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    buffer=io.StringIO();writer=csv.writer(buffer)
    writer.writerow(['number','id','standard_matches','loose_matches','pre_matches','quotient_graph_matches','nearest_distance','crossmetal_score_0_10'])
    for r in rows:writer.writerow([r['number'],r['id'],len(r['standard_matches']),len(r['loose_only_matches']),len(r['pre_relax_standard_matches']),len(r['equal_primitive_quotient_graph_matches']),r['nearest_normalized_metal_references'][0]['distance'],r['crossmetal_score_0_10']])
    (OUT/'summary_scored.csv').write_text(buffer.getvalue())
    print(json.dumps({'report':str(OUT/'report.md'),'score_min':min(r['crossmetal_score_0_10'] for r in rows),'score_max':max(r['crossmetal_score_0_10'] for r in rows)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['download','public_download','prepare','evaluate','prepare_public','evaluate_public','summarize_public']);a=p.parse_args()
    if a.mode.endswith('_public'):
        OUT=OUT/'with_public_mp';a.mode=a.mode.removesuffix('_public')
    globals()[a.mode]()

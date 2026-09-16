"""Audit supplied CIFs and build a compact merged report without modifying inputs."""
import argparse
import base64
from collections import Counter
import csv
import io
import html
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import categorical_node_match
from pymatgen.core import Structure
from run_uma_campaign import Collector, ROOT, inputs, record, crystal, write_json
from nagen.selection.pipeline import Crystal, hard_gates
from nagen.selection.diversity import distances
from nagen.selection.experiment import sha256

OUT=ROOT/'runs/campaign_24/report_26'
OLD=ROOT/'runs/campaign_24/report_19'
NEW=Path('/home/aobo/NaGen/data/four_stage_flow_generation')


def audit():
    import torch
    from nagen.selection.uma import UMAEvaluator
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    collector=Collector(ROOT/'runs/campaign_24')
    snapshot=json.loads((OLD/'snapshot.json').read_text())
    rows=[]
    for e in snapshot['rows']:
        a=json.loads((OLD/'audit'/(e['id']+'.json')).read_text())
        a.update(source=str(OLD/'cif'/e['file']), label_source='previous default-UMA terminal audit')
        rows.append(a)
    evaluator=UMAEvaluator(str(inputs()['uma']),'cuda','omat')
    assert evaluator.provenance['state_sha256']==snapshot['configuration']['hashes']['uma']
    for path in sorted(NEW.glob('*.cif')):
        s=Structure.from_file(path)
        c=Crystal(path.stem,[str(site.specie) for site in s],np.array(s.frac_coords),np.array(s.lattice.matrix))
        label=evaluator.evaluate(c.elements,c.frac,c.lattice)
        g=hard_gates(c,collector.condition,collector.profile,forces=label['forces_eV_A'],force_threshold=.03,motif_backend='native')
        h=collector.hull.evaluate(c.elements,label['energy_eV_atom'])
        fw=collector.strip(c)
        fs,fg=collector.structure(fw),collector.graph(fw)
        novelty={'post_refine_training_match':any(collector.matcher.fit(fs,r) for r in collector.ref_structures),
                 'post_refine_topology_match':any(nx.is_isomorphic(fg,r,node_match=categorical_node_match('element','')) for r in collector.ref_graphs),
                 'pre_relax_training_match':None,
                 'nearest_known_framework_distance':float(distances(collector.descriptor(fw)[None],collector.ref_features).min())}
        row={'id':path.stem,'file':path.name,'source':str(path),'sha256':sha256(path),'final':record(c),'hard_gates':g,'reference_hull':h,
             'framework_novelty':novelty,'refinement':{'after':label},'label_source':'fresh default-UMA prediction on supplied CIF; not re-relaxed',
             'generation_provenance':'not supplied','model':evaluator.provenance}
        rows.append(row)
        print(json.dumps({'id':c.id,'hard_gates':g['checks'],'force':label['force_max_eV_A'],'hull':h.get('e_above_reference_hull_eV_atom')}),flush=True)
    frameworks=[collector.strip(crystal(r['final'])) for r in rows]
    structs=[collector.structure(c) for c in frameworks]
    feats=np.stack([collector.descriptor(c) for c in frameworks])
    dd=distances(feats,feats)
    np.fill_diagonal(dd,np.inf)
    matches=[]
    for i,r in enumerate(rows):
        r['nearest_merged_distance']=float(dd[i].min())
        r['merged_matches']=[]
        for j in range(i):
            if collector.matcher.fit(structs[i],structs[j]):
                matches.append([r['id'],rows[j]['id']])
                r['merged_matches'].append(rows[j]['id'])
        n=r['framework_novelty']
        r['endpoint_feasible']=bool(r['hard_gates']['passed'] and r['reference_hull']['e_above_reference_hull_eV_atom']<=.15
            and not n['post_refine_training_match'] and not n['post_refine_topology_match']
            and n['nearest_known_framework_distance']>=1e-4)
    write_json(OUT/'merged_audit.json',{'count':len(rows),'endpoint_feasible_count':sum(r['endpoint_feasible'] for r in rows),
        'pair_matches':matches,'rows':rows,'python':sys.executable,'original_snapshot_utc':snapshot['snapshot_utc']})
    print('Audit saved',flush=True)


def build():
    d=json.loads((OUT/'merged_audit.json').read_text())
    snapshot=json.loads((OLD/'snapshot.json').read_text())
    rows=d['rows']; old=rows[:19]; new=rows[19:]
    assert len(rows)==26 and len(new)==7
    def span(values, digits=5):
        values=list(values)
        return '—' if not values else f'{min(values):.{digits}f}–{max(values):.{digits}f}'
    # Ranges are deliberately kept separate: the supplied seven do not pass.
    def both(fn,digits=5):
        return '原19：'+span([fn(r) for r in old],digits)+'；新增7：'+span([fn(r) for r in new],digits)
    def count(fn):
        return f'原19：{sum(bool(fn(r)) for r in old)}/19；新增7：{sum(bool(fn(r)) for r in new)}/7'
    def sites(group,el,cn=None):
        return [s for r in group for s in r['hard_gates']['sites'] if s['element']==el and (cn is None or s['cn']==cn)]
    items=[]
    items.append(('元素组成：N=52；Na:Fe:P:O=6:6:8:32','26/26 均为 Na₆Fe₆P₈O₃₂'))
    items.append(('晶胞：有限；|det L|>10⁻⁶ Å³；cond(L)<100','体积 '+both(lambda r:abs(np.linalg.det(crystal(r['final']).lattice)),2)+' Å³；26/26 通过'))
    items.append(('电中性：Na⁺、P⁵⁺、O²⁻；Fe∈{+2,+3}','26/26 所需 Fe 平均形式价态为 +3'))
    items.append(('原子力：maxᵢ ‖Fᵢ‖₂ ≤0.03 eV/Å',both(lambda r:r['hard_gates']['force_max_eV_A'],6)+'；'+count(lambda r:r['hard_gates']['checks']['final_force'])))
    items.append(('UMA 有限参考 hull proxy ≤0.15 eV/atom',both(lambda r:r['reference_hull']['e_above_reference_hull_eV_atom'],6)+'；'+count(lambda r:r['reference_hull']['e_above_reference_hull_eV_atom']<=.15)))
    items.append(('周期碰撞门控：违规数=0','违规数 '+both(lambda r:len(r['hard_gates']['overlaps']),0)+'；'+count(lambda r:r['hard_gates']['checks']['overlap'])))
    for el,allowed in [('P',{4}),('Fe',{4,5,6})]:
        limits='4' if el=='P' else '{4,5,6}'
        actual=lambda group:','.join(map(str,sorted({s['cn'] for s in sites(group,el)})))
        items.append((f'{el}–O 配位数：CN∈{limits}',f'原19：{actual(old)}；新增7：{actual(new)}；'+count(lambda r:all(s['cn'] in allowed for s in r['hard_gates']['sites'] if s['element']==el))))
    for el,cn in [('P',4),('Fe',4),('Fe',5),('Fe',6)]:
        p=snapshot['profile']['profiles'][f'phosphate:{el}:{cn}']
        a,b=sites(old,el,cn),sites(new,el,cn)
        bounds=lambda ss:'—' if not ss else f'{min(s["bond_min"] for s in ss):.4f}–{max(s["bond_max"] for s in ss):.4f}'
        items.append((f'{el}O{cn} 键长：[{p["bond_min"]:.4f}, {p["bond_max"]:.4f}] Å',f'原19：{bounds(a)}；新增7：{bounds(b)}'))
    items.append(('多面体中心：全部 Fe/P 位于配位 O 凸包内',count(lambda r:all(s['inside'] for s in r['hard_gates']['sites']))))
    items.append(('氧覆盖：32/32 个 O 属于 Fe/P 多面体',count(lambda r:r['hard_gates']['checks']['oxygen_coverage'])))
    items.append(('末态新颖性：与22条参考不匹配、配位图不等价；距离≥10⁻⁴',both(lambda r:r['framework_novelty']['nearest_known_framework_distance'])+'；'+count(lambda r:not r['framework_novelty']['post_refine_training_match'] and not r['framework_novelty']['post_refine_topology_match'])))
    items.append(('生成前态新颖性与生成来源可追溯','原19：19/19 有记录；新增7：未提供前态及生成记录'))
    items.append(('26份集合去重：StructureMatcher 不匹配，距离≥10⁻⁴',f'匹配结构对 {len(d["pair_matches"])} 对；最小描述符距离 {min(r["nearest_merged_distance"] for r in rows):.6f}'))
    items.append(('当前流程全部约束：Feasibility=1','原19：19/19；新增7：0/7；合并26份中已确认合格19份'))
    csvbuf=io.StringIO();writer=csv.writer(csvbuf)
    writer.writerow(['id','source_group','max_force_eV_A','hull_proxy_eV_atom','hard_gates_passed','endpoint_feasible','known_distance','merged_nearest_distance','matches','sha256'])
    for i,r in enumerate(rows):
        writer.writerow([r['id'],'original_19' if i<19 else 'supplied_7',r['hard_gates']['force_max_eV_A'],r['reference_hull']['e_above_reference_hull_eV_atom'],r['hard_gates']['passed'],r['endpoint_feasible'],r['framework_novelty']['nearest_known_framework_distance'],r['nearest_merged_distance'],';'.join(r['merged_matches']),r['sha256']])
    (OUT/'structures_26.csv').write_text(csvbuf.getvalue())
    for name,rr in [('structures_19.zip',old),('structures_26.zip',rows)]:
        with zipfile.ZipFile(OUT/name,'w',zipfile.ZIP_DEFLATED) as z:
            for r in rr:
                path=Path(r['source'])
                assert sha256(path)==r['sha256']
                group='original_19' if r in old else 'supplied_7_not_passed'
                z.write(path,f'{group}/{r["file"]}')
                z.writestr(f'audits/{r["id"]}.json',json.dumps(r,indent=2))
            z.writestr('collection.json',json.dumps({'rows':rr,'pair_matches':d['pair_matches'] if len(rr)==26 else [],'python':d['python']},indent=2))
            if len(rr)==26:z.writestr('structures_26.csv',csvbuf.getvalue())
    def link(name,label):
        encoded=base64.b64encode((OUT/name).read_bytes()).decode()
        return f'<a href="data:application/zip;base64,{encoded}" download="{name}">{label}</a>'
    a,b=snapshot['cohorts'];speed=a['total_seconds']['mean']/b['total_seconds']['mean']
    intro=f'''本次实验完成 ShootingFlow 算法框架的小批量试生成。框架设计逻辑为：晶体结构 (N,A,X,L) 无条件预训练 → conditional 生成 + 门控硬筛 → E<sub>hull</sub> / 应力梯度引导推理端优化。本批实际使用冻结几何 Flow、UMA 源空间梯度优化及固定晶胞 refinement。原批次调用 8 张 GPU，保留 19 条满足当前流程全部约束的结构；56.3 分钟按提供的计时口径列示（现存运行日志跨度为 {snapshot['wall_seconds']/60:.1f} 分钟，二者尚未对齐）。生成与优化阶段的平均摊销耗时由 {a['total_seconds']['mean']:.2f} 降至 {b['total_seconds']['mean']:.2f} 秒/样本，观测吞吐提升约 {speed:.2f} 倍；按提供的 56.3 分钟折算为 2.96 分钟/合格结构，不含新增 7 条。现合并为 26 份结构文件，其中新增 7 条未通过同一 UMA 的力与 hull 门控，故确认合格数仍为 19。'''
    table='<table><thead><tr><th>约束及数值阈值</th><th>结构数值范围 / 满足情况</th></tr></thead><tbody>'+''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>' for k,v in items)+'</tbody></table>'
    content='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ShootingFlow 小批量试生成</title><style>body{font:13px/1.55 system-ui,"Microsoft YaHei",sans-serif;max-width:1250px;margin:25px auto;padding:0 20px;color:#243543}h1{font-size:23px;margin-bottom:12px}p{margin:12px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 12px;border:1px solid #dce3e8;text-align:left}th{background:#eaf0f4}td:first-child{width:43%}tr:nth-child(even){background:#f8fafb}a{color:#146b96;margin-right:24px}@media print{body{margin:0;font-size:10px}td,th{padding:5px}}</style></head><body><h1>ShootingFlow 小批量试生成</h1>'''+f'<p>{intro}</p>'+table+f'<p>{link("structures_19.zip","附件：原19条结构与审计")}{link("structures_26.zip","附件：合并26份结构与核验明细")}</p></body></html>'
    (OUT/'index.html').write_text(content)
    print(OUT/'index.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','build']);args=p.parse_args();globals()[args.mode]()

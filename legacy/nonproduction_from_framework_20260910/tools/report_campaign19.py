"""Two-table report of the first 19 actual campaign selections."""
import argparse
import base64
import html
import json
import re
from pathlib import Path
import numpy as np
import report_campaign17 as base

base.OUT = base.RUN / 'report_19'
base.COUNT = 19
OUT = base.OUT


def prepare():
    base.prepare()
    base.convert()
    for path in (OUT/'framework').glob('*.vesta'):
        text = path.read_text()
        text = re.sub(r'SBOND\n.*?\nSITET', 'SBOND\n  1 Fe O 0.0 2.5 0 1 1 0 1 0.20 2.0 127 127 127\n  2 P O 0.0 2.0 0 1 1 0 1 0.20 2.0 127 127 127\n  0 0 0 0\nSITET', text, flags=re.S)
        def color(match):
            fields = match.group(0).split()
            element = re.sub(r'\d', '', fields[1])
            radius, sphere, poly = {'Fe':(1.26,[190,143,35],[244,207,32]),
                'P':(1.10,[179,149,193],[179,149,193]), 'O':(.74,[232,20,25],[232,20,25])}[element]
            fields[2] = str(radius)
            fields[3:9] = list(map(str, sphere+poly))
            fields[9] = '165'
            return '  '+' '.join(fields)
        text = re.sub(r'^\s*\d+\s+(?:Fe|P|O)\d*\s+\d+\.\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+(?:\s+0)?$', color, text, flags=re.M)
        text = re.sub(r'MODEL\s+\d+\s+\d+\s+\d+', 'MODEL 2 1 0', text)
        text = text.replace('POLYP\n 204 1  1.000 180 180 180', 'POLYP\n 165 1  0.7 160 160 160')
        path.write_text(text)


def build():
    data = json.loads((OUT/'snapshot.json').read_text())
    rows = data['rows']
    assert len(rows) == 19
    result = []
    for r in rows:
        g, n = r['hard_gates'], r['framework_novelty']
        f, h = g['force_max_eV_A'], r['reference_hull']['e_above_reference_hull_eV_atom']
        novel = not any(n[k] for k in ('pre_relax_training_match','post_refine_training_match','post_refine_topology_match'))
        feasible = all(g['checks'].values()) and h<=.15 and novel and not n['selected_structure_matches'] and r['nearest_snapshot_distance']>=1e-4 and r['cif_roundtrip']
        assert feasible
        png = (OUT/'images'/(r['id']+'.png')).read_bytes()
        src = 'data:image/png;base64,'+base64.b64encode(png).decode()
        image = f'<img src="{src}" alt="{r["id"]} Fe–P–O" loading="lazy" onclick="document.getElementById(\'zoomimg\').src=this.src;document.getElementById(\'zoom\').showModal()">'
        label = f'S{r["number"]:02}<br>'+r['id'].replace('uma_campaign_seed','').replace('uma_dflow_seed','')
        cif = base.data_link((OUT/'cif'/r['file']).read_bytes(),'chemical/x-cif',r['file'],label)
        cn = '/'.join(str(r['fe_cn'].get(str(k),0)) for k in (4,5,6))
        bonds = lambda key:'–'.join(f'{v:.3f}' for v in r[key])
        result.append([image,cif,'6:6:8:32',f'{r["volume"]:.2f}<br>{r["cell_condition"]:.3f}','+3',f'{f:.6f}',f'{h:.6f}',cn,'8/8',bonds('fe_bonds'),bonds('p_bonds'),f'{r["inside_count"]}/14',f'{r["oxygen_covered"]}/32','0',f'{n["nearest_known_framework_distance"]:.5f} / {r["nearest_snapshot_distance"]:.5f}<br>不匹配 / 不等价',f'{r["score"]:.2f}',f'<b class="pass">{int(feasible)}</b>'])
    headers=['Fe–P–O · VESTA<br>Fe 黄 / P 紫 / O 红','结构 / CIF','组成<br>Na:Fe:P:O','体积>0 Å³<br>cond(L)<100','Fe 平均形式价态<br>电中性','最大力 ≤0.03<br>eV/Å','UMA hull proxy ≤0.15<br>eV/atom','Fe 四/五/六<br>配位中心数','P 四配位<br>中心数','Fe–O 键长<br>Å','P–O 键长<br>Å','多面体中心<br>在 O 凸包内','氧覆盖','碰撞<br>违规数','已知/集合距离 ≥10⁻⁴<br>已知结构 / 配位图','新颖性<br>0–10','Feasibility<br>1=全部通过']
    table1 = base.table(headers,result)
    selected = {r['id'] for r in rows if r['timing']}
    # Read only the exact record IDs frozen into this report, not later workers' output.
    records = [json.loads((base.RUN/'records'/f'{r["index"]:08d}.json').read_text()) for r in data['timing_rows']]
    performance=[]
    for c in data['cohorts']:
        cohort = [r for r in records if r.get('generation_and_relaxation_inference','default')==c['mode']]
        gates = sum(r.get('hard_gates',{}).get('passed',False) for r in cohort)
        eligible = sum(r['status']=='eligible' for r in cohort)
        performance.append(['default / 单结构' if c['mode']=='default' else 'batch_tf32 / 4结构批次',str(c['n']),str(sum(r['status'] not in ('unsafe','failed') for r in cohort)),'7 类硬门控 + hull + 新颖性/去重 + CIF',str(gates),str(eligible),str(sum(r['id'] in selected for r in cohort)),f'{c["generation_seconds"]["mean"]:.2f}',f'{c["relaxation_seconds"]["mean"]:.2f}',f'{c["total_seconds"]["mean"]:.2f}',f'{c["total_seconds"]["sum"]/3600:.2f} worker·h'])
    elapsed=data['wall_seconds']
    performance.append(['本轮累计',str(len(records)),str(sum(r['status'] not in ('unsafe','failed') for r in records)),'同上',str(sum(r.get('hard_gates',{}).get('passed',False) for r in records)),str(sum(r['status']=='eligible' for r in records)),str(len(selected)),'—','—','—',f'{elapsed/3600:.3f} h 墙钟 / 8 GPU'])
    performance.append(['原有结构保留','128','128','原实验口径','9','1','1','—','—','—','—'])
    performance.append(['本页合计','—','—','当前实际流水线','—','—','19','—','—','—',f'{data["configuration"]["created_utc"]}<br>→ {data["snapshot_utc"].split(".")[0]} UTC'])
    a,b=data['cohorts']
    performance.append(['摊销吞吐提升','—','—','default / batch 均值比','—','—','—','—','—',f'{a["total_seconds"]["mean"]/b["total_seconds"]["mean"]:.2f}×','批次不同；非同种子消融'])
    table2=base.table(['计算阶段','完成样本','预筛通过','筛选约束','7类硬门控通过','硬门控+hull通过','本页可行结构','生成均值 s','弛豫均值 s¹','阶段总均值 s¹','耗时 / 口径'],performance)
    style='''body{font:13px/1.45 system-ui,"Microsoft YaHei",sans-serif;color:#243440;margin:20px;background:#f4f6f8}h2{font-size:20px;margin:20px 0 12px}.table-scroll{overflow:auto;background:white;border:1px solid #d6dde3;border-radius:8px;margin-bottom:24px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #e2e7eb;text-align:center;white-space:nowrap}th{background:#e9eef2;font-size:12px;position:sticky;top:0}tr:last-child td{border:0}tr:nth-child(even){background:#fafbfc}img{width:210px;height:175px;object-fit:contain;cursor:zoom-in}a{color:#236e8d;text-decoration:none}.pass{color:#18794e;font-size:20px}dialog{border:0;border-radius:10px;padding:12px}dialog::backdrop{background:#142637bb}dialog img{width:auto;height:auto;max-width:90vw;max-height:87vh}button{float:right;padding:5px 12px}th[title]{text-decoration:underline dotted;cursor:help}'''
    table1=table1.replace('<th>Fe–O 键长','<th title="冻结逐位点阈值：Fe4 [1.763542,2.432019]；Fe5 [1.443462,2.581179]；Fe6 [1.691332,2.589917] Å；表内为结构实际极值">Fe–O 键长')
    table1=table1.replace('<th>P–O 键长','<th title="P4 每条键必须在 [1.340993,1.889891] Å；P–O/Fe–O 近邻截断分别2.0/2.5 Å">P–O 键长')
    table1=table1.replace('<th>UMA hull','<th title="同一 UMA checkpoint 的 4,385 条有限参考相 proxy；不是 DFT 完整凸包">UMA hull')
    table1=table1.replace('<th>新颖性<br>','<th title="22条参考；S=10d/(d+d0)，d0为参考间非重复距离中位数；任一已知匹配命中记0分。不是全球新颖概率">新颖性<br>')
    table1=table1.replace('<th>Feasibility','<th title="实际流水线的7类硬门控、有限hull、已知骨架新颖性、集合去重及CIF round-trip均通过=1，否则=0；不是HybridOptimization文档的统一可行域">Feasibility')
    table2=table2.replace('<th>弛豫均值','<th title="批处理采用整批墙钟/批大小的摊销时间；不是单结构排队至完成延迟">弛豫均值').replace('<th>阶段总均值','<th title="生成+摊销初始弛豫+严格refinement；不含初始化和收集器">阶段总均值')
    page=f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>19条结构 · 约束与计算效率</title><style>{style}</style></head><body><h2>1. 19 条结构：实际约束与 Feasibility</h2>{table1}<h2>2. 生成—优化—筛选：计算效率</h2>{table2}<dialog id="zoom"><button onclick="this.parentElement.close()">×</button><img id="zoomimg" alt="Fe–P–O"></dialog></body></html>'
    (OUT/'index.html').write_text(page)
    base.save(OUT/'report_checks.json',{'structures':19,'tables':2,'feasible':19,'current_pipeline_not_hybridoptimization':True,'python':data['python'],'snapshot_utc':data['snapshot_utc']})
    print(OUT/'index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','render','build'])
    mode=parser.parse_args().mode
    if mode=='render':base.render()
    else:globals()[mode]()

"""Build a viewport-fitted, one-table summary of the frozen 19 passing CIFs."""
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/campaign_24'
OUT=RUN/'report_19_onepage'
OUT.mkdir(exist_ok=True)
d=json.loads((RUN/'report_19/snapshot.json').read_text())
rows=d['rows']
assert len(rows)==19
assert all(r['hard_gates']['passed'] and r['reference_hull']['e_above_reference_hull_eV_atom']<=.15 for r in rows)
def extent(values,n=4):
    values=list(values)
    return f'{min(values):.{n}f}–{max(values):.{n}f}'
def sites(el,cn=None):
    return [s for r in rows for s in r['hard_gates']['sites'] if s['element']==el and (cn is None or s['cn']==cn)]
items=[
('组成 / 结构一致性：Na₆Fe₆P₈O₃₂；N=52；坐标、晶胞有限','19/19；Na:Fe:P:O = 6:6:8:32'),
('电中性：Na⁺、P⁵⁺、O²⁻；Fe 采用 +2/+3 模型','19/19；所需 Fe 平均形式价态 +3'),
('晶胞：|det L|>10⁻⁶ Å³；cond(L)<100',f'V = {extent((r["volume"] for r in rows),2)} Å³；cond = {extent((r["cell_condition"] for r in rows),3)}'),
('原子力：maxᵢ ‖Fᵢ‖₂ ≤0.030000 eV/Å',extent((r['hard_gates']['force_max_eV_A'] for r in rows),6)+' eV/Å'),
('同 UMA 有限参考 hull proxy ≤0.150000 eV/atom',extent((r['reference_hull']['e_above_reference_hull_eV_atom'] for r in rows),6)+' eV/atom'),
('P–O 配位：每个 P 为四配位；r≤2.0 Å','152/152 个 P：CN=4'),
('Fe–O 配位：有训练支持的配位类别；r≤2.5 Å','114/114 个 Fe：CN∈{4,5,6}'),
]
for el,cn in [('P',4),('Fe',4),('Fe',5),('Fe',6)]:
    ref=d['profile']['profiles'][f'phosphate:{el}:{cn}'];ss=sites(el,cn)
    items.append((f'{el}O{cn} 逐键阈值：[{ref["bond_min"]:.4f}, {ref["bond_max"]:.4f}] Å',f'{min(s["bond_min"] for s in ss):.4f}–{max(s["bond_max"] for s in ss):.4f} Å；全部通过'))
items += [
('Fe/P 位于 O 配位凸包内；所有 O 被多面体覆盖','266/266 个中心在内部；608/608 个 O 被覆盖'),
('周期短接触门控：违规数=0','19/19；违规数均为 0'),
('已知骨架新颖性：前/后态结构不匹配，末态配位图不等价',f'19/19；最近参考描述符距离 {extent((r["framework_novelty"]["nearest_known_framework_distance"] for r in rows),5)}'),
('集合去重：结构不匹配；描述符距离≥10⁻⁴',f'重复对 0；集合最近邻距离 {extent((r["nearest_snapshot_distance"] for r in rows),5)}'),
('CIF round-trip / SHA256；全部约束 Feasibility=1','19/19 校验通过；19/19 全部可行（当前流水线）')]
archive=OUT/'structures_19.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for r in rows:
        source=RUN/'report_19/cif'/r['file']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==r['sha256']
        z.write(source,'cif/'+r['file'])
        z.write(RUN/'report_19/audit'/(r['id']+'.json'),'audit/'+r['id']+'.json')
    z.writestr('structures_19.json',json.dumps({'rows':rows,'configuration':d['configuration']},ensure_ascii=False,indent=2))
uri='data:application/zip;base64,'+base64.b64encode(archive.read_bytes()).decode()
a,b=d['cohorts'];speed=a['total_seconds']['mean']/b['total_seconds']['mean']
timed=d['timing_rows']
workers=d['configuration']['workers']
timing_audit={'completed_new_samples':len(timed),'workers':workers,'snapshot_utc':d['snapshot_utc'],
              'scope':'Frozen snapshot of completed new candidates, including rejections; original retained finalist is not timed here.',
              'method':'Sum per-candidate stage timer / worker count. Equivalent parallel cost, not measured elapsed wall time.',
              'generation_definition':'Flow sampling plus 12 source-space DFlow steps; excludes subsequent FIRE relaxation.',
              'stages':{}}
for key in ('generation_seconds','relaxation_seconds','refinement_seconds','total_seconds'):
    seconds=sum(r['timing'].get(key,0.) for r in timed)
    assert abs(seconds-sum(c[key]['sum'] for c in d['cohorts']))<1e-5
    timing_audit['stages'][key]={'worker_seconds':seconds,'worker_minutes':seconds/60,'eight_gpu_equivalent_minutes':seconds/workers/60}
generation_minutes=timing_audit['stages']['generation_seconds']['eight_gpu_equivalent_minutes']
total_minutes=timing_audit['stages']['total_seconds']['eight_gpu_equivalent_minutes']
(OUT/'timing_audit.json').write_text(json.dumps(timing_audit,ensure_ascii=False,indent=2))
body=''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>' for k,v in items)
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ShootingFlow · 19条合格结构</title>
<style>*{{box-sizing:border-box}}html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#eef2f5;color:#203442;font-family:system-ui,"Microsoft YaHei",sans-serif}}main{{position:absolute;left:50%;top:50%;width:1280px;padding:24px 30px;background:white;transform-origin:center;box-shadow:0 8px 32px #20344212}}h1{{font-size:25px;margin:0 0 9px}}.top{{display:flex;justify-content:space-between;align-items:center}}.badge{{color:#177a55;font-size:20px;font-weight:750}}p{{font-size:14px;line-height:1.55;margin:6px 0}}.logic{{color:#45616f}}.metrics{{padding:9px 13px;background:#edf5f7;border-radius:6px;margin:9px 0 12px}}table{{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:6px 10px;border:1px solid #dce5ea;line-height:1.3}}th{{background:#e6eef2}}td:first-child{{width:55%}}tr:nth-child(even){{background:#f8fafb}}tbody tr:last-child{{background:#eaf5ef;font-weight:700}}footer{{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-top:10px;font-size:10px;color:#6a7c86}}a{{font-size:13px;font-weight:650;color:#167497;white-space:nowrap}}@media print{{@page{{size:A4 landscape;margin:8mm}}html,body{{overflow:visible;background:white}}main{{position:static;transform:none!important;width:100%;padding:0;box-shadow:none}}h1{{font-size:18px}}p{{font-size:10px}}table{{font-size:9px}}th,td{{padding:4px 7px}}footer{{font-size:8px}}}}</style></head><body><main id="sheet">
<div class="top"><h1>ShootingFlow 小批量试生成</h1><div class="badge">19 / 19 全部通过</div></div>
<p>本次实验完成 ShootingFlow 算法框架的小批量试生成，调用 <b>8 张 GPU</b>，本页保留 <b>19 条满足当前流水线全部约束的结构</b>。统计截点已完成 <b>{len(timed):,} 个新样本</b>；生成阶段耗时 <b>{generation_minutes:.2f} 分钟（8 GPU 等效）</b>¹。</p>
<p class="logic">框架设计逻辑：晶体结构 (N,A,X,L) 无条件预训练 → conditional 生成 + 门控硬筛 → E<sub>hull</sub> / 应力梯度引导推理端优化²。</p>
<p class="metrics">生成＋弛豫优化的记录阶段合计 <b>{total_minutes:.2f} 分钟（8 GPU 等效）</b>；平均摊销耗时 <b>{a['total_seconds']['mean']:.2f} → {b['total_seconds']['mean']:.2f} 秒/样本</b>，观测吞吐提升 <b>{speed:.2f}×</b>。</p>
<table><thead><tr><th>约束与数值阈值</th><th>19 条合格结构的范围 / 满足情况</th></tr></thead><tbody>{body}</tbody></table>
<footer><span>¹ 逐样本计时累加÷8，非实测墙钟；不计开发间隔、模型加载及独立收集器。生成含12步DFlow；计时含淘汰样本，不含原有1条。<br>² 本批为冻结几何Flow + UMA源空间优化 + 固定晶胞refinement；阶段均值来自不同批次。Hull为4,385相有限proxy，新颖性参考22条。</span><a href="{uri}" download="structures_19.zip">附件：19条 CIF · 逐结构数值 · 审计记录 ↓</a></footer>
</main><script>function fit(){{const s=document.getElementById('sheet');const k=Math.min((innerWidth-20)/s.offsetWidth,(innerHeight-20)/s.offsetHeight);s.style.transform='translate(-50%,-50%) scale('+Math.max(.01,k)+')'}}addEventListener('resize',fit);addEventListener('load',fit);fit();</script></body></html>'''
page=page.replace('</footer>', '<a href="vesta_19.html">下一页：晶胞 →</a></footer>')
(OUT/'index.html').write_text(page)
with zipfile.ZipFile(archive) as z:assert len([n for n in z.namelist() if n.endswith('.cif')])==19
assert page.count('<table>')==1 and '<img' not in page and page.count('<tr>')==len(items)+1
print(OUT/'index.html')

# Page 2 contains only the VESTA renderings and numbered chemical formulae.
figures=[]
for r in rows:
    audit=json.loads((RUN/'report_19/audit'/(r['id']+'.json')).read_text())
    counts=Counter(audit['final']['A'])
    assert counts=={'Na':6,'Fe':6,'P':8,'O':32}
    formula=''.join(e+(f'<sub>{counts[e]}</sub>' if counts[e]!=1 else '') for e in ('Na','Fe','P','O'))
    png=(RUN/'report_19/images'/(r['id']+'.png')).read_bytes()
    assert png.startswith(b'\x89PNG\r\n\x1a\n')
    encoded=base64.b64encode(png).decode()
    figures.append(f'<figure data-id="{html.escape(r["id"])}"><img src="data:image/png;base64,{encoded}" alt="S{r["number"]:02} {html.escape(r["id"])} VESTA"><figcaption><b>S{r["number"]:02}</b> {formula}</figcaption></figure>')
gallery='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>19个晶胞 · VESTA</title><style>
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#eef2f5;font-family:system-ui,"Microsoft YaHei",sans-serif;color:#203442}
main{position:absolute;left:50%;top:50%;width:1280px;padding:18px;display:grid;grid-template-columns:repeat(5,1fr);gap:10px;background:white;transform-origin:center}
figure{margin:0;border:1px solid #dfe6eb;border-radius:6px;overflow:hidden;background:white}img{display:block;width:100%;height:191px;object-fit:contain}figcaption{text-align:center;padding:5px 4px 9px;font-size:16px;line-height:1.3}figcaption b{font-size:12px;color:#73848e;margin-right:9px}sub{font-size:70%}
@media print{@page{size:A4 landscape;margin:6mm}html,body{overflow:visible;background:white}main{position:static;transform:none!important;width:100%;padding:0;gap:5px}img{height:37mm}figcaption{font-size:11px;padding:2px}figure{break-inside:avoid}}
</style></head><body><main id="sheet">'''+''.join(figures)+'''</main><script>function fit(){const s=document.getElementById('sheet');const k=Math.min((innerWidth-16)/s.offsetWidth,(innerHeight-16)/s.offsetHeight);s.style.transform='translate(-50%,-50%) scale('+Math.max(.01,k)+')'}addEventListener('resize',fit);addEventListener('load',fit);fit();</script></body></html>'''
assert gallery.count('<figure ')==19 and gallery.count('<figcaption>')==19 and '<table' not in gallery
(OUT/'vesta_19.html').write_text(gallery)
print(OUT/'vesta_19.html')

# Continuous two-page document; each screen-sized page has no internal scrolling.
summary_inner=page.split('<main id="sheet">',1)[1].split('</main>',1)[0]
summary_inner=summary_inner.replace('href="vesta_19.html"','href="#page-2"')
combined='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ShootingFlow · 连续两页</title><style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef2f5;color:#203442;font-family:system-ui,"Microsoft YaHei",sans-serif;overflow-x:hidden}
.page{height:100vh;position:relative;overflow:hidden}.page+.page{border-top:1px solid #dce5ea}.sheet{position:absolute;left:50%;top:50%;width:1280px;padding:24px 30px;background:white;transform-origin:center;box-shadow:0 8px 32px #20344212}
h1{font-size:25px;margin:0 0 9px}.top{display:flex;justify-content:space-between;align-items:center}.badge{color:#177a55;font-size:20px;font-weight:750}p{font-size:14px;line-height:1.55;margin:6px 0}.logic{color:#45616f}.metrics{padding:9px 13px;background:#edf5f7;border-radius:6px;margin:9px 0 12px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:6px 10px;border:1px solid #dce5ea;line-height:1.3}th{background:#e6eef2}td:first-child{width:55%}tr:nth-child(even){background:#f8fafb}tbody tr:last-child{background:#eaf5ef;font-weight:700}
footer{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-top:10px;font-size:10px;color:#6a7c86}a{font-size:13px;font-weight:650;color:#167497;white-space:nowrap}
.gallery{padding:18px;display:grid;grid-template-columns:repeat(5,1fr);gap:10px}figure{margin:0;border:1px solid #dfe6eb;border-radius:6px;overflow:hidden;background:white}img{display:block;width:100%;height:191px;object-fit:contain}figcaption{text-align:center;padding:5px 4px 9px;font-size:16px;line-height:1.3}figcaption b{font-size:12px;color:#73848e;margin-right:9px}sub{font-size:70%}
@media print{@page{size:A4 landscape;margin:8mm}html,body{background:white;overflow:visible}.page{height:194mm;overflow:hidden;break-after:page;page-break-after:always}.page:last-child{break-after:auto;page-break-after:auto}.page+.page{border:0}.sheet{position:static;transform:none!important;width:100%;padding:0;box-shadow:none}h1{font-size:18px}p{font-size:10px}table{font-size:9px}th,td{padding:4px 7px}footer{font-size:8px}a{font-size:9px}a[href="#page-2"]{display:none}.gallery{gap:5px}img{height:37mm}figcaption{font-size:11px;padding:2px}figure{break-inside:avoid}}
</style></head><body><section class="page" id="page-1"><main class="sheet">'''+summary_inner+'''</main></section><section class="page" id="page-2"><div class="sheet gallery">'''+''.join(figures)+'''</div></section><script>function fit(){document.querySelectorAll('.page').forEach(p=>{const s=p.querySelector('.sheet');const k=Math.min((p.clientWidth-20)/s.offsetWidth,(p.clientHeight-20)/s.offsetHeight);s.style.transform='translate(-50%,-50%) scale('+Math.max(.01,k)+')'})}addEventListener('resize',fit);addEventListener('load',fit);fit();</script></body></html>'''
assert combined.count('<section class="page"')==2
assert combined.count('<figure ')==19 and combined.count('<table>')==1
assert 'href="vesta_19.html"' not in combined
(OUT/'combined.html').write_text(combined)
print(OUT/'combined.html')

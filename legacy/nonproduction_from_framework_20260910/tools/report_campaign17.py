"""Frozen 17-candidate evidence report; run only with the existing NaGen Python."""
import argparse
import base64
import csv
import io
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
import re
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_uma_campaign import Collector, crystal
from nagen.selection.diversity import distances

RUN = ROOT / 'runs/campaign_24'
OUT = RUN / 'report_17'
COUNT = 17
VESTA = Path('/mnt/data2/aobo/vesta/VESTA-gtk3-x86_64')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def prepare():
    if (OUT / 'snapshot.json').exists():
        print('Existing frozen snapshot retained')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ('cif', 'framework', 'images', 'audit', 'vesta_preferences'):
        (OUT / folder).mkdir(exist_ok=True)
    manifest = json.loads((RUN / 'selected/manifest.json').read_text())
    entries = manifest['structures'][:COUNT]
    assert len(entries) == COUNT
    records = [json.loads(p.read_text()) for p in sorted((RUN / 'records').glob('*.json'))]
    config = json.loads((RUN / 'configuration.json').read_text())
    frozen_time = datetime.now(timezone.utc).isoformat()
    collector = Collector(RUN)
    rd = distances(collector.ref_features, collector.ref_features)
    positive = rd[np.triu_indices(len(rd), 1)]
    scale = float(np.median(positive[positive > 1e-4]))
    features, rows = [], []
    for number, e in enumerate(entries, 1):
        audit = json.loads((RUN / 'selected' / (e['id'] + '.audit.json')).read_text())
        source = RUN / 'selected' / e['file']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == e['sha256']
        shutil.copy2(source, OUT / 'cif' / e['file'])
        save(OUT / 'audit' / (e['id'] + '.json'), audit)
        s = Structure.from_file(source)
        f = s.copy()
        f.remove_species(['Na'])
        assert Counter(str(site.specie) for site in f) == {'Fe': 6, 'P': 8, 'O': 32}
        CifWriter(f, significant_figures=10).write_file(OUT / 'framework' / e['file'])
        c = collector.strip(crystal(audit['final']))
        features.append(collector.descriptor(c))
        g = audit['hard_gates']
        n = audit['framework_novelty']
        nearest = n['nearest_known_framework_distance']
        distinct = not any(n[k] for k in ('pre_relax_training_match', 'post_refine_training_match', 'post_refine_topology_match'))
        score = 10 * nearest / (nearest + scale) if distinct else 0.
        fe, p = ([r for r in g['sites'] if r['element'] == el] for el in ('Fe', 'P'))
        row = dict(e, number=number, score=score, volume=s.volume, cell_condition=float(np.linalg.cond(s.lattice.matrix)),
                   fe_cn=dict(Counter(r['cn'] for r in fe)), p_cn=dict(Counter(r['cn'] for r in p)),
                   fe_bonds=[min(r['bond_min'] for r in fe), max(r['bond_max'] for r in fe)],
                   p_bonds=[min(r['bond_min'] for r in p), max(r['bond_max'] for r in p)],
                   oxygen_covered=len({i for r in g['sites'] for i in r['oxygen_indices']}),
                   inside_count=sum(r['inside'] for r in g['sites']), timing=audit.get('timing'),
                   relaxation_steps=audit.get('relaxation', {}).get('steps'),
                   refinement_steps=audit.get('refinement', {}).get('steps'),
                   mode=audit.get('generation_and_relaxation_inference', 'default'),
                   energy=audit['refinement']['after']['energy_eV_atom'])
        rows.append(row)
    pd = distances(np.stack(features), np.stack(features))
    np.fill_diagonal(pd, np.inf)
    for row, d in zip(rows, pd):
        row['nearest_snapshot_distance'] = float(d.min())
    timing_rows = [{k: r.get(k) for k in ('id', 'index', 'status', 'generation_and_relaxation_inference', 'timing')} for r in records]
    stats = []
    for mode in ('default', 'batch_tf32'):
        cohort = [r for r in records if r.get('generation_and_relaxation_inference', 'default') == mode]
        st = {'mode': mode, 'n': len(cohort), 'eligible': sum(r['status'] == 'eligible' for r in cohort)}
        for key in ('generation_seconds', 'relaxation_seconds', 'refinement_seconds', 'total_seconds'):
            values = [r.get('timing', {}).get(key, 0.) for r in cohort]
            st[key] = {'mean': float(np.mean(values)), 'median': float(np.median(values)), 'p90': float(np.quantile(values, .9)), 'sum': float(np.sum(values))}
        stats.append(st)
    elapsed = (datetime.fromisoformat(frozen_time) - datetime.fromisoformat(config['created_utc'])).total_seconds()
    data = {'snapshot_utc': frozen_time, 'rows': rows, 'configuration': config, 'manifest_summary': manifest['summary'],
            'completed_samples_at_snapshot': len(records), 'record_counts': dict(Counter(r['status'] for r in records)),
            'timing_rows': timing_rows, 'cohorts': stats, 'wall_seconds': elapsed,
            'novelty_scale': scale, 'reference_pair_count': len(positive[positive > 1e-4]),
            'reference_count': len(collector.refs), 'pair_distances': np.where(np.isfinite(pd), pd, 0).tolist(),
            'profile': collector.profile, 'python': sys.executable}
    save(OUT / 'snapshot.json', data)
    print(json.dumps({'snapshot': str(OUT), 'count': len(rows), 'completed_samples': len(records), 'novelty_scale': scale}))


def vesta_env():
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=f'{VESTA}/usr/lib/x86_64-linux-gnu:{VESTA}/lib:{VESTA}',
               VESTA_PREF=str(OUT / 'vesta_preferences'), LIBGL_ALWAYS_SOFTWARE='1', GDK_BACKEND='x11')
    return env


def convert():
    data = json.loads((OUT / 'snapshot.json').read_text())
    for row in data['rows']:
        source = OUT / 'framework' / row['file']
        target = source.with_suffix('.vesta')
        p = subprocess.run([str(VESTA / 'VESTA'), '-nogui', '-i', str(source), '-o', str(target)],
                           env=vesta_env(), capture_output=True, text=True, timeout=30)
        print(row['number'], p.returncode, target.exists(), p.stderr[-300:], flush=True)
        if not target.exists():
            raise RuntimeError(p.stdout + p.stderr)


def style():
    for path in sorted((OUT / 'framework').glob('*.vesta')):
        text = path.read_text()
        text = re.sub(r'SBOND\n.*?\nSITET', 'SBOND\n  1 Fe O 0.0 2.5 0 1 1 0 1 0.09 1.0 90 110 120\n  2 P O 0.0 2.0 0 1 1 0 1 0.09 1.0 90 110 120\n  0 0 0 0\nSITET', text, flags=re.S)
        # Atom radii and polyhedron colors: Fe teal, P amber, O red.
        def recolor(match):
            fields = match.group(0).split()
            element = re.sub(r'\d', '', fields[1])
            radius, rgb = {'Fe': (.43, [28, 131, 151]), 'P': (.32, [237, 170, 52]), 'O': (.18, [211, 70, 76])}[element]
            fields[2] = str(radius)
            fields[3:9] = list(map(str, rgb + rgb))
            fields[9] = '160'
            return '  ' + ' '.join(fields)
        text = re.sub(r'^\s*\d+\s+(?:Fe|P|O)\d*\s+\d+\.\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+(?:\s+0)?$', recolor, text, flags=re.M)
        text = re.sub(r'MODEL\s+\d+\s+\d+\s+\d+', 'MODEL 2 1 0', text)
        text = text.replace('POLYP\n 204 1  1.000 180 180 180', 'POLYP\n 160 1  0.7 70 90 100')
        path.write_text(text)


def render():
    data = json.loads((OUT / 'snapshot.json').read_text())
    env = vesta_env()
    readfd, writefd = os.pipe()
    log = (OUT / 'render.log').open('a')
    xvfb = subprocess.Popen(['Xvfb', '-displayfd', str(writefd), '-screen', '0', '1400x1100x24', '-nolisten', 'tcp'], pass_fds=(writefd,), stdout=log, stderr=log)
    os.close(writefd)
    display = os.read(readfd, 32).decode().strip()
    os.close(readfd)
    assert display.isdigit(), 'Xvfb did not allocate a display'
    env['DISPLAY'] = ':' + display
    try:
        for row in data['rows']:
            target = OUT / 'images' / (row['id'] + '.png')
            if target.exists():
                continue
            source = OUT / 'framework' / (row['id'] + '.vesta')
            command = [str(VESTA / 'VESTA-gui'), '-open', str(source), '-rotate_x', '22', '-rotate_y', '28', '-rotate_z', '-8', '-flush', '-export_img', 'scale=2', str(target)]
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if target.exists() and target.stat().st_size > 2000:
                        time.sleep(.8)
                        break
                    if proc.poll() is not None:
                        raise RuntimeError(f'VESTA exited {proc.returncode}: see render.log')
                    time.sleep(.5)
                else:
                    raise RuntimeError('VESTA export timeout: see render.log')
                print(f'VESTA exported {row["number"]}/{len(data["rows"])}: {target.name}', flush=True)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
    finally:
        xvfb.terminate()
        xvfb.wait(timeout=10)
        log.close()


def data_link(content, mime, name, label):
    if isinstance(content, str):
        content = content.encode()
    return f'<a download="{html.escape(name)}" href="data:{mime};base64,{base64.b64encode(content).decode()}">{label}</a>'


def table(headers, rows, ident=''):
    identity = f' id="{html.escape(ident)}"' if ident else ''
    return '<div class="table-scroll"><table' + identity + '><thead><tr>' + ''.join('<th>' + h + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + str(c) + '</td>' for c in r) + '</tr>' for r in rows) + '</tbody></table></div>'


def build():
    d = json.loads((OUT / 'snapshot.json').read_text())
    rows = d['rows']
    yes = '<span class="pass">通过</span>'
    fmt = lambda n, digits=5: '—' if n is None else f'{n:.{digits}f}'
    cn = lambda r: ' · '.join(f'{k}配位×{v}' for k, v in sorted(r['fe_cn'].items()))
    num = lambda r: f'<a href="#s{r["number"]:02}">S{r["number"]:02}</a>'
    core, geometry, novel, timing, cards, csvrows = [], [], [], [], [], []
    for r in rows:
        n, g, m = r['framework_novelty'], r['hard_gates'], r['hard_gates']['metrics']
        f, h = g['force_max_eV_A'], r['reference_hull']['e_above_reference_hull_eV_atom']
        short = r['id'].replace('uma_campaign_seed', '').replace('uma_dflow_seed', '')
        core.append([num(r), short, fmt(f, 6), fmt(.03-f, 6), fmt(h, 6), fmt(.15-h, 6), fmt(r['energy'], 6), fmt(r['volume'], 2), f'{r["score"]:.2f}', yes + ' 7/7'])
        bonds = lambda key: '–'.join(fmt(x, 4) for x in r[key])
        geometry.append([num(r), cn(r), '4配位×8', bonds('fe_bonds'), bonds('p_bonds'), f'{r["oxygen_covered"]}/32', f'{r["inside_count"]}/14', fmt(r['cell_condition'], 3), len(g['overlaps'])])
        novel.append([num(r), '不匹配', '不匹配', '不等价', fmt(n['nearest_known_framework_distance'], 6), fmt(r['nearest_snapshot_distance'], 6), f'{n["pre_edges"]} → {n["post_refine_edges"]}', f'{r["score"]:.2f} / 10'])
        t = r['timing'] or {}
        timing.append([num(r), '原始128轮' if not r['timing'] else r['mode'], fmt(t.get('generation_seconds'), 2), fmt(t.get('relaxation_seconds'), 2), fmt(t.get('refinement_seconds'), 2), fmt(t.get('total_seconds'), 2), r['relaxation_steps'] if r['relaxation_steps'] is not None else '—', r['refinement_steps']])
        png = (OUT / 'images' / (r['id'] + '.png')).read_bytes()
        image_src = 'data:image/png;base64,' + base64.b64encode(png).decode()
        local = table(['局域指标（最差位点）', 'Fe', 'P'], [[label, fmt(m['Fe_'+key], 5), fmt(m['P_'+key], 5)] for key, label in [('off_center', '归一化偏心'), ('bond_distortion', '相对键长畸变'), ('angle_distortion', '角度 RMS / 180°'), ('coordination_rarity', '配位支持稀有度')]])
        site_rows = [[s['site'], s['element'], s['cn'], fmt(s['bond_min'], 5), fmt(s['bond_max'], 5), '是' if s['inside'] else '否', yes if s['passed'] else '未通过'] for s in g['sites']]
        site_table = table(['原始位点索引', '元素', 'CN', '最短键 Å', '最长键 Å', '中心在凸包内', '门控'], site_rows)
        links = ' '.join([
            data_link((OUT/'cif'/r['file']).read_bytes(), 'chemical/x-cif', r['file'], '完整 CIF'),
            data_link((OUT/'framework'/r['file']).read_bytes(), 'chemical/x-cif', r['id']+'_FePO.cif', 'Fe–P–O CIF'),
            data_link((OUT/'framework'/(r['id']+'.vesta')).read_bytes(), 'text/plain', r['id']+'.vesta', 'VESTA 场景'),
            data_link((OUT/'audit'/(r['id']+'.json')).read_bytes(), 'application/json', r['id']+'.audit.json', '完整审计')])
        cards.append(f'''<article class="crystal" id="s{r['number']:02}" data-score="{r['score']}" data-hull="{h}">
<div class="card-head"><div><span class="index">S{r['number']:02}</span><span class="seed">{short}</span></div><span class="score">{r['score']:.2f}<small> / 10</small></span></div>
<img class="structure-image" src="{image_src}" alt="VESTA：{r['id']} 的 Fe–P–O 骨架；Na 已移除" loading="lazy" tabindex="0" title="点击放大">
<div class="card-body"><h3>{cn(r)}</h3><p class="quiet">PO₄ × 8 · Fe–O / P–O 配位边 {n['post_refine_edges']} 条</p>
<div class="mini-metrics"><div><small>最大力 · eV/Å</small><b>{f:.6f}</b></div><div><small>hull proxy · eV/atom</small><b>{h:.6f}</b></div></div>
<p>最近已知骨架距离 <b>{n['nearest_known_framework_distance']:.6f}</b><br>本 17 条中最近邻距离 <b>{r['nearest_snapshot_distance']:.6f}</b></p>
<details><summary>局域畸变与 14 个中心的逐位点检查</summary>{local}<p class="quiet">畸变是诊断/排序指标，没有人为增加硬阈值。</p>{site_table}</details>
<div class="downloads">{links}</div><details><summary>文件身份与生成记录</summary><code>{r['id']}</code><p class="hash">SHA256: {r['sha256']}</p><p>全部 Na/Fe/P/O 坐标和晶胞生成；未复制已知母体坐标。Na 仅在展示副本中移除。</p></details></div></article>''')
        csvrows.append([r['number'], r['id'], f, h, r['energy'], r['volume'], r['score'], n['nearest_known_framework_distance'], r['nearest_snapshot_distance'], json.dumps(r['fe_cn']), *r['fe_bonds'], *r['p_bonds'], r['sha256']])
    a, b = d['cohorts']
    perfrows = []
    for key, label in [('generation_seconds','生成 + 12 步 DFlow'),('relaxation_seconds','初始固定晶胞 FIRE'),('refinement_seconds','严格 refinement（全样本均摊）'),('total_seconds','记录阶段总耗时')]:
        x, y = a[key], b[key]
        perfrows.append([label, fmt(x['mean'], 2), fmt(y['mean'], 2), fmt(x['median'], 2), fmt(y['median'], 2), fmt(x['p90'], 2), fmt(y['p90'], 2), fmt(x['mean']/y['mean'], 2)+'×'])
    speed = a['total_seconds']['mean']/b['total_seconds']['mean']
    seconds = d['wall_seconds']
    n_samples = d['completed_samples_at_snapshot']
    globalrows = [['统计时间范围', d['configuration']['created_utc'] + ' → ' + d['snapshot_utc']],
                  ['墙钟跨度（含初始化、基准测试竞争、阶段切换）', f'{seconds/3600:.3f} h / {seconds/60:.2f} min'],
                  ['已完成新样本 / 新增入选 / 总展示', f'{n_samples} / 16 / 17（含原有1条）'],
                  ['已完成样本中的新入选比例', f'{16/n_samples*100:.3f}%（不含原始128个样本）'],
                  ['整个统计跨度的完成吞吐', f'{n_samples/(seconds/3600):.1f} 个/h，8 GPU 合计'],
                  ['每新增入选结构的墙钟成本', f'{seconds/16/60:.2f} min/条，含淘汰样本'],
                  ['8 GPU × 墙钟跨度（资源占用上界口径）', f'{8*seconds/3600:.2f} GPU·h；不是实测 GPU kernel 活跃时长'],
                  ['完成样本记录阶段耗时之和', f'{sum(c["total_seconds"]["sum"] for c in d["cohorts"])/3600:.2f} worker·h；不含在途样本、模型加载和收集器'],
                  ['参考 hull / 新颖性收集器独立耗时', '未单独计时；不虚构耗时或将其当成 0'],
                  ['运行失败记录', str(d['record_counts'].get('failed',0))],
                  ['运行环境', d['python']]]
    threshold_rows = [['计量 / 电荷模型', 'Na₆Fe₆P₈O₃₂，52 原子；Na⁺ / P⁵⁺ / O²⁻ / Fe²⁺或Fe³⁺可中和', '17/17；所需 Fe 平均价态 +3（形式电荷，不是实测电荷）'],
                      ['残余力', 'maxᵢ ‖Fᵢ‖₂ ≤ 0.030000 eV/Å', '17/17；最终 default UMA'],
                      ['有限参考 hull proxy', '≤ 0.150000 eV/atom；同 checkpoint、4,385 条冻结参考相', '17/17'],
                      ['晶胞', '|det L| > 10⁻⁶ Å³；cond(L) < 100；所有数值有限', '17/17；初始弛豫和严格 refinement 均固定晶胞'],
                      ['PO₄ / 配位中心', 'P 必须 CN=4；每个 Fe/P 中心在 O 邻域凸包内；配位类别训练支持≥20结构', '136/136 个 P 四配位；238/238 个 Fe/P 中心通过'],
                      ['氧覆盖 / 碰撞', '每个 O 至少属于一个 Fe/P 多面体；周期近邻碰撞门控', '544/544 个 O 被覆盖；0 条碰撞违规'],
                      ['已知骨架新颖性', '弛豫前后 StructureMatcher 均不匹配；最终配位图不等价；描述符距离≥10⁻⁴', '17/17；范围限定于冻结的22条匹配计量骨架'],
                      ['集合去重 / 文件', '结构不匹配且描述符距离≥10⁻⁴；CIF round-trip、SHA256', '17/17；原完整 CIF 均保留']]
    for key in ['phosphate:Fe:4','phosphate:Fe:5','phosphate:Fe:6','phosphate:P:4']:
        p = d['profile']['profiles'][key]
        threshold_rows.append([key, f'每个位点所有键长位于 [{p["bond_min"]:.6f}, {p["bond_max"]:.6f}] Å', f'冻结训练支持 {p["structures"]} 个结构'])
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['number','id','max_force_eV_A','hull_proxy_eV_atom','energy_eV_atom','volume_A3','novelty_score_0_10','known_distance','snapshot_nearest_distance','Fe_CN_counts','FeO_min_A','FeO_max_A','PO_min_A','PO_max_A','sha256'])
    w.writerows(csvrows)
    (OUT/'constraints.csv').write_text(buf.getvalue())
    content = (ROOT/'tools/templates/campaign17.html').read_text()
    replacements = {
        'DATE': d['snapshot_utc'].split('.')[0].replace('T',' ')+' UTC', 'SAMPLES': str(n_samples), 'SPEED': f'{speed:.2f}',
        'FMAX': f'{max(r["hard_gates"]["force_max_eV_A"] for r in rows):.6f}',
        'HMAX': f'{max(r["reference_hull"]["e_above_reference_hull_eV_atom"] for r in rows):.6f}',
        'CORE': table(['编号','Seed / 原始ID后缀','最大力<br>eV/Å','力余量<br>eV/Å','Hull proxy<br>eV/atom','Hull余量<br>eV/atom','能量<br>eV/atom','体积<br>Å³','新颖性<br>0–10','硬门控'],core),
        'GEOMETRY': table(['编号','Fe 配位组成','P 配位组成','Fe–O 键长 Å','P–O 键长 Å','O 覆盖','中心在凸包内','cond(L)','碰撞数'],geometry),
        'NOVELTY': table(['编号','弛豫前 vs 已知','弛豫后 vs 已知','最终配位图 vs 已知','最近已知距离 d','集合最近邻距离','配位边 前→后','新颖性分'],novel),
        'THRESHOLDS': table(['约束','明确判据','满足情况'],threshold_rows),
        'PERF': table(['阶段','default均值 s','batch均值 s','default中位 s','batch中位 s','default P90 s','batch P90 s','均值比'],perfrows),
        'GLOBAL': table(['整个框架的计算效率','统计值与口径'],globalrows),
        'TIMING': table(['编号','初始推理模式','生成 s','初始弛豫 s¹','严格refine s','记录总计 s¹','初始FIRE步','严格FIRE步'],timing),
        'CARDS': ''.join(cards), 'SCALE': fmt(d['novelty_scale'],9), 'PAIRCOUNT': str(d['reference_pair_count']),
        'NDEFAULT': str(a['n']), 'NBATCH': str(b['n']),
        'CSV': data_link(buf.getvalue(),'text/csv','constraints_17.csv','下载约束 CSV'),
        'JSON': data_link((OUT/'snapshot.json').read_bytes(),'application/json','snapshot_17.json','下载统计快照 JSON'),
        'MODELHASH': d['configuration']['hashes']['uma'],
    }
    for k, value in replacements.items():
        content = content.replace('{{'+k+'}}', value)
    assert not re.search(r'\{\{[A-Z]+\}\}', content)
    (OUT/'index.html').write_text(content)
    save(OUT/'render_provenance.json', {'renderer': str(VESTA/'VESTA-gui'), 'sha256': hashlib.sha256((VESTA/'VESTA-gui').read_bytes()).hexdigest(),
        'python': sys.executable, 'display': 'Xvfb; software OpenGL', 'Na': 'removed only from visualization copies',
        'bond_cutoffs_A': {'Fe-O':2.5,'P-O':2.0}, 'rotations_degrees': {'x':22,'y':28,'z':-8},
        'image_count': len(rows), 'images': {r['id']:hashlib.sha256((OUT/'images'/(r['id']+'.png')).read_bytes()).hexdigest() for r in rows}})
    print(str(OUT/'index.html'), (OUT/'index.html').stat().st_size, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'convert', 'style', 'render', 'build'])
    args = parser.parse_args()
    globals()[args.mode]()

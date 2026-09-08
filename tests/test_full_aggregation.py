"""Integration check using the real small-batch artifacts when available."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_full_aggregation_of_verified_small_batch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    smoke = root/'outputs/relax_audit_2333337'
    if not smoke.exists():
        pytest.skip('real-potential local artifacts not distributed with source')
    pairs = [json.loads(l) for l in (smoke/'pairs.jsonl').read_text().splitlines()]
    pairs = [r for r in pairs if all(v for k,v in r['gates']['before']['checks'].items() if k != 'final_force')]
    ids = {r['id'] for r in pairs}
    structures = [json.loads(l) for l in (smoke/'relaxed.jsonl').read_text().splitlines()]
    structures = [r for r in structures if r['material_id'] in ids]
    labels = [json.loads(l) for l in (root/'outputs/chgnet_all14_labels.jsonl').read_text().splitlines()]
    labels = [r for r in labels if r['id'] in ids]
    data, audit, shard = tmp_path/'data', tmp_path/'audit', tmp_path/'run/shard_00'
    for directory in [data, audit, shard/'relax']:
        directory.mkdir(parents=True)
    def write_rows(path, rows):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    metadata = [{**r, 'split': 'train'} for r in structures]
    write_rows(data/'target.jsonl', metadata)
    write_rows(audit/'shard_00.jsonl', metadata)
    write_rows(audit/'relax_shard_00.jsonl', metadata)
    write_rows(audit/'geometry.jsonl', [r['gates']['before'] for r in pairs])
    (audit/'profile.json').write_text('{}')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    model = json.loads((smoke/'summary.json').read_text())['model']
    summary = {'failed': 0, 'model': model, 'input_sha256': digest(audit/'shard_00.jsonl'),
               'profile_sha256': digest(audit/'profile.json')}
    (shard/'labels.manifest.json').write_text(json.dumps(summary))
    (shard/'relax/summary.json').write_text(json.dumps(summary))
    write_rows(shard/'labels.jsonl', labels)
    write_rows(shard/'relax/pairs.jsonl', pairs)
    write_rows(shard/'relax/relaxed.jsonl', structures)
    matching = tmp_path/'matching.json'
    matching.write_text(json.dumps({'status': 'completed', 'excluded_heldout_ids': []}))
    out = tmp_path/'aggregate'
    subprocess.run([sys.executable, str(root/'tools/summarize_full_chgnet.py'),
        '--run', str(tmp_path/'run'), '--data', str(data), '--audit', str(audit),
        '--matching', str(matching), '--out', str(out), '--shards', '1'], check=True, capture_output=True)
    result = json.loads((out/'summary.json').read_text())
    assert result['status'] == 'completed'
    assert result['scored'] == result['hard_pass_after'] == 3
    assert result['hard_pass_before'] == 0

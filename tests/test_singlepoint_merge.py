import hashlib
import json
from pathlib import Path
import subprocess
import sys


def test_singlepoint_merge_checks_all_shards_and_ignores_relaxation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    audit, first, remaining = [tmp_path/p for p in ['audit', 'first', 'remaining']]
    audit.mkdir()
    for i in range(8):
        source = audit/f'shard_{i:02d}.jsonl'
        source.write_text(json.dumps({'material_id': str(i)})+'\n')
        directory = (first if i < 4 else remaining)/f'shard_{i:02d}'
        directory.mkdir(parents=True)
        row = {'id': str(i), 'status': 'scored', 'model_hash': 'same-model', 'force_max_eV_A': .04}
        (directory/'labels.jsonl').write_text(json.dumps(row)+'\n')
        manifest = {'failed': 0, 'scored': 1,
                    'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'model': {'state_sha256': 'same-model'}}
        (directory/'labels.manifest.json').write_text(json.dumps(manifest))
        (directory/'relax').mkdir()
        (directory/'relax/broken.txt').write_text('Interrupted historical relaxation; must not read')
    command = [sys.executable, str(root/'tools/merge_singlepoint.py'),
               '--first-run', str(first), '--remaining-run', str(remaining),
               '--audit', str(audit), '--out', str(tmp_path/'merged')]
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads((tmp_path/'merged/manifest.json').read_text())
    assert report['count'] == 8
    assert report['force_threshold_counts']['0.05'] == 8
    assert report['force_threshold_counts']['0.03'] == 0
    (audit/'shard_07.jsonl').write_text('{"material_id":"changed"}\n')
    command[-1] = str(tmp_path/'should_not_exist')
    result = subprocess.run(command, capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path/'should_not_exist').exists()

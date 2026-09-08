import importlib.util
from pathlib import Path
import subprocess
import sys
import json


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/prepare_chgnet_dataset.py'
spec = importlib.util.spec_from_file_location('prepare_dataset', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_related_na_variants_share_parent():
    a = 'NaSICON_Na4Fe2P3O12_10_Fe_Na6_Na6Fe4P6O24_0_POSCAR'
    b = 'NaSICON_Na4Fe2P3O12_10_Fe_Na5_Na5Fe4P6O24_6_POSCAR'
    assert module.parent_group(a) == module.parent_group(b)
    assert module.parent_group('unrecognized') is None


def test_partial_input_refused(tmp_path):
    source = tmp_path/'partial.jsonl'
    source.write_text('{}\n')
    output = tmp_path/'output'
    result = subprocess.run([sys.executable, str(SCRIPT), '--input', str(source),
                             '--out', str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'refusing a partial-data benchmark' in result.stderr
    assert not output.exists()


def test_balanced_split_preserves_parent_groups_and_drops_energy(tmp_path):
    source = tmp_path/'raw.jsonl'
    records = []
    for parent, count in [('first', 10), ('second', 5), ('third', 3), ('fourth', 2)]:
        for i in range(count):
            records.append({'material_id': f'{parent}_Fe_Na1_NaFePO4_{i}',
                            'optimized_energy_per_atom': -999,
                            'structure': {'sites': [{'species': [{'element': e, 'occu': 1}]} for e in ['Na', 'Fe', 'P', 'O']]}})
    source.write_text(''.join(json.dumps(r)+'\n' for r in records))
    output = tmp_path/'output'
    subprocess.run([sys.executable, str(SCRIPT), '--input', str(source), '--out', str(output),
                    '--expected-bytes', str(source.stat().st_size)], check=True, capture_output=True)
    assigned = {}
    for split in ['train', 'val', 'test']:
        for row in map(json.loads, (output/f'{split}.jsonl').read_text().splitlines()):
            assert 'optimized_energy_per_atom' not in row
            assert assigned.setdefault(row['parent_group'], split) == split
    assert set(assigned.values()) == {'train', 'val', 'test'}
    assert len((output/'target.jsonl').read_text().splitlines()) == 20

import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import pytest
from pymatgen.core import Structure

OUT = Path(__file__).resolve().parents[1]/'runs/campaign_24/report_19'
pytestmark=pytest.mark.skipif(not (OUT/'index.html').exists(),reason='19-structure report not built')


def test_exactly_two_tables_and_19_images():
    class Parser(HTMLParser):
        tables=0
        images=[]
        paragraphs=0
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='table':self.tables+=1
            if tag=='p':self.paragraphs+=1
            if tag=='img' and a.get('src'):self.images.append(a['src'])
    p=Parser()
    p.feed((OUT/'index.html').read_text())
    assert p.tables==2 and p.paragraphs==0 and len(p.images)==19
    assert len({hashlib.sha256(base64.b64decode(s.split(',',1)[1])).hexdigest() for s in p.images})==19


def test_19_actual_pipeline_feasible_and_visualization_has_no_na():
    d=json.loads((OUT/'snapshot.json').read_text())
    assert len(d['rows'])==19
    for r in d['rows']:
        assert all(r['hard_gates']['checks'].values())
        assert r['hard_gates']['force_max_eV_A']<=.03
        assert r['reference_hull']['e_above_reference_hull_eV_atom']<=.15
        assert r['nearest_snapshot_distance']>=1e-4
        assert r['cif_roundtrip']
        assert hashlib.sha256((OUT/'cif'/r['file']).read_bytes()).hexdigest()==r['sha256']
        f=Structure.from_file(OUT/'framework'/r['file'])
        assert len(f)==46 and 'Na' not in {str(s.specie) for s in f}


def test_efficiency_denominators_match_snapshot():
    d=json.loads((OUT/'snapshot.json').read_text())
    assert len(d['timing_rows'])==sum(c['n'] for c in d['cohorts'])==d['completed_samples_at_snapshot']
    assert sum(r['timing'] is not None for r in d['rows'])==18

"""Checks on the frozen, self-contained report deliverable (no GPU needed)."""
import base64
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pymatgen.core import Structure
import pytest

OUT = Path(__file__).resolve().parents[1] / 'runs/campaign_24/report_17'
pytestmark = pytest.mark.skipif(not (OUT/'index.html').exists(), reason='17-structure report not built')


class Page(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.ids, self.images, self.links, self.cards = [], [], [], []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'img' and a.get('src'):
            self.images.append(a)
        if tag == 'a':
            self.links.append(a)
        if tag == 'article':
            self.cards.append(a)


def test_page_complete_and_offline():
    content = (OUT/'index.html').read_text()
    page = Page(content)
    assert len(page.cards) == len(page.images) == 17
    assert len(set(page.ids)) == len(page.ids)
    assert '{{' not in content
    for a in page.links:
        if a['href'].startswith('#'):
            assert a['href'][1:] in page.ids
        if 'download' in a:
            assert a['href'].startswith('data:')
            assert base64.b64decode(a['href'].split(',', 1)[1], validate=True)
    assert content.count('download=') == 70


def test_vesta_images_not_blank_and_unique():
    page = Page((OUT/'index.html').read_text())
    hashes = []
    for img in page.images:
        data = base64.b64decode(img['src'].split(',', 1)[1])
        picture = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
        assert picture.shape[0] >= 700 and picture.shape[1] >= 900
        assert np.mean(np.min(picture, axis=2) < 220) > .07
        hashes.append(hashlib.sha256(data).hexdigest())
    assert len(set(hashes)) == 17


def test_constraints_and_original_cif_integrity():
    d = json.loads((OUT/'snapshot.json').read_text())
    assert len(d['rows']) == 17
    for r in d['rows']:
        original = OUT/'cif'/r['file']
        assert hashlib.sha256(original.read_bytes()).hexdigest() == r['sha256']
        assert len(Structure.from_file(original)) == 52
        framework = Structure.from_file(OUT/'framework'/r['file'])
        assert len(framework) == 46
        assert 'Na' not in [str(s.specie) for s in framework]
        assert all(r['hard_gates']['checks'].values())
        assert r['hard_gates']['force_max_eV_A'] <= .03
        assert r['reference_hull']['e_above_reference_hull_eV_atom'] <= .15
        assert r['oxygen_covered'] == 32 and r['inside_count'] == 14
        assert r['p_cn'] == {'4': 8}


def test_novelty_score_recomputes_without_cohort_stretch():
    d = json.loads((OUT/'snapshot.json').read_text())
    for r in d['rows']:
        n = r['framework_novelty']
        nearest = n['nearest_known_framework_distance']
        assert r['score'] == pytest.approx(10*nearest/(nearest+d['novelty_scale']))
        assert 0 <= r['score'] <= 10
        assert r['nearest_snapshot_distance'] >= 1e-4
        assert not any(n[k] for k in ('pre_relax_training_match', 'post_refine_training_match', 'post_refine_topology_match'))


def test_timings_include_rejections_and_keep_missing_unknown():
    d = json.loads((OUT/'snapshot.json').read_text())
    assert sum(c['n'] for c in d['cohorts']) == d['completed_samples_at_snapshot'] == len(d['timing_rows'])
    assert sum(r['timing'] is None for r in d['rows']) == 1
    for c in d['cohorts']:
        cohort = [r for r in d['timing_rows'] if (r['generation_and_relaxation_inference'] or 'default') == c['mode']]
        assert len(cohort) == c['n']
        for key in ('generation_seconds','relaxation_seconds','refinement_seconds','total_seconds'):
            assert c[key]['mean'] == pytest.approx(np.mean([r['timing'].get(key,0) for r in cohort]))

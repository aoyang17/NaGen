"""Predeclared train-calibrated / validation-only Fe cutoff sensitivity."""
import argparse
from collections import Counter
import json
from pathlib import Path

from nagen.selection.audit import load_crystals
from nagen.selection.pipeline import calibrate, hard_gates, Conditioning


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    data = Path(args.data)
    train = list(load_crystals(data/'train.jsonl'))
    val = list(load_crystals(data/'val.jsonl'))
    results = {}
    for cutoff in [2.3, 2.5, 2.7]:
        profile = calibrate(train, 20, {'P': 2., 'Fe': cutoff})
        checks, cn = Counter(), Counter()
        with (out/f'validation_{cutoff}.jsonl').open('x') as handle:
            for c in val:
                row = hard_gates(c, Conditioning(dict(Counter(c.elements))), profile, motif_backend='native')
                handle.write(json.dumps(row, allow_nan=False)+'\n')
                checks.update(k for k,v in row['checks'].items() if v)
                cn.update(str(s['cn']) for s in row['sites'] if s['element'] == 'Fe')
        results[str(cutoff)] = {'validation_count': len(val), 'passed_counts': dict(checks),
                                'validation_Fe_CN_sites': dict(cn), 'profiles': profile['profiles']}
        print(json.dumps({'cutoff': cutoff, 'passed_counts': dict(checks)}), flush=True)
    (out/'summary.json').write_text(json.dumps({'results': results,
        'note': 'Diagnostic only; frozen production cutoff remains 2.5 A. Test split not examined. '
                'Validation results include all original validation IDs; matching exclusions are separate.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()

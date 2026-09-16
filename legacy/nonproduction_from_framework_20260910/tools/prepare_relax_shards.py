"""Derive relaxation eligibility from an existing frozen geometry audit."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True)
    args = p.parse_args()
    root = Path(args.audit)
    eligible = set()
    with (root/'geometry.jsonl').open() as handle:
        for row in map(json.loads, handle):
            if all(v for k, v in row['checks'].items() if k != 'final_force'):
                eligible.add(row['id'])
    counts = {}
    for source in sorted(root.glob('shard_*.jsonl')):
        rows = [r for r in map(json.loads, source.read_text().splitlines()) if r['material_id'] in eligible]
        with (root/('relax_'+source.name)).open('x') as handle:
            for r in rows:
                handle.write(json.dumps(r)+'\n')
        counts[source.name] = len(rows)
    with (root/'relax_plan.json').open('x') as handle:
        json.dump({'eligible': len(eligible), 'counts': counts,
                   'policy': 'All target structures receive single-point scores; only structures '
                             'passing all non-force hard gates undergo fixed-cell relaxation.'}, handle, indent=2)
    print(counts)


if __name__ == '__main__':
    main()

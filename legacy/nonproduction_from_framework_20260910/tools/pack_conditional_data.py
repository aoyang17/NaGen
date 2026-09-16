"""Pack original geometry only; exclude known heldout leakage; never read energy labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from pymatgen.core import Structure

from nagen.selection.audit import load_crystals
from nagen.selection.experiment import sha256
from nagen.inverse.spec import DEFAULT_SPEC


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--matching', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    excluded = set(json.loads(Path(args.matching).read_text())['excluded_heldout_ids'])
    mapping = DEFAULT_SPEC.element_to_index
    types, frac, cells, ids, splits, offsets = [], [], [], [], [], [0]
    conditions = Counter()
    for code, split in enumerate(['train', 'val', 'test']):
        for c in load_crystals(Path(args.data)/f'{split}.jsonl'):
            if c.id in excluded:
                continue
            # Basis change only: keep N/A and physical geometry, no relaxation.
            original = Structure(c.lattice, c.elements, c.frac)
            reduced = original.get_reduced_structure(reduction_algo='niggli')
            if len(reduced) != len(original) or reduced.composition != original.composition or not np.isclose(reduced.volume, original.volume, rtol=1e-7):
                raise ValueError('cell reduction changed the physical structure')
            types.extend(mapping[e] for e in c.elements)
            if [s.specie.symbol for s in reduced] != c.elements:
                raise ValueError('unexpected atom ordering change during basis reduction')
            frac.extend(reduced.frac_coords % 1)
            cells.append(reduced.lattice.matrix)
            ids.append(c.id)
            splits.append(code)
            offsets.append(len(types))
            counts = Counter(c.elements)
            required = 2*counts['O']-counts['Na']-5*counts['P']
            if split == 'train' and 2*counts['Fe'] <= required <= 3*counts['Fe'] and len(c.elements) <= 100:
                conditions[tuple(sorted(counts.items()))] += 1
    payload = {'version': 'conditional-original-geometry-niggli-v2', 'element_to_index': mapping,
        'types': torch.tensor(types, dtype=torch.int16), 'frac_coords': torch.tensor(np.array(frac), dtype=torch.float32),
        'lattices': torch.tensor(np.array(cells), dtype=torch.float32),
        'ids': ids, 'split': torch.tensor(splits, dtype=torch.uint8), 'offsets': torch.tensor(offsets),
        'filter_definition': 'Original ordered Na/Fe/P/O geometry, Niggli basis only; heldout StructureMatcher exclusions. No energy labels.',
        'source_hashes': {s: sha256(Path(args.data)/f'{s}.jsonl') for s in ['train', 'val', 'test']},
        'matching_sha256': sha256(args.matching)}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    support = conditions.most_common(3)
    manifest = {'count': len(ids), 'split_counts': dict(Counter(splits)), 'sha256': sha256(out),
                'contains_energy_labels': False, 'pilot_conditions': [
                    {'counts': dict(counts), 'family': 'phosphate', 'training_support': count} for counts,count in support]}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for i, (counts, count) in enumerate(support):
        (out.parent/f'condition_{i}.json').write_text(json.dumps({'counts': dict(counts), 'family': 'phosphate'}, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

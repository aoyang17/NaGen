"""Export only explicitly passed post-relaxation structures with provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    source, out = Path(args.run), Path(args.out)
    labels = {r['id']: r for r in map(json.loads, (source/'pairs.jsonl').read_text().splitlines())
              if r['status'] == 'completed' and r['gates']['after']['passed']}
    structures = [json.loads(line) for line in (source/'relaxed.jsonl').read_text().splitlines()]
    if not labels.keys() <= {r['material_id'] for r in structures}:
        raise ValueError('missing feasible structure geometry')
    out.mkdir(parents=True, exist_ok=False)
    manifest = []
    matcher = StructureMatcher(ltol=1e-6, stol=1e-5, angle_tol=1e-5,
                               primitive_cell=False, scale=False)
    for r in structures:
        identifier = r['material_id']
        if identifier not in labels:
            continue
        label = labels[identifier]
        structure = Structure(r['L_matrix'], r['A'], r['X_frac'])
        name = f'feasible_{len(manifest)+1:03d}_{hashlib.sha256(identifier.encode()).hexdigest()[:10]}.cif'
        CifWriter(structure, symprec=None, significant_figures=10).write_file(out/name)
        reloaded = Structure.from_file(out/name)
        assert reloaded.composition == structure.composition
        assert np.allclose(reloaded.lattice.abc, structure.lattice.abc, atol=1e-7)
        # CIF parsers may reorder sites; verify geometry modulo site permutation.
        assert matcher.fit(reloaded, structure)
        manifest.append({'file': name, 'material_id': identifier,
                         'formula': structure.composition.formula,
                         'energy_eV_atom': label['after']['energy_eV_atom'],
                         'force_max_eV_A': label['after']['force_max_eV_A'],
                         'checks': label['gates']['after']['checks'],
                         'cif_sha256': hashlib.sha256((out/name).read_bytes()).hexdigest()})
    (out/'manifest.json').write_text(json.dumps({'source_run': str(source.resolve()),
        'model': json.loads((source/'summary.json').read_text())['model'],
        'structures': manifest}, indent=2)+'\n')
    (out/'README.txt').write_text('These existing dataset structures passed the current native hard gates after\n'
        'CHGNet fixed-cell relaxation (final max force <=0.05 eV/Angstrom).\n'
        'They are NOT certified thermodynamically stable, newly generated, or novelty-selected.\n'
        'Finite-sample motif calibration and Fe2+/Fe3+ assumptions apply.\n'
        'CIF stores geometry; energies, forces, checks, and model provenance are in manifest.json.\n')
    print(json.dumps({'exported': len(manifest), 'out': str(out)}))


if __name__ == '__main__':
    main()

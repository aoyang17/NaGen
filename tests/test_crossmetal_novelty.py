import importlib.util
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

spec=importlib.util.spec_from_file_location('crossmetal_audit',Path(__file__).resolve().parents[1]/'tools/audit_crossmetal_novelty.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def sample():
    return Structure(Lattice.cubic(12),['Na']*3+['Fe']*3+['P']*4+['O']*16,np.random.default_rng(73).random((26,3)))


def test_substitution_and_supercell_are_not_novel():
    s=sample();r=s.copy();r.replace_species({'Fe':'Cr','Na':'K'})
    r.make_supercell([2,1,1])
    a,reason,_=module.framework(s);b,reason2,metals=module.framework(r)
    assert reason is None and reason2 is None and metals==['Cr']
    assert module.ratio(a)==module.ratio(b)==(('Fe',3),('O',16),('P',4))
    matcher=StructureMatcher(primitive_cell=True,scale=True,attempt_supercell=True)
    assert matcher.fit(a,b)


def test_p_and_o_are_not_anonymized():
    s=sample();changed=s.copy()
    changed.replace(6,'O');changed.replace(10,'P')
    a,_,_=module.framework(s);b,_,_=module.framework(changed)
    assert module.ratio(a)==module.ratio(b)
    assert [str(x.specie) for x in a]!=[str(x.specie) for x in b]
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,attempt_supercell=True)
    assert not matcher.fit(a,b)


def test_mixed_metal_site_supported_but_vacancies_flagged():
    s=sample();s.replace(3,{'Cr':.5,'Mn':.5})
    f,reason,metals=module.framework(s)
    assert reason is None and set(metals)=={'Cr','Mn','Fe'}
    assert len(f)==23
    s.replace(10,{'O':.5})
    assert module.framework(s)[1]=='partial_occupancy'

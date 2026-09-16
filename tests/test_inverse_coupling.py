import torch

from nagen.inverse.coupling import assignment_source_frac


def test_assignment_source_is_valid_and_species_relabel_only():
    torch.manual_seed(7)
    target=torch.rand(2,6,3)
    types=torch.tensor([[1,1,2,2,3,0],[1,2,3,4,4,4]])
    mask=types>0
    lattice=torch.eye(3).repeat(2,1,1)*5
    torch.manual_seed(99)
    expected=torch.rand_like(target)
    torch.manual_seed(99)
    result=assignment_source_frac(target,types,mask,lattice)
    assert torch.all(result[~mask]==0)
    for b in range(2):
        for species in types[b,mask[b]].unique():
            idx=torch.where(mask[b] & (types[b]==species))[0]
            assert torch.allclose(result[b,idx].sort(dim=0).values,
                                  expected[b,idx].sort(dim=0).values)


def test_assignment_rejects_bad_shapes():
    try:
        assignment_source_frac(torch.rand(3,4),torch.ones(3,4),torch.ones(3,4),torch.eye(3))
    except ValueError:
        pass
    else:
        raise AssertionError('bad shape accepted')

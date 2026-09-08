import torch
from nagen.inverse.polyhedral_prior import polyhedral_source_frac
from nagen.inverse.sample import integrate_flow
from nagen.inverse.model import CrystalState


def test_polyhedral_prior_exact_shape_and_po4_shells():
    types=torch.tensor([[1]*6+[2]*6+[3]*8+[4]*32])
    mask=torch.ones_like(types,dtype=torch.bool)
    lattice=torch.eye(3).repeat(1,1,1)*14
    torch.manual_seed(3)
    x=polyhedral_source_frac(types,mask,lattice)
    assert x.shape==(1,52,3) and torch.all((x>=0)&(x<1))
    p=x[0,types[0]==3]; o=x[0,types[0]==4]
    d=p[:,None]-o[None]; d-=torch.floor(d+.5); d=torch.linalg.vector_norm(d@lattice[0],dim=-1)
    nearest=d.topk(4,largest=False).values
    assert torch.allclose(nearest,torch.full_like(nearest,1.55),atol=2e-4)


def test_polyhedral_prior_rejects_non_orthophosphate():
    types=torch.tensor([[1,2,3,4,4,4]])
    try: polyhedral_source_frac(types,types>0,torch.eye(3)[None]*10)
    except ValueError: pass
    else: raise AssertionError('invalid O/P accepted')


def test_zero_horizon_flow_preserves_source():
    class Model:
        def __call__(self,*args):
            state=args[:3]
            return CrystalState(*(torch.ones_like(x) for x in state))
    source=CrystalState(torch.randn(1,2,4),torch.rand(1,2,3),torch.randn(1,6))
    result=integrate_flow(Model(),source,torch.ones(1,2,dtype=torch.bool),steps=3,end_time=0)
    assert all(torch.equal(a,b) for a,b in zip(source,result))


def test_polyhedral_prior_explicit_generator_replays():
    types=torch.tensor([[1]*6+[2]*6+[3]*8+[4]*32]); mask=types>0; lattice=torch.eye(3)[None]*14
    a=polyhedral_source_frac(types,mask,lattice,generator=torch.Generator().manual_seed(19))
    torch.manual_seed(999)
    b=polyhedral_source_frac(types,mask,lattice,generator=torch.Generator().manual_seed(19))
    assert torch.equal(a,b)

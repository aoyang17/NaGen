import torch
from nagen.selection.conditional_flow import assigned_uniform_source, conditional_flow_loss, ATOM_SCALE
from nagen.inverse.model import CrystalState, CrystalVectorField, FlowConfig


def test_assignment_only_permutes_each_species():
    target = torch.rand(2, 7, 3)
    types = torch.tensor([[1,1,2,2,3,4,0], [1,1,2,2,3,4,4]])
    mask = types > 0
    torch.manual_seed(17)
    original = torch.rand_like(target)
    torch.manual_seed(17)
    matched = assigned_uniform_source(target, types, mask)
    for b in range(2):
        for species in [1,2,3,4]:
            indices = types[b] == species
            assert sorted(map(tuple, original[b,indices].tolist())) == sorted(map(tuple, matched[b,indices].tolist()))
    assert torch.equal(matched[~mask], torch.zeros_like(matched[~mask]))


def test_training_uses_fixed_atom_input_and_finite_geometry_gradients():
    model = CrystalVectorField(FlowConfig(hidden_dim=32, layers=1, heads=4), torch.zeros(6), torch.ones(6))
    types = torch.tensor([[1,2,3,4,4,4,4]])
    mask = types > 0
    seen = []
    hook = model.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    loss, metrics = conditional_flow_loss(model, types, torch.rand(1,7,3), torch.zeros(1,6), mask)
    loss.backward()
    expected = ATOM_SCALE*torch.nn.functional.one_hot(types-1, 4)
    assert torch.equal(seen[0], expected)
    assert model.atom_velocity.weight.grad is None
    assert torch.isfinite(model.frac_velocity.weight.grad).all()
    assert set(metrics) == {'frac_mse', 'lattice_mse'}
    hook.remove()

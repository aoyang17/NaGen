import torch
from nagen.selection.occupancy_flow import OccupancyFlow, OccupancyFlowConfig, flow_matching_loss, topk_occupancy


def test_topk_enforces_condition_not_penalty():
    logits=torch.randn(9,16)
    decoded=topk_occupancy(logits,12)
    assert decoded.shape == logits.shape
    assert torch.all(decoded.sum(1)==12)


def test_flow_loss_is_differentiable():
    model=OccupancyFlow(OccupancyFlowConfig(16,hidden=16,depth=2))
    occupancy=torch.zeros(4,16); occupancy[:,:12]=1
    loss=flow_matching_loss(model,occupancy,12)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())

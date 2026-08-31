"""条件 flow matching 生成模型：母体模板畸变场（X_delta, L_delta）。

网络：逐原子共享 MLP 输出坐标速度 + 全局池化头输出晶格速度。
采样：显式欧拉 ODE（torch 自动求图，供 D-Flow 反传）。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    def __init__(self, cond_dim: int, n_hidden: int = 256, n_layers: int = 3):
        super().__init__()
        self.n_hidden = n_hidden
        self.atom_mlp = nn.Sequential(
            nn.Linear(3 + 16 + 1 + cond_dim, n_hidden),
            nn.SiLU(),
            *[
                layer
                for _ in range(n_layers - 1)
                for layer in (nn.Linear(n_hidden, n_hidden), nn.SiLU())
            ],
        )
        self.vx = nn.Linear(n_hidden, 3)
        self.global_head = nn.Sequential(
            nn.Linear(n_hidden + 1 + cond_dim, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 9),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, z: torch.Tensor, mask: torch.Tensor):
        """x: (B,N,3) 当前畸变坐标；t:(B,1)；cond:(B,C)；z:(B,N)；mask:(B,N)。"""
        B, N, _ = x.shape
        z_emb = torch.nn.functional.one_hot(z.clamp(min=0, max=15), num_classes=16).float()  # (B,N,16)
        tb = t.unsqueeze(-1).expand(B, N, 1)
        cb = cond.unsqueeze(1).expand(B, N, -1)
        feat = torch.cat([x, z_emb, tb, cb], dim=-1)  # (B,N,3+16+1+C)
        h = self.atom_mlp(feat)  # (B,N,H)
        vx = self.vx(h) * mask.unsqueeze(-1)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        vL = self.global_head(torch.cat([pooled, t, cond], dim=-1))  # (B,9)
        return vx, vL


def flow_matching_loss(model, x1, L1, z, mask, cond, eps=1e-4):
    B = x1.shape[0]
    t = torch.rand(B, 1, device=x1.device) * (1 - 2 * eps) + eps
    x0 = torch.randn_like(x1)
    L0 = torch.randn_like(L1)
    xt = (1 - t[:, :, None]) * x0 + t[:, :, None] * x1
    Lt = (1 - t) * L0 + t * L1
    vx, vL = model(xt, t, cond, z, mask)
    loss_x = ((vx - (x1 - x0)) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1)
    loss_L = ((vL - (L1 - L0)) ** 2).mean()
    return loss_x + loss_L, loss_x, loss_L


def euler_sample(model, x0, L0, z, mask, cond, steps: int = 24, wrap: bool = True):
    """显式欧拉 ODE：t: 0->1，返回 (x1, L1)。保留计算图供 D-Flow。"""
    x, L = x0, L0
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0], 1), (i + 0.5) * dt, device=x.device)
        vx, vL = model(x, t, cond, z, mask)
        x = x + vx * dt
        L = L + vL * dt
        if wrap:
            x = torch.remainder(x + 0.5, 1.0) - 0.5
    return x, L


def train_flow(ds: dict, out_path: str, epochs: int = 300, lr: float = 3e-4, seed: int = 0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = VelocityNet(cond_dim=ds["cond_dim"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X1, L1 = ds["X_delta"].to(dev), ds["L_delta"].to(dev)
    Z, mask, cond = ds["Z"].to(dev), ds["mask"].to(dev), ds["cond"].to(dev)
    n = X1.shape[0]
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 32):
            idx = perm[i : i + 32]
            loss, lx, ll = flow_matching_loss(model, X1[idx], L1[idx], Z[idx], mask[idx], cond[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history.append(total / n)
        if (ep + 1) % 50 == 0:
            print(f"epoch {ep + 1}/{epochs} loss={history[-1]:.5f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "cond_dim": ds["cond_dim"], "N": ds["N_max"]}, out_path)
    return model, history


if __name__ == "__main__":
    import argparse
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/generation/flow_dataset.pt")
    ap.add_argument("--out", default="data/generation/models/flow.pt")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    ds = torch.load(args.dataset, weights_only=False)
    model, hist = train_flow(ds, args.out, epochs=args.epochs)
    print(f"训练完成：{args.out}（最后 10 epoch 均值 {sum(hist[-10:]) / 10:.5f}）")

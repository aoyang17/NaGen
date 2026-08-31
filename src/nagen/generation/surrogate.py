"""可微能量代理：平滑 RDF 描述符 + MLP → MLFF 能量（供 D-Flow 源空间优化）。"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

R_CUT = 6.0
N_BINS = 32
SIGMA = 0.08


def pbc_pair_distances(x: torch.Tensor, L: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x:(B,N,3) 分数坐标；L:(B,3,3)；mask:(B,N)。返回 (B,N,N) 最小镜像距离（掩码外=0）。"""
    B, N, _ = x.shape
    dev = x.device
    shifts = torch.tensor(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=torch.float32,
        device=dev,
    )  # (27,3)
    d = x[:, :, None, None, :] - x[:, None, :, None, :] + shifts[None, None, None, :, :]
    d = d.float()
    L = L.float()
    cart = torch.einsum("bnmsk,bkl->bnmsl", d, L)  # (B,N,N,27,3)
    dist = torch.norm(cart, dim=-1).min(dim=-1).values  # (B,N,N)
    mm = mask[:, :, None] * mask[:, None, :]
    eye = torch.eye(N, device=dev, dtype=torch.bool)
    mm = mm * (~eye[None]).float()
    return dist * mm.float()


def rdf_descriptor(x: torch.Tensor, L: torch.Tensor, mask: torch.Tensor, comp: torch.Tensor) -> torch.Tensor:
    """平滑 RDF 直方图 + 组成计数 + 体积 → (B, N_BINS+8+1)。可微 wrt x, L。"""
    B = x.shape[0]
    if L.shape[-1] == 9:
        L = L.reshape(B, 3, 3)
    dist = pbc_pair_distances(x, L, mask)  # (B,N,N)
    d_all = dist.reshape(B, -1)
    centers = torch.linspace(0.3, R_CUT, N_BINS, device=x.device)
    # 高斯涂抹：exp(-(d-c)^2/(2σ^2))，只保留 cutoff 内
    d_exp = d_all[:, :, None]  # (B,P,1)
    g = torch.exp(-((d_exp - centers[None, None, :]) ** 2) / (2 * SIGMA**2))
    g = g * (d_all[:, :, None] < R_CUT).float()
    hist = g.sum(dim=1)  # (B,N_BINS)
    vol = torch.linalg.det(L).abs()  # (B,)
    return torch.cat([hist, comp, vol.unsqueeze(-1) / 1000.0], dim=-1)


class EnergySurrogate(nn.Module):
    def __init__(self, n_bins: int = N_BINS, n_comp: int = 8, n_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bins + n_comp + 1, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1),
        )
        self.register_buffer("e_mean", torch.zeros(1))
        self.register_buffer("e_std", torch.ones(1))

    def forward(self, x, L, mask, comp):
        desc = rdf_descriptor(x, L, mask, comp)
        return self.net(desc)

    def predict_eV_atom(self, x, L, mask, comp, n_atoms):
        return self.forward(x, L, mask, comp).squeeze(-1) * self.e_std + self.e_mean


def train_surrogate(ds: dict, out_path: str, epochs: int = 600, lr: float = 3e-4, seed: int = 1):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    e = ds["energy"]
    ok = ~torch.isnan(e)
    e_mean = e[ok].mean()
    e_std = e[ok].std()
    model = EnergySurrogate().to(dev)
    model.e_mean.copy_(e_mean)
    model.e_std.copy_(e_std)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = ds["X_delta"].to(dev)
    L = ds["L_delta"].to(dev)
    mask = ds["mask"].to(dev)
    z = ds["Z"].to(dev)
    Xt = ds["X_tpl"].to(dev)
    Lt = ds["L_tpl"].to(dev)
    comp_all = ds["cond"][:, -8:]
    y = (e - e_mean) / e_std
    n = X.shape[0]
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 16):
            idx = perm[i : i + 16]
            x_full = torch.remainder(Xt[idx] + X[idx] + 0.5, 1.0) - 0.5 + 0.5
            L_full = (Lt[idx] + L[idx]) * 5.0
            pred = model(x_full, L_full, mask[idx], comp_all[idx].to(dev)).squeeze(-1)
            loss = ((pred - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history.append(total / n)
        if (ep + 1) % 100 == 0:
            print(f"epoch {ep + 1}/{epochs} loss={history[-1]:.5f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "e_mean": e_mean, "e_std": e_std}, out_path)
    return model, history


if __name__ == "__main__":
    import argparse
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/generation/flow_dataset.pt")
    ap.add_argument("--out", default="data/generation/models/surrogate.pt")
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()
    ds = torch.load(args.dataset, weights_only=False)
    model, hist = train_surrogate(ds, args.out, epochs=args.epochs)
    print(f"代理训练完成：{args.out}")

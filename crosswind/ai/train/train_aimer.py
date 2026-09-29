"""Train the aimer on the practice shots: (weapon, angle, where it landed, wind) -> power.

    python -m crosswind.ai.train.train_aimer
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

from crosswind.ai.nets import MODELS_DIR, MLP
from crosswind.ai.obs import aimer_inputs


def features(shots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weapon, angle, power, dx, dy, wind, trend = (shots[:, i] for i in range(7))
    return aimer_inputs(weapon, angle, dx, dy, wind, trend), (power / 100.0).astype(np.float32)[:, None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="runs/practice.npz")
    ap.add_argument("--out", default=str(MODELS_DIR / "aimer.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()
    shots = np.load(args.data)["shots"]
    x, y = features(shots)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    split = int(len(x) * 0.95)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt, yt = torch.tensor(x[idx[:split]], device=dev), torch.tensor(y[idx[:split]], device=dev)
    xv, yv = torch.tensor(x[idx[split:]], device=dev), torch.tensor(y[idx[split:]], device=dev)
    h = args.hidden
    model = nn.Sequential(nn.Linear(x.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                          nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 4096):
            b = perm[i:i + 4096]
            loss = nn.functional.smooth_l1_loss(model(xt[b]), yt[b], beta=0.02)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            err = (model(xv) - yv).abs().mul(100)
        print(f"epoch {epoch + 1}: power error median {err.median():.2f}, p90 {err.quantile(0.9):.2f}", flush=True)
    layers = [m for m in model if isinstance(m, nn.Linear)]
    MLP([m.weight.detach().cpu().numpy().T for m in layers], [m.bias.detach().cpu().numpy() for m in layers],
        "relu").save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

"""1D-CNN RUL regressor (Data-Driven baseline), mirroring the GWO-1DCNN paper.

Architecture patterns from Shen et al. (Machines 2025, 13, 583):
  - 1DCNN over the windowed, PCA-reduced sensor features
  - ReLU activations
  - MSE loss, Adam, mini-batch 20, ~60 epochs
  - L2 regularization & drop layers included (the paper tunes L2 / hidden /
    learn-rate with improved-GWO; we expose those as args so the tuning loop in
    tuner_gwo.py can drive them).

This is deliberately a *data-driven* model: it learns only from the windowed
sensor matrix. It is the baseline against which the PINN (which adds physics
constraints on top) is compared on the same held-out engines.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class CNN1DModel(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, out_dim: int = 1,
                 l2_reg: float = 1e-4, dropout: float = 0.2,
                 kernel_sizes=(5, 3, 3), pool=2):
        super().__init__()
        self.l2_reg = l2_reg
        self.dropout = dropout

        self.conv1 = nn.Conv1d(in_features, hidden, kernel_sizes[0], padding=2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden * 2, kernel_sizes[1], padding=1)
        self.bn2 = nn.BatchNorm1d(hidden * 2)
        self.conv3 = nn.Conv1d(hidden * 2, hidden * 4, kernel_sizes[2], padding=1)
        self.bn3 = nn.BatchNorm1d(hidden * 4)

        self.pool = nn.MaxPool1d(pool)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Global average pooling + small MLP head (keeps the model light for
        # a CPU-only demo, still representative of the paper's depth).
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> permute to (B, F, T) for Conv1d
        x = x.permute(0, 2, 1).contiguous()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x

    def loss(self, pred, target):
        """MSE plus L2 regularization (weight decay), as in the paper."""
        mse = nn.functional.mse_loss(pred, target)
        return mse


# ---------------------------------------------------------------------------
# Training harness (data-driven baseline only)
# ---------------------------------------------------------------------------
def train_cnn(build_loader, model, epochs=60, lr=1e-3, device="cpu",
              log_every=5, **kwargs):
    """Generic training loop. `build_loader` is a factory returning train/val
    DataLoaders (so the same harness serves the PINN training too)."""
    import time
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    train_ld, val_ld = build_loader()
    best_mae = np.inf
    history = []
    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = model.loss(out, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        # eval
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                preds.append(model(xb).cpu().numpy().ravel())
                truths.append(yb.cpu().numpy().ravel())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        mae = float(np.mean(np.abs(preds - truths)))
        rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
        history.append({"ep": ep + 1, "loss": float(np.mean(losses)),
                        "mae": mae, "rmse": rmse})
        if mae < best_mae:
            best_mae = mae
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"[cnn] ep {ep+1:>3}/{epochs}  loss {np.mean(losses):.4f}  "
                  f"val MAE {mae:.3f}  RMSE {rmse:.3f}")
    return model, history


def load_windows(npy_path, meta_path, split="train"):
    """Load the cached windows + metadata for one split. Returns X (N,T,F)
    and y (N,1)."""
    X = np.load(npy_path)
    import pandas as pd
    meta = pd.read_parquet(meta_path)
    mask = meta["split"] == split
    return X[mask.values], meta.loc[mask.values]

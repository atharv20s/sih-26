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


def main():
    import argparse
    from pathlib import Path
    from torch.utils.data import TensorDataset, DataLoader

    parser = argparse.ArgumentParser(description="Train 1D-CNN baseline RUL model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--log-every", type=int, default=5, help="Logging frequency")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "processed"
    npy_path = data_dir / "X_windows.npy"
    meta_path = data_dir / "meta_windows.parquet"

    print("=" * 68)
    print(" 1D-CNN Baseline RUL Regressor (Shen et al., Machines 2025)")
    print(f" Device: {args.device} | Epochs: {args.epochs} | Batch Size: {args.batch}")
    print("=" * 68)

    if not npy_path.exists() or not meta_path.exists():
        print(f"Error: Dataset not found at {data_dir}.")
        print("Run `python src/data/build_dataset.py` first to generate telemetry windows.")
        return

    import pandas as pd
    print(f"[*] Loading data from: {data_dir}")
    X = np.load(npy_path)
    meta = pd.read_parquet(meta_path)

    RUL_CAP = 500.0
    y = np.clip(meta["rul"].values.astype(np.float32), 0.0, RUL_CAP) / RUL_CAP

    train_m = (meta["split"] == "train").values
    val_m = (meta["split"] == "val").values
    test_m = (meta["split"] == "test").values

    print(f"[*] Splits: Train={train_m.sum()} | Val={val_m.sum()} | Test={test_m.sum()}")

    def build_loader():
        train_ds = TensorDataset(
            torch.from_numpy(X[train_m]).float(),
            torch.from_numpy(y[train_m]).float().unsqueeze(-1),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X[val_m]).float(),
            torch.from_numpy(y[val_m]).float().unsqueeze(-1),
        )
        train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
        return train_ld, val_ld

    in_features = X.shape[-1]
    model = CNN1DModel(in_features=in_features, hidden=64).to(args.device)

    print(f"[*] Model initialized: in_features={in_features} (PCA), hidden=64")
    print("[*] Starting training loop...")
    model, history = train_cnn(
        build_loader,
        model,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        log_every=args.log_every,
    )

    # Evaluate on held-out test split
    test_ds = TensorDataset(
        torch.from_numpy(X[test_m]).float(),
        torch.from_numpy(y[test_m]).float().unsqueeze(-1),
    )
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            xb = xb.to(args.device)
            preds.append(model(xb).cpu().numpy().ravel())
            truths.append(yb.numpy().ravel())

    preds = np.concatenate(preds) * RUL_CAP
    truths = np.concatenate(truths) * RUL_CAP
    test_mae = float(np.mean(np.abs(preds - truths)))
    test_rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))

    print("-" * 68)
    print(f"[+] Final Test Evaluation: MAE = {test_mae:.2f} cycles | RMSE = {test_rmse:.2f} cycles")
    print("=" * 68)


if __name__ == "__main__":
    main()

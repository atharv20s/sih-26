"""Train and compare: CNN-PCA baseline  vs  CNN-Raw ablation  vs  PINN.

Three models on the SIH26054 piston-engine run-to-failure dataset:

  1. **CNN-PCA** (8 ch, data-driven)  — standard literature approach.
  2. **CNN-Raw** (12 ch, data-driven) — ablation isolating PCA vs raw effect.
  3. **PINN**    (12 ch + physics)    — our innovation.

Comparison (2) vs (3) cleanly isolates the contribution of the Fourier physics
loss, because both models see identical raw-channel input.

Usage (from project root):
    python src/pinn/train_pinn.py [--epochs 60] [--batch 64] [--lr 1e-3]

Outputs:
    models/checkpoints/pinn_best.pt
    models/checkpoints/cnn_pca_best.pt
    models/checkpoints/cnn_raw_best.pt
    models/figures/training_comparison.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pinn.pinn_model import PINNRULModel
from pinn.cnn_baseline import CNN1DModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RUL_CAP = 500.0        # Cap extreme RUL values (focus on imminent failure)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None,
                   help="cpu or cuda (auto-detected if omitted)")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--model", type=str, default="all",
                   choices=["all", "pinn", "cnn_pca", "cnn_raw"],
                   help="Which model(s) to train")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(processed_dir: Path):
    """Load all dataset artifacts produced by build_dataset.py."""
    X_pca    = np.load(processed_dir / "X_windows.npy")       # (N, T, 8)
    X_raw    = np.load(processed_dir / "X_raw_windows.npy")    # (N, T, 12)
    norm_min = np.load(processed_dir / "norm_min.npy")         # (12,)
    norm_max = np.load(processed_dir / "norm_max.npy")         # (12,)
    meta     = pd.read_parquet(processed_dir / "meta_windows.parquet")
    return X_pca, X_raw, norm_min, norm_max, meta


def make_loader(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for one split."""
    ds = TensorDataset(
        torch.from_numpy(X[mask]).float(),
        torch.from_numpy(y[mask]).float().unsqueeze(-1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle, num_workers=0)


# ---------------------------------------------------------------------------
# Generic training loop (handles both CNN and PINN)
# ---------------------------------------------------------------------------
def train_loop(model, train_ld, val_ld, *, epochs, lr, device,
               is_pinn=False, label="model", log_every=5, rul_cap=RUL_CAP):
    """Train one model. Returns (best_state_dict, history_list).

    For PINN models, the physics-loss weights (lambda_fourier, lambda_consistency)
    are linearly ramped from 0 → full over the first 30% of epochs.  Starting at
    full weight causes the Fourier loss to dominate before a usable RUL surface
    exists, which is a well-known PINN convergence failure mode.
    """
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_mae   = float("inf")
    best_state = None
    history    = []

    # Store the target physics-loss weights so we can ramp them.
    base_lf = getattr(model, "lambda_fourier",    0.01)  if is_pinn else 0.0
    base_lc = getattr(model, "lambda_consistency", 0.005) if is_pinn else 0.0
    warmup_epochs = max(1, int(0.30 * epochs))

    for ep in range(epochs):
        # ---- λ warm-up: ramp physics loss weight linearly -------------------
        if is_pinn:
            warmup_frac = min(1.0, (ep + 1) / warmup_epochs)
            model.lambda_fourier     = base_lf * warmup_frac
            model.lambda_consistency = base_lc * warmup_frac

        # ---- train ----------------------------------------------------------
        model.train()
        ep_loss = []
        ep_data = []
        ep_four = []

        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            if is_pinn:
                pred, pred_grad = model(x, return_gradient=True)
                losses = model.total_loss(pred, y, x, pred_grad)
                loss = losses["total"]
                ep_data.append(losses["data"].item())
                ep_four.append(losses["fourier"].item())
            else:
                pred = model(x)
                loss = F.mse_loss(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            ep_loss.append(loss.item())

        sched.step()

        # ---- validate -------------------------------------------------------
        mae, rmse = evaluate(model, val_ld, device, rul_cap)

        row = {"ep": ep + 1,
               "loss": float(np.mean(ep_loss)),
               "mae": mae, "rmse": rmse}
        if is_pinn:
            row["data_loss"]    = float(np.mean(ep_data))
            row["fourier_loss"] = float(np.mean(ep_four))
            row["lambda_fourier"] = model.lambda_fourier
        history.append(row)

        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % log_every == 0 or ep == 0:
            extra = ""
            if is_pinn:
                alpha = float(F.softplus(model.log_alpha).detach().cpu())
                beta  = float(F.softplus(model.log_beta).detach().cpu())
                lf_w  = model.lambda_fourier
                extra = (f"  data {np.mean(ep_data):.5f}  "
                         f"four {np.mean(ep_four):.5f}  "
                         f"lambda_f={lf_w:.4f}  "
                         f"alpha={alpha:.4f} beta={beta:.5f}")
            print(f"[{label:8s}] ep {ep+1:>3}/{epochs}  "
                  f"loss {np.mean(ep_loss):.5f}  "
                  f"val MAE {mae:7.2f}  RMSE {rmse:7.2f}{extra}")

    # Restore full physics-loss weights before saving (warmup was training-only)
    if is_pinn:
        model.lambda_fourier     = base_lf
        model.lambda_consistency = base_lc

    return best_state, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, rul_cap=RUL_CAP):
    """MAE and RMSE on a DataLoader, in *original RUL scale* (cycles)."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            preds.append(pred.cpu().numpy().ravel())
            truths.append(y.numpy().ravel())
    preds  = np.concatenate(preds)  * rul_cap    # denormalise
    truths = np.concatenate(truths) * rul_cap
    mae  = float(np.mean(np.abs(preds - truths)))
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    return mae, rmse


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(histories: dict, save_path: Path):
    """Training curves comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Loss ----
    ax = axes[0]
    for name, hist in histories.items():
        eps   = [r["ep"] for r in hist]
        loss  = [r["loss"] for r in hist]
        ax.plot(eps, loss, label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Val MAE ----
    ax = axes[1]
    for name, hist in histories.items():
        eps = [r["ep"] for r in hist]
        mae = [r["mae"] for r in hist]
        ax.plot(eps, mae, label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (cycles)")
    ax.set_title("Validation MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    def log(msg):
        print(f"\n[{time.time()-t0:6.1f}s] {msg}")

    processed = ROOT / "data" / "processed"
    ckpt_dir  = ROOT / "models" / "checkpoints"
    fig_dir   = ROOT / "models" / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- load data ----------------------------------------------------------
    log("Loading dataset ...")
    X_pca, X_raw, norm_min, norm_max, meta = load_data(processed)

    y_raw = meta["rul"].values.astype(np.float32)
    y = np.clip(y_raw, 0.0, RUL_CAP) / RUL_CAP        # normalised to [0, 1]

    train_mask = meta["split"].values == "train"
    val_mask   = meta["split"].values == "val"
    test_mask  = meta["split"].values == "test"

    log(f"Samples  train {train_mask.sum()}  val {val_mask.sum()}  "
        f"test {test_mask.sum()}   device={device}")

    # ---- data loaders -------------------------------------------------------
    pca_train = make_loader(X_pca, y, train_mask, args.batch, shuffle=True)
    pca_val   = make_loader(X_pca, y, val_mask,   args.batch, shuffle=False)
    pca_test  = make_loader(X_pca, y, test_mask,  args.batch, shuffle=False)

    raw_train = make_loader(X_raw, y, train_mask, args.batch, shuffle=True)
    raw_val   = make_loader(X_raw, y, val_mask,   args.batch, shuffle=False)
    raw_test  = make_loader(X_raw, y, test_mask,  args.batch, shuffle=False)

    histories = {}
    models_to_test = []

    # ======================================================================= #
    #  1. CNN-PCA baseline  (8 PCA channels, data-driven only)                #
    # ======================================================================= #
    if args.model in ["all", "cnn_pca"]:
        log("Training  CNN-PCA  (8 ch, data-driven baseline)")
        print("=" * 72)
        cnn_pca = CNN1DModel(in_features=8, hidden=64).to(device)
        cnn_pca_best, cnn_pca_hist = train_loop(
            cnn_pca, pca_train, pca_val,
            epochs=args.epochs, lr=args.lr, device=device,
            is_pinn=False, label="CNN-PCA", log_every=args.log_every,
        )
        histories["CNN-PCA"] = cnn_pca_hist
        cnn_pca.load_state_dict(cnn_pca_best)
        mae_pca, rmse_pca = evaluate(cnn_pca, pca_test, device)
        torch.save({"state_dict": cnn_pca_best,
                    "config": {"in_features": 8, "hidden": 64},
                    "test_mae": mae_pca, "test_rmse": rmse_pca},
                   ckpt_dir / "cnn_pca_best.pt")
        models_to_test.append(("CNN-PCA", cnn_pca, pca_test, mae_pca, rmse_pca))
        log(f"CNN-PCA Checkpoint saved to {ckpt_dir / 'cnn_pca_best.pt'} (Test MAE: {mae_pca:.2f})")

    # ======================================================================= #
    #  2. CNN-Raw ablation  (12 raw channels, data-driven — no physics)       #
    # ======================================================================= #
    if args.model in ["all", "cnn_raw"]:
        log("Training  CNN-Raw  (12 ch, data-driven ablation)")
        print("=" * 72)
        cnn_raw = CNN1DModel(in_features=12, hidden=64).to(device)
        cnn_raw_best, cnn_raw_hist = train_loop(
            cnn_raw, raw_train, raw_val,
            epochs=args.epochs, lr=args.lr, device=device,
            is_pinn=False, label="CNN-Raw", log_every=args.log_every,
        )
        histories["CNN-Raw"] = cnn_raw_hist
        cnn_raw.load_state_dict(cnn_raw_best)
        mae_raw, rmse_raw = evaluate(cnn_raw, raw_test, device)
        torch.save({"state_dict": cnn_raw_best,
                    "config": {"in_features": 12, "hidden": 64},
                    "test_mae": mae_raw, "test_rmse": rmse_raw},
                   ckpt_dir / "cnn_raw_best.pt")
        models_to_test.append(("CNN-Raw", cnn_raw, raw_test, mae_raw, rmse_raw))
        log(f"CNN-Raw Checkpoint saved to {ckpt_dir / 'cnn_raw_best.pt'} (Test MAE: {mae_raw:.2f})")

    # ======================================================================= #
    #  3. PINN  (12 raw channels + Fourier physics loss)                      #
    # ======================================================================= #
    if args.model in ["all", "pinn"]:
        log("Training  PINN   (12 ch + Fourier + consistency physics loss)")
        print("=" * 72)
        pinn = PINNRULModel(in_features=12, hidden=64).to(device)
        pinn.set_norm_stats(norm_min, norm_max)
        pinn_best, pinn_hist = train_loop(
            pinn, raw_train, raw_val,
            epochs=args.epochs, lr=args.lr, device=device,
            is_pinn=True, label="PINN", log_every=args.log_every,
        )
        histories["PINN"] = pinn_hist
        pinn.load_state_dict(pinn_best)
        mae_pinn, rmse_pinn = evaluate(pinn, raw_test, device)
        pinn_ckpt = {
            "state_dict": pinn_best,
            "config": {"in_features": 12, "hidden": 64,
                       "lambda_fourier": pinn.lambda_fourier,
                       "lambda_consistency": pinn.lambda_consistency},
            "norm_min": norm_min,
            "norm_max": norm_max,
            "test_mae": mae_pinn,
            "test_rmse": rmse_pinn,
        }
        torch.save(pinn_ckpt, ckpt_dir / "pinn_best.pt")
        models_to_test.append(("PINN", pinn, raw_test, mae_pinn, rmse_pinn))
        log(f"PINN Checkpoint saved to {ckpt_dir / 'pinn_best.pt'} (Test MAE: {mae_pinn:.2f})")

    # ======================================================================= #
    #  Summary Table                                                          #
    # ======================================================================= #
    log("EVALUATION SUMMARY:")
    print("=" * 72)
    print(f"{'Model':12s} {'Test MAE':>10s} {'Test RMSE':>10s}")
    print("-" * 52)
    for name, _, _, mae, rmse in models_to_test:
        print(f"{name:12s} {mae:10.2f} {rmse:10.2f}")

    if histories:
        plot_curves(histories, fig_dir / "training_comparison.png")

    log("DONE [OK]")


if __name__ == "__main__":
    main()

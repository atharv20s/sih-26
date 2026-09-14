"""Train the FaultClassifierNet on the engine fleet dataset.

Uses the same `data/processed/` files as train_pinn.py.
The `archetype` column in meta_windows.parquet provides the 4-class label:
  0: vibration_over  1: cht_over  2: egt_over  3: oil_starvation

Outputs:
  models/checkpoints/fault_classifier.pt

Usage (from project root):
    python scripts/train_classifier.py [--epochs 40] [--batch 256] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classifier.fault_classifier import FaultClassifierNet, FAULT_CLASSES


CLASS_TO_IDX = {name: i for i, name in enumerate(FAULT_CLASSES)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs",    type=int,   default=40)
    p.add_argument("--batch",     type=int,   default=256)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--device",    type=str,   default=None)
    p.add_argument("--log-every", type=int,   default=5)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processed = ROOT / "data" / "processed"
    ckpt_dir  = ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_classifier] Loading dataset from {processed} ...")
    X_raw = np.load(processed / "X_raw_windows.npy")   # (N, T, 12) float32
    meta  = pd.read_parquet(processed / "meta_windows.parquet")

    # Apply early-life classifier mask (prevents per-archetype fingerprinting)
    # This mask is written by build_dataset.py --early-life-frac.
    # If missing, warn and fall back to full dataset (reproduces data-leakage condition).
    clf_mask_path = processed / "clf_early_life_mask.npy"
    if clf_mask_path.exists():
        clf_mask = np.load(clf_mask_path)   # (N,) bool
        n_masked = clf_mask.sum()
        print(f"[train_classifier] Early-life mask: {n_masked}/{len(clf_mask)} windows "
              f"({n_masked/max(len(clf_mask),1)*100:.1f}%) -- restricting to early-life windows "
              f"to prevent fault-signature fingerprinting.")
    else:
        clf_mask = np.ones(len(meta), dtype=bool)   # all windows
        print("[train_classifier] WARNING: clf_early_life_mask.npy not found. "
              "Training on full lifecycle -- accuracy may be artificially high (data leakage). "
              "Re-run build_dataset.py with --early-life-frac 0.60 to fix this.")

    # Build class labels (integer)
    if "archetype" in meta.columns:
        y_raw = meta["archetype"].map(CLASS_TO_IDX).values.astype(np.int64)
    else:
        eng_col = "engine_id" if "engine_id" in meta.columns else ("run_id" if "run_id" in meta.columns else None)
        if eng_col:
            unique_engs = {eng: idx % 4 for idx, eng in enumerate(meta[eng_col].unique())}
            y_raw = meta[eng_col].map(unique_engs).values.astype(np.int64)
        else:
            y_raw = np.zeros(len(meta), dtype=np.int64)

    train_mask = (meta["split"].values == "train") & clf_mask
    val_mask   = (meta["split"].values == "val")   & clf_mask
    test_mask  = (meta["split"].values == "test")  & clf_mask

    def make_loader(mask, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X_raw[mask]),
            torch.from_numpy(y_raw[mask]),
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          drop_last=shuffle, num_workers=0)

    train_ld = make_loader(train_mask, shuffle=True)
    val_ld   = make_loader(val_mask,   shuffle=False)
    test_ld  = make_loader(test_mask,  shuffle=False)

    print(f"[train_classifier] train={train_mask.sum()}  val={val_mask.sum()}  "
          f"test={test_mask.sum()}  device={device}")

    model = FaultClassifierNet(in_channels=12, num_classes=4, hidden=48).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc   = 0.0
    best_state = None

    for ep in range(args.epochs):
        model.train()
        ep_loss, ep_correct, ep_total = [], 0, 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            ep_loss.append(loss.item())
            ep_correct += (logits.argmax(dim=1) == y).sum().item()
            ep_total   += len(y)
        sched.step()

        # Validation accuracy
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                val_correct += (model(x).argmax(dim=1) == y).sum().item()
                val_total   += len(y)
        val_acc   = val_correct / max(val_total, 1)
        train_acc = ep_correct  / max(ep_total, 1)

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % args.log_every == 0 or ep == 0:
            print(f"[classifier] ep {ep+1:>3}/{args.epochs}  "
                  f"loss {np.mean(ep_loss):.4f}  "
                  f"train_acc {train_acc*100:.1f}%  "
                  f"val_acc {val_acc*100:.1f}%")

    # Test accuracy
    model.load_state_dict(best_state)
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(device), y.to(device)
            test_correct += (model(x).argmax(dim=1) == y).sum().item()
            test_total   += len(y)
    test_acc = test_correct / max(test_total, 1)

    torch.save({
        "state_dict":   best_state,
        "config":       {"in_channels": 12, "num_classes": 4, "hidden": 48},
        "val_acc":      best_acc,
        "test_acc":     test_acc,
        "fault_classes": FAULT_CLASSES,
    }, ckpt_dir / "fault_classifier.pt")

    print(f"\n[train_classifier] Best val acc: {best_acc*100:.1f}%  "
          f"Test acc: {test_acc*100:.1f}%")
    print(f"[train_classifier] Checkpoint -> {ckpt_dir / 'fault_classifier.pt'}")


if __name__ == "__main__":
    main()

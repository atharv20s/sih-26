"""Multi-Class Aero Engine Fault Classifier (SIH26054).

Classifies run-to-failure telemetry windows into 4 failure archetypes:
  0: vibration_over  (Bearing degradation, mechanical imbalance)
  1: cht_over        (Cylinder head overheating, cooling airflow loss)
  2: oil_starvation  (Lubrication pressure drop, pump cavitation)
  3: egt_over        (Abnormal combustion, lean mixture thermal surge)

Used by:
  - LangGraph Orchestrator (Tool 1 & 2 integration for root-cause diagnosis)
  - 3D Visual Dashboard (Real-time fault probability gauges and component alerting)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]

FAULT_CLASSES = [
    "vibration_over",
    "cht_over",
    "oil_starvation",
    "egt_over",
]

FAULT_DESCRIPTIONS = {
    "vibration_over": "Mechanical fatigue / crankshaft bearing degradation causing high vibration RMS.",
    "cht_over": "Excessive cylinder head thermal stress due to cooling boundary layer breakdown.",
    "oil_starvation": "Lubrication circuit pressure decay risking catastrophic boundary friction.",
    "egt_over": "Exhaust gas thermal runaway indicative of abnormal combustion or extreme lean AFR.",
}

CLASS_TO_IDX = {name: i for i, name in enumerate(FAULT_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(FAULT_CLASSES)}


class FaultClassifierNet(nn.Module):
    """1D-CNN + Global Pooling feature extractor for fault archetype classification."""

    def __init__(self, in_channels: int = 12, num_classes: int = 4, hidden: int = 48, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden * 2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, 12) -> (B, num_classes) logits."""
        z = x.permute(0, 2, 1).contiguous()
        z = self.relu(self.bn1(self.conv1(z)))
        z = self.pool(z)
        z = self.relu(self.bn2(self.conv2(z)))
        z = self.gap(z).squeeze(-1)
        return self.classifier(z)


def load_dataset():
    """Load telemetry windows and associate engine failure archetypes."""
    processed = ROOT / "data" / "processed"
    X_raw = np.load(processed / "X_raw_windows.npy")      # (N, 40, 12)
    meta = pd.read_parquet(processed / "meta_windows.parquet")
    tele = pd.read_parquet(processed / "long_telemetry.parquet", columns=["engine", "failure_type"]).drop_duplicates()

    engine_to_fault = dict(zip(tele["engine"], tele["failure_type"]))
    meta["failure_type"] = meta["engine"].map(engine_to_fault)

    # Filter out rare "none" class (engines that did not reach terminal failure threshold)
    valid_mask = meta["failure_type"].isin(FAULT_CLASSES)
    X_filtered = X_raw[valid_mask]
    meta_filtered = meta[valid_mask].reset_index(drop=True)

    y_labels = np.array([CLASS_TO_IDX[ft] for ft in meta_filtered["failure_type"]], dtype=np.int64)
    splits = meta_filtered["split"].values

    return X_filtered, y_labels, splits


def train_classifier(epochs: int = 15, batch_size: int = 128, lr: float = 1e-3, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FaultClassifier] Training on device: {device}")

    X, y, splits = load_dataset()
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    train_ds = TensorDataset(torch.from_numpy(X[train_mask]).float(), torch.from_numpy(y[train_mask]))
    val_ds = TensorDataset(torch.from_numpy(X[val_mask]).float(), torch.from_numpy(y[val_mask]))
    test_ds = TensorDataset(torch.from_numpy(X[test_mask]).float(), torch.from_numpy(y[test_mask]))

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = FaultClassifierNet(in_channels=12, num_classes=4, hidden=48).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        losses = []
        for bx, by in train_ld:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Evaluate validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_ld:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += len(by)
        val_acc = correct / max(total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:2d}/{epochs:2d} | Train Loss: {np.mean(losses):.4f} | Val Acc: {val_acc*100:5.2f}%")

    # Evaluate test set with best model
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_ld:
            bx = bx.to(device)
            preds = model(bx).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(by.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = float((all_preds == all_targets).mean())
    print(f"\n[FaultClassifier] Final Test Accuracy: {test_acc*100:5.2f}% (Best Val: {best_val_acc*100:5.2f}%)")

    # Save checkpoint
    ckpt_dir = ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "classes": FAULT_CLASSES,
        "test_acc": test_acc,
        "in_channels": 12,
        "num_classes": 4,
    }, ckpt_dir / "fault_classifier.pt")
    print(f"[FaultClassifier] Checkpoint saved to {ckpt_dir / 'fault_classifier.pt'}")

    # Generate and save confusion matrix
    save_confusion_matrix(all_targets, all_preds, ROOT / "models" / "figures" / "fault_confusion_matrix.png")
    return model, test_acc


def save_confusion_matrix(y_true, y_pred, save_path: Path):
    """Plot and save confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(FAULT_CLASSES))))
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(cm_norm, cmap="Blues", alpha=0.85)

        for i in range(len(FAULT_CLASSES)):
            for j in range(len(FAULT_CLASSES)):
                val = cm[i, j]
                pct = cm_norm[i, j] * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=10)

        fig.colorbar(cax)
        ax.set_xticks(range(len(FAULT_CLASSES)))
        ax.set_yticks(range(len(FAULT_CLASSES)))
        ax.set_xticklabels(FAULT_CLASSES, rotation=30, ha="left")
        ax.set_yticklabels(FAULT_CLASSES)
        ax.set_xlabel("Predicted Archetype", fontweight="bold")
        ax.set_ylabel("True Archetype", fontweight="bold")
        ax.set_title("Aero-Piston Engine Fault Archetype Confusion Matrix", pad=20, fontweight="bold")

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[FaultClassifier] Confusion matrix saved to {save_path}")
    except Exception as e:
        print(f"[FaultClassifier] Confusion matrix generation error: {e}")


def predict_fault_from_window(model: FaultClassifierNet, x_norm: torch.Tensor | np.ndarray, device: str = "cpu") -> dict:
    """Inference helper for LangGraph and Dashboard."""
    if isinstance(x_norm, np.ndarray):
        x_tensor = torch.from_numpy(x_norm).float()
    else:
        x_tensor = x_norm.float()

    if x_tensor.dim() == 2:
        x_tensor = x_tensor.unsqueeze(0)  # (1, 40, 12)

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor.to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    top_class = IDX_TO_CLASS[top_idx]
    confidence = float(probs[top_idx])

    return {
        "fault_class": top_class,
        "confidence": confidence,
        "probabilities": {cls: float(probs[i]) for i, cls in enumerate(FAULT_CLASSES)},
        "description": FAULT_DESCRIPTIONS[top_class],
        "severity": "HIGH" if confidence > 0.75 else "MODERATE",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_classifier(epochs=args.epochs, batch_size=args.batch, lr=args.lr)

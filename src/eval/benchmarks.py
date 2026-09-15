"""Benchmark / validation utilities backing MODEL_CARD.md's numbers.

Two things live here:

  - summarize_checkpoints(): re-derives the same metadata dict server.py's
    `_ckpt_meta` exposes via /api/benchmarks, as an importable/testable
    function instead of duplicated top-level script code.
  - evaluate_classifier_ensemble(): actually runs every loaded fault-classifier
    checkpoint (Layer 2 ensemble members) against the held-out test split and
    reports per-member accuracy plus the ensemble's majority-vote accuracy and
    mean agreement score — the real evidence behind the model card's ensemble
    claims (and the source of the "held-out accuracy vs. live-simulator
    accuracy gap" limitation noted there).

Usage:
    python -m src.eval.benchmarks
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def summarize_checkpoints() -> Dict[str, Any]:
    """Loads whatever checkpoints exist in models/checkpoints/ and returns the
    same benchmark metadata shape server.py serves at /api/benchmarks."""
    ckpt_dir = ROOT / "models" / "checkpoints"
    meta: Dict[str, Any] = {}

    pinn_p = ckpt_dir / "pinn_best.pt"
    if pinn_p.exists():
        d = torch.load(pinn_p, map_location="cpu", weights_only=False)
        meta["pinn_mae"] = round(float(d.get("test_mae", float("nan"))), 2)
        meta["pinn_rmse"] = round(float(d.get("test_rmse", float("nan"))), 2)
        meta["pinn_cfg"] = d.get("config", {})

    cls_p = ckpt_dir / "fault_classifier.pt"
    ensemble_paths = sorted(ckpt_dir.glob("fault_classifier*.pt"))
    if cls_p.exists():
        d = torch.load(cls_p, map_location="cpu", weights_only=False)
        meta["clf_test_acc"] = round(float(d.get("test_acc", float("nan"))) * 100, 1)
        meta["clf_val_acc"] = round(float(d.get("val_acc", float("nan"))) * 100, 1)
    meta["clf_ensemble_members"] = [p.name for p in ensemble_paths]

    drl_p = ckpt_dir / "drl_policy.pt"
    if drl_p.exists():
        d = torch.load(drl_p, map_location="cpu", weights_only=False)
        meta["drl_avg_endurance"] = round(float(d.get("avg_endurance", float("nan"))), 1)

    return meta


def evaluate_classifier_ensemble(device: str = "cpu") -> Dict[str, Any]:
    """Runs every fault_classifier*.pt checkpoint against data/processed/'s
    held-out test split and reports per-member + ensemble-vote accuracy.

    Returns a dict summarizing:
      members: [{name, test_acc}]
      ensemble_accuracy: majority-vote accuracy across all loaded members
      mean_agreement: average fraction of members agreeing with the majority
    """
    from classifier.fault_classifier import FaultClassifierNet, IDX_TO_CLASS
    from agent.defense_layers import vote_fault_classification
    import pandas as pd

    processed = ROOT / "data" / "processed"
    ckpt_dir = ROOT / "models" / "checkpoints"
    ensemble_paths = sorted(ckpt_dir.glob("fault_classifier*.pt"))
    if not ensemble_paths:
        return {"members": [], "ensemble_accuracy": None, "mean_agreement": None}

    X_raw = np.load(processed / "X_raw_windows.npy")
    meta = pd.read_parquet(processed / "meta_windows.parquet")

    # archetype label isn't a column on meta_windows.parquet directly — join
    # it in from long_telemetry.parquet by engine, same as
    # classifier/fault_classifier.py's own load_dataset().
    classes = ["vibration_over", "cht_over", "oil_starvation", "egt_over"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    tele = pd.read_parquet(processed / "long_telemetry.parquet", columns=["engine", "failure_type"]).drop_duplicates()
    engine_to_fault = dict(zip(tele["engine"], tele["failure_type"]))
    meta = meta.copy()
    meta["failure_type"] = meta["engine"].map(engine_to_fault)

    clf_mask_path = processed / "clf_early_life_mask.npy"
    clf_mask = np.load(clf_mask_path) if clf_mask_path.exists() else np.ones(len(meta), dtype=bool)
    valid_class = meta["failure_type"].isin(classes)
    test_mask = (meta["split"].values == "test") & clf_mask & valid_class.values

    if test_mask.sum() == 0:
        return {"members": [], "ensemble_accuracy": None, "mean_agreement": None}

    y_true = meta.loc[test_mask, "failure_type"].map(class_to_idx).values
    X_test = torch.from_numpy(X_raw[test_mask]).float()

    members = []
    all_preds: List[np.ndarray] = []
    for p in ensemble_paths:
        data = torch.load(p, map_location=device, weights_only=False)
        model = FaultClassifierNet(in_channels=12, num_classes=4, hidden=48)
        model.load_state_dict(data["state_dict"])
        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(dim=-1).numpy()
        acc = float((preds == y_true).mean())
        members.append({"name": p.name, "test_acc": round(acc, 4)})
        all_preds.append(preds)

    all_preds_arr = np.stack(all_preds, axis=1)  # (N, n_members)
    votes, agreements = [], []
    for row in all_preds_arr:
        cls_names = [IDX_TO_CLASS[int(idx)] for idx in row]
        majority, agreement = vote_fault_classification(cls_names)
        votes.append(majority)
        agreements.append(agreement)

    class_to_idx = {c: i for i, c in enumerate(["vibration_over", "cht_over", "oil_starvation", "egt_over"])}
    vote_idx = np.array([class_to_idx[v] for v in votes])
    ensemble_accuracy = float((vote_idx == y_true).mean())
    mean_agreement = float(np.mean(agreements))

    return {
        "members": members,
        "ensemble_accuracy": round(ensemble_accuracy, 4),
        "mean_agreement": round(mean_agreement, 4),
        "n_test_windows": int(test_mask.sum()),
    }


if __name__ == "__main__":
    print("[eval.benchmarks] Checkpoint summary:")
    for k, v in summarize_checkpoints().items():
        print(f"  {k}: {v}")

    print("\n[eval.benchmarks] Classifier ensemble evaluation (held-out test split):")
    result = evaluate_classifier_ensemble()
    for m in result["members"]:
        print(f"  {m['name']}: test_acc={m['test_acc']*100:.1f}%")
    if result["ensemble_accuracy"] is not None:
        print(f"  Ensemble (majority vote): accuracy={result['ensemble_accuracy']*100:.1f}%  "
              f"mean_agreement={result['mean_agreement']*100:.1f}%  "
              f"n_test_windows={result['n_test_windows']}")

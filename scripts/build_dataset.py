"""Build the sliding-window dataset for PINN/CNN training and fault classification.

Uses `src/sim/engine_sim.generate_fleet()` to produce physically realistic run-to-failure
telemetry, then windows and normalises it into the exact files that `src/pinn/train_pinn.py`
and `scripts/train_classifier.py` expect.

Outputs (all written to `data/processed/`):
  X_raw_windows.npy      (N, T, 12)  float32  -- min-max normalised raw channels
  X_windows.npy          (N, T,  8)  float32  -- PCA-reduced (8 components from 12)
  norm_min.npy           (12,)        float32  -- per-channel min (train split only)
  norm_max.npy           (12,)        float32  -- per-channel max (train split only)
  meta_windows.parquet   DataFrame    -- run_id, cycle, rul, health, split, archetype

Usage (from project root):
    python scripts/build_dataset.py [--engines 80] [--cycles 600] [--window 30]
                                    [--stride 1] [--seed 0] [--rul-cap 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.engine_sim import generate_fleet, SENSOR_ORDER

ARCHETYPE_NAMES = ["vibration_over", "cht_over", "egt_over", "oil_starvation"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engines",  type=int,   default=80,   help="Engines per archetype group (total = 4 x this)")
    p.add_argument("--cycles",   type=int,   default=600,  help="Max cycles per engine run")
    p.add_argument("--window",   type=int,   default=30,   help="Sliding window length T")
    p.add_argument("--stride",   type=int,   default=1,    help="Window stride (1 = maximum overlap, keeps failure tail)")
    p.add_argument("--seed",     type=int,   default=0,    help="Master RNG seed")
    p.add_argument("--rul-cap",  type=float, default=500.0, help="Cap applied to RUL before normalising to [0,1]")
    p.add_argument("--pca-components", type=int, default=8, help="PCA components for reduced X_windows")
    p.add_argument(
        "--early-life-frac", type=float, default=0.60,
        help=(
            "Fraction of each engine's life used for CLASSIFIER training windows (0-1). "
            "Only windows where rul > (1 - frac) * max_rul_of_run are included. "
            "Default 0.60 = first 60%% of life, preventing fault-signature fingerprinting. "
            "PINN/CNN windows always use the full life (fault tail is required for RUL). "
            "Set to 1.0 to disable the restriction (reproduces the data-leakage condition)."
        ),
    )
    return p.parse_args()


def sliding_windows(data_cols: dict, rul: np.ndarray, health: np.ndarray,
                    run_id: str, archetype: str,
                    window: int, stride: int,
                    early_life_frac: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Slice one engine run into overlapping windows.

    Parameters
    ----------
    early_life_frac : float
        When < 1.0, also produces a mask array (``clf_mask``) that is True only
        for windows falling in the first ``early_life_frac`` of each engine's life.
        This is the classifier-only subset — restricting to early-life windows
        prevents the classifier from learning per-archetype trajectory fingerprints
        that are trivially identifiable in late-degradation windows.

    Returns
    -------
    X      : (n_windows, window, 12)  float32 — raw channel values (physical units)
    clf_ok : (n_windows,)             bool    — True for windows valid for classifier
    meta   : list of dicts with keys: run_id, cycle, rul, health, archetype
    """
    n = len(rul)
    sensor_matrix = np.stack([data_cols[k] for k in SENSOR_ORDER], axis=1)  # (n, 12)
    max_rul = float(rul.max()) if rul.max() > 0 else 1.0
    # Classifier early-life threshold: keep windows where rul > threshold
    # rul high = early life; rul low = near failure
    clf_rul_threshold = (1.0 - early_life_frac) * max_rul

    windows, clf_oks, metas = [], [], []
    for start in range(0, n - window + 1, stride):
        end = start + window
        win = sensor_matrix[start:end]          # (T, 12)
        rul_label = float(rul[end - 1])
        h_label   = float(health[end - 1])
        # Classifier early-life gate: True if this window is in the early life zone
        clf_ok = (rul_label > clf_rul_threshold)
        windows.append(win.astype(np.float32))
        clf_oks.append(clf_ok)
        metas.append({
            "run_id":    run_id,
            "cycle":     end - 1,
            "rul":       rul_label,
            "health":    h_label,
            "archetype": archetype,
            "clf_ok":    clf_ok,
        })

    if not windows:
        return (np.empty((0, window, 12), dtype=np.float32),
                np.empty((0,), dtype=bool),
                [])

    return np.stack(windows, axis=0), np.array(clf_oks, dtype=bool), metas


def main():
    args = parse_args()
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_engines = args.engines * 4   # 4 archetypes, engines per archetype
    print(f"[build_dataset] Generating {total_engines} engine runs "
          f"({args.engines} × 4 archetypes) × {args.cycles} cycles each ...")

    engines = generate_fleet(
        n_engines=total_engines,
        cycles_per_engine=args.cycles,
        seed=args.seed,
        profile="cruise_dominant",
    )

    # -------------------------------------------------------------------------
    # Slice all engines into windows
    # -------------------------------------------------------------------------
    all_X, all_meta = [], []
    rng = np.random.default_rng(args.seed + 99)

    # Assign train/val/test split per engine (80/10/10)
    split_arr = []
    for i in range(total_engines):
        r = rng.random()
        if r < 0.80:
            split_arr.append("train")
        elif r < 0.90:
            split_arr.append("val")
        else:
            split_arr.append("test")

    for i, (eng_id, data, truth) in enumerate(engines):
        archetype = ARCHETYPE_NAMES[i % 4]
        X_run, clf_ok_run, meta_run = sliding_windows(
            data_cols=data,
            rul=truth["rul"],
            health=truth["health"],
            run_id=eng_id,
            archetype=archetype,
            window=args.window,
            stride=args.stride,
            early_life_frac=args.early_life_frac,
        )
        if len(X_run) == 0:
            continue
        for j, m in enumerate(meta_run):
            m["split"] = split_arr[i]
        all_X.append(X_run)
        all_meta.extend(meta_run)

    X_all = np.concatenate(all_X, axis=0)  # (N_total, T, 12)
    meta_df = pd.DataFrame(all_meta)

    N_total = len(X_all)
    print(f"[build_dataset] Total windows: {N_total}  "
          f"(train {(meta_df.split=='train').sum()}  "
          f"val {(meta_df.split=='val').sum()}  "
          f"test {(meta_df.split=='test').sum()})")

    # -------------------------------------------------------------------------
    # Fit min-max normaliser on TRAINING windows only (prevents data leakage)
    # -------------------------------------------------------------------------
    train_mask = meta_df["split"].values == "train"
    X_train = X_all[train_mask]                    # (N_train, T, 12)
    X_flat   = X_train.reshape(-1, 12)             # (N_train*T, 12)

    norm_min = X_flat.min(axis=0).astype(np.float32)   # (12,)
    norm_max = X_flat.max(axis=0).astype(np.float32)    # (12,)
    rng_safe  = np.maximum(norm_max - norm_min, 1e-8)

    X_norm = ((X_all - norm_min[None, None, :]) / rng_safe[None, None, :]).astype(np.float32)
    X_norm = np.clip(X_norm, 0.0, 1.0)

    np.save(out_dir / "norm_min.npy", norm_min)
    np.save(out_dir / "norm_max.npy", norm_max)
    print(f"[build_dataset] Saved norm_min.npy, norm_max.npy")

    # -------------------------------------------------------------------------
    # Save raw normalised windows  (N, T, 12)
    # -------------------------------------------------------------------------
    np.save(out_dir / "X_raw_windows.npy", X_norm)
    print(f"[build_dataset] Saved X_raw_windows.npy  shape={X_norm.shape}")

    # -------------------------------------------------------------------------
    # Classifier early-life windows: PINN needs full lifecycle, classifier
    # should only see early-life windows to avoid per-archetype fingerprinting.
    # -------------------------------------------------------------------------
    clf_mask_col = meta_df["clf_ok"].values.astype(bool)
    n_clf = clf_mask_col.sum()
    early_pct = args.early_life_frac * 100
    print(f"[build_dataset] Classifier early-life windows (first {early_pct:.0f}% of life): "
          f"{n_clf} / {N_total} = {n_clf/max(N_total,1)*100:.1f}%")
    if args.early_life_frac >= 0.99:
        print("[build_dataset] WARNING: --early-life-frac=1.0 includes the full lifecycle. "
              "The fault classifier WILL see late-degradation windows where archetype "
              "fingerprints are dominant. Expect artificially high accuracy (data leakage).")

    # Save the classifier-specific normalised windows
    # (same normalisation stats — the restriction is only on which windows are included)
    np.save(out_dir / "X_clf_windows.npy", X_norm)          # full set (for loading convenience)
    np.save(out_dir / "clf_early_life_mask.npy", clf_mask_col)  # boolean mask
    print(f"[build_dataset] Saved clf_early_life_mask.npy  ({clf_mask_col.sum()} True)")

    # -------------------------------------------------------------------------
    # PCA reduction: fit on training flat frames, transform all windows
    # -------------------------------------------------------------------------
    print(f"[build_dataset] Fitting PCA ({args.pca_components} components) on training data ...")
    X_train_norm_flat = X_norm[train_mask].reshape(-1, 12)   # (N_train*T, 12)
    pca = PCA(n_components=args.pca_components, random_state=args.seed)
    pca.fit(X_train_norm_flat)
    print(f"[build_dataset] PCA explained variance: "
          f"{pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}% with {args.pca_components} components")

    # Transform window-by-window to preserve temporal structure
    N, T, _ = X_norm.shape
    X_pca = pca.transform(X_norm.reshape(-1, 12)).reshape(N, T, args.pca_components)
    X_pca = X_pca.astype(np.float32)

    np.save(out_dir / "X_windows.npy", X_pca)
    print(f"[build_dataset] Saved X_windows.npy  shape={X_pca.shape}")

    # -------------------------------------------------------------------------
    # Save metadata parquet
    # -------------------------------------------------------------------------
    meta_df.to_parquet(out_dir / "meta_windows.parquet", index=False)
    print(f"[build_dataset] Saved meta_windows.parquet  ({len(meta_df)} rows)")

    # -------------------------------------------------------------------------
    # Quick sanity check
    # -------------------------------------------------------------------------
    print("\n[build_dataset] Sanity check:")
    print(f"  X_raw_windows : {X_norm.shape}  dtype={X_norm.dtype}  "
          f"min={X_norm.min():.3f}  max={X_norm.max():.3f}")
    print(f"  X_windows     : {X_pca.shape}  dtype={X_pca.dtype}")
    print(f"  RUL range     : {meta_df.rul.min():.0f} – {meta_df.rul.max():.0f} cycles")
    print(f"  Archetypes    : {meta_df.archetype.value_counts().to_dict()}")
    print("[build_dataset] Done.")


if __name__ == "__main__":
    main()

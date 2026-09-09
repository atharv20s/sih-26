"""Build the SIH26054 piston-engine run-to-failure dataset on this PC.

All data stays local under data/raw and data/processed. Nothing uploads.

Pipeline (mirrors Shen et al. 2025 preprocessing, adapted to our synthetic
aero-piston engine telemetry):
  1. Generate a heterogeneous fleet (varied starting health, degradation rate)
     from the physics-based simulator in src/sim/engine_sim.py.
  2. Per-engine light EMA smoothing (already applied in the simulator).
  3. Max-min normalization fit on TRAIN engines only (no leakage to val/test).
  4. PCA to `PCA_COMPONENTS` dims, fit on train, transform all (paper: 8 comps
     retain >99.9% variance).
  5. Sliding-window each engine's normalized trajectory into (T x n_feat)
     samples; label = true RUL at the last cycle of the window.

Outputs (in data/):
  processed/X_windows.npy        (N, T, F) float32  sliding windows
  processed/meta_windows.parquet  window metadata (engine, cycle, rul, split...)
  processed/long_telemetry.parquet (engine, cycle, all 12 sensors, rul, health)
  raw/fleet_raw.parquet           fleet-wide raw telemetry
"""
from __future__ import annotations

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from sim.engine_sim import SENSOR_ORDER, generate_fleet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_TRAIN = 240
N_VAL = 30
N_TEST = 30
N_ENGINES = N_TRAIN + N_VAL + N_TEST
CYCLES = 1200
SEED = 20260909

PCA_COMPONENTS = 8
WINDOW = 40
STRIDE = 5          # stride 5 keeps the set dense enough but 5x smaller than 1
PER_FEATURE_CACHE = False


def main():
    t0 = time.time()
    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    processed_dir = ROOT / "data" / "processed"
    raw_dir = ROOT / "data" / "raw"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. generate fleet --------------------------------------------------
    log(f"generate fleet {N_ENGINES} x {CYCLES} cycles")
    engines = generate_fleet(N_ENGINES, CYCLES, seed=SEED,
                             profile="cruise_dominant",
                             health_range=(0.55, 1.0), driver_range=(0.85, 1.30))

    # Pack raw telemetry into one float32 matrix (E, C, S) + per-engine truth.
    E, C, S = len(engines), CYCLES, len(SENSOR_ORDER)
    tele = np.zeros((E, C, S), dtype=np.float32)
    rul = np.zeros((E, C), dtype=np.float32)
    health = np.zeros((E, C), dtype=np.float32)
    failure_cycle = np.zeros(E, dtype=np.int64)
    failure_type = np.array([""], dtype=object)
    for e, (eid, data, truth) in enumerate(engines):
        for si, s in enumerate(SENSOR_ORDER):
            tele[e, :, si] = data[s][:C]
        rul[e] = truth["rul"][:C]
        health[e] = truth["health"][:C]
        failure_cycle[e] = truth["failure_cycle"]
    failure_type = np.array([e[2]["failure_type"] for e in engines], dtype=object)
    eng_ids = np.array([e[0] for e in engines])
    log(f"telemetry tensor {tele.shape}")

    np.save(raw_dir / "tele.npy", tele)
    np.save(raw_dir / "rul_truth.npy", rul)
    np.save(raw_dir / "health_truth.npy", health)
    np.save(raw_dir / "failure_cycle.npy", failure_cycle)
    np.save(raw_dir / "engine_ids.npy", eng_ids)
    np.save(raw_dir / "failure_types.npy", failure_type)

    # ---- 2. normalization (train-only stats) --------------------------------
    train_ids = np.arange(N_TRAIN)
    val_ids = np.arange(N_TRAIN, N_TRAIN + N_VAL)
    test_ids = np.arange(N_TRAIN + N_VAL, N_ENGINES)
    tr = tele[train_ids]                      # (N_train, C, S)
    mins = tr.min(axis=(0, 1))                # (S,)
    maxs = tr.max(axis=(0, 1))
    rng = np.maximum(maxs - mins, 1e-9)
    norm = (tele - mins) / rng                # broadcast over all engines
    log(f"max-min normalized (train stats) min={mins.round(3)} max={maxs.round(3)}")

    # ---- 3. PCA (train-only, fit on normalized train slice) -----------------
    from sklearn.decomposition import PCA
    tr_norm = norm[train_ids]               # normalized train engines only
    tr_flat = tr_norm.reshape(-1, S)
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(tr_flat)
    var = float(pca.explained_variance_ratio_.sum())
    log(f"PCA fit on train: {PCA_COMPONENTS} comps, variance kept {var:.5f}")
    all_flat = norm.reshape(-1, S)
    pc = pca.transform(all_flat).astype(np.float32).reshape(E, C, PCA_COMPONENTS)
    np.save(processed_dir / "pca_components.npy", pca.components_)
    np.save(processed_dir / "pca_mean.npy", pca.mean_)
    del all_flat, tr_flat
    log(f"PCA transform -> {pc.shape}")

    # ---- 4. windowing --------------------------------------------------------
    T = WINDOW
    windows = []
    raw_windows = []
    w_engine = []
    w_cycle = []
    w_rul = []
    w_health = []
    for e in range(E):
        n = C - T + 1
        idxs = np.arange(0, n, STRIDE)
        if len(idxs) == 0:
            continue
        w = np.stack([pc[e, i:i + T, :] for i in idxs])  # (m, T, F)
        raw_w = np.stack([norm[e, i:i + T, :] for i in idxs])  # (m, T, S)
        windows.append(w)
        raw_windows.append(raw_w)
        w_engine.append(np.full(len(idxs), e, dtype=np.int32))
        w_cycle.append(idxs + T - 1)
        w_rul.append(rul[e, idxs + T - 1])
        w_health.append(health[e, idxs + T - 1])
    X = np.concatenate(windows, axis=0)             # (N, T, F_pca)
    X_raw = np.concatenate(raw_windows, axis=0)     # (N, T, S) normalized raw channels
    meta = pd.DataFrame({
        "engine_id": np.concatenate(w_engine),
        "engine": eng_ids[np.concatenate(w_engine).astype(int)],
        "cycle": np.concatenate(w_cycle),
        "rul": np.concatenate(w_rul),
        "health": np.concatenate(w_health),
    })
    # split label by engine
    split = np.empty(len(meta), dtype=object)
    split[np.isin(meta["engine_id"].values, train_ids)] = "train"
    split[np.isin(meta["engine_id"].values, val_ids)] = "val"
    split[np.isin(meta["engine_id"].values, test_ids)] = "test"
    meta["split"] = split
    log(f"windows {X.shape}  split counts { {s: int((split==s).sum()) for s in ['train','val','test']} }")

    # ---- 5. save -------------------------------------------------------------
    np.save(processed_dir / "X_windows.npy", X)
    np.save(processed_dir / "X_raw_windows.npy", X_raw)
    # Norm min/max needed to denormalize raw channels back to physical units
    # (the PINN physics loss runs on CHT in degC and fuel flow in L/h).
    np.save(processed_dir / "norm_min.npy", mins)
    np.save(processed_dir / "norm_max.npy", maxs)
    meta.to_parquet(processed_dir / "meta_windows.parquet", index=True)

    # Long table for dashboard + EDA.
    long = pd.DataFrame({
        "engine": np.repeat(eng_ids, C),
        "cycle": np.tile(np.arange(C), E),
    })
    for si, s in enumerate(SENSOR_ORDER):
        long[s] = tele[:, :, si].ravel()
    long["rul"] = rul.ravel()
    long["health"] = health.ravel()
    long["failure_cycle"] = np.repeat(failure_cycle, C)
    long["failure_type"] = np.repeat(failure_type, C)
    eng_split = np.empty(E, dtype=object)
    eng_split[train_ids] = "train"
    eng_split[val_ids] = "val"
    eng_split[test_ids] = "test"
    long["split"] = np.repeat(eng_split, C)
    long.to_parquet(processed_dir / "long_telemetry.parquet", index=False)
    log(f"saved long_telemetry {long.shape}")

    log(f"DONE. RUL mean train {meta.loc[meta.split=='train','rul'].mean():.1f} "
        f"test {meta.loc[meta.split=='test','rul'].mean():.1f}")
    log("failure type mix (per engine): " +
        str(pd.Series(failure_type).value_counts().to_dict()))
    log("all artifacts under data/raw and data/processed")


if __name__ == "__main__":
    main()

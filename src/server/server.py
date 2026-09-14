"""FastAPI Server & WebSocket Telemetry Streamer for SIH26054 Digital Twin.

Provides:
  - Real-time WebSocket streaming (/ws/telemetry) at 10 Hz
  - REST endpoints for system state, benchmarks, and interactive fault injection
  - Static file hosting for the 3D WebGL Three.js Digital Twin Dashboard
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="SIH26054 MALE UAV Engine Digital Twin Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic request models
class FaultInjectionRequest(BaseModel):
    fault_type: Optional[str] = None

class EnvironmentRequest(BaseModel):
    altitude: float
    ambient_temp: float

# ---------------------------------------------------------------------------
# Live Engine Simulation Adapter (P0.5)
# ---------------------------------------------------------------------------
# Wraps src/sim/engine_sim.EngineSim so the live demo uses the same physics
# that generated the training dataset.  Pre-computes 5 fault-archetype
# trajectories at startup; inject_fault() switches between them.
# The interface (cycle, health, throttle, cht, etc.) is identical to the
# original SimulationController so no other code in this file needs to change.
# ---------------------------------------------------------------------------

_FAULT_ARCHETYPES = {
    None:              {},                                                           # nominal
    "thermal_shock":   {"r_th": {"coef": 1.40, "exp": 0.50}, "vib": {"exp": 3.00}},# cht_over
    "oil_leak":        {"eta_lub": {"coef": 1.30, "exp": 0.45},
                        "mu_f":    {"coef": 1.30, "exp": 0.50},
                        "r_th":    {"coef": 0.25, "exp": 3.00},
                        "vib":     {"exp": 3.50}},                                  # oil_starvation
    "vibration_spike": {"vib": {"exp": 0.55}},                                      # vibration_over
    "egt_over_demo":   {"eta_c": {"coef": 1.20, "exp": 0.50},
                        "r_th":  {"coef": 0.22, "exp": 3.00},
                        "vib":   {"exp": 3.50}},                                    # egt_over
}


def _presim(fault_type: Optional[str], total_cycles: int, seed: int = 7) -> tuple:
    """Run EngineSim for one fault archetype and return (data_dict, truth_dict)."""
    try:
        from sim.engine_sim import EngineSim, MissionProfile
        bias = _FAULT_ARCHETYPES.get(fault_type, {})
        sim  = EngineSim(
            total_cycles=total_cycles,
            seed=seed,
            profile=MissionProfile.cruise_dominant(total_cycles),
            initial_health=0.92,
            degradation_driver=1.05,
            failure_bias=bias,
        )
        return sim.run(add_noise=True)
    except Exception as exc:
        print(f"[LiveEngineSim] EngineSim presim failed for {fault_type!r}: {exc}")
        return None, None


class SimulationController:
    """Live telemetry adapter backed by EngineSim (same physics as training data).

    The five fault modes correspond to EngineSim failure_bias archetypes:
      - None / clear     → nominal cruise degradation
      - thermal_shock    → cht_over archetype
      - oil_leak         → oil_starvation archetype
      - vibration_spike  → vibration_over archetype
      - egt_over_demo    → egt_over archetype

    sensor_dropout is applied as a NaN injection post-step (independent of physics).

    When EngineSim is not importable (e.g. missing dependency), the adapter
    falls back to the previous hand-scripted physics and logs a warning.
    """

    _TOTAL_CYCLES = 1200   # pre-simulated trajectory length (each fault mode)

    def __init__(self):
        self.cycle         = 0
        self.throttle      = 0.72
        self.mixture       = 13.8
        self.injected_fault: Optional[str] = None
        self.is_paused     = False
        self.altitude      = 12500.0
        self.ambient_temp  = 15.0
        self.air_density   = 0.812
        self.cooling_factor = 1.0

        # ---- Pre-compute trajectories for all fault archetypes ---------------
        print("[LiveEngineSim] Pre-computing fault trajectories (this takes ~2–5 s) …")
        self._trajs: Dict[Optional[str], tuple] = {}
        for fault in list(_FAULT_ARCHETYPES.keys()):
            data, truth = _presim(fault, self._TOTAL_CYCLES)
            if data is not None:
                self._trajs[fault] = (data, truth)
        self._using_enginesim = bool(self._trajs)

        if not self._using_enginesim:
            print("[LiveEngineSim] WARNING: EngineSim unavailable — falling back to "
                  "hand-scripted physics.  Live demo will NOT use the same physics as "
                  "the training dataset.")

        # Expose attributes the rest of server.py reads directly
        self._reset_state_attrs()

    def _reset_state_attrs(self):
        """Initialise all sensor attributes to nominal cruise values."""
        self.rpm          = 4535.4
        self.cht          = 148.6
        self.egt          = 667.7
        self.oil_temp     = 96.4
        self.oil_pressure = 281.8
        self.fuel_flow    = 8.30
        self.vibration_rms = 1.41
        self.map_kpa      = 92.0
        self.coolant_temp = 82.0
        self.health       = 1.0

    def set_environment(self, altitude: float, ambient_temp: float):
        self.altitude     = float(np.clip(altitude,     0.0, 25000.0))
        self.ambient_temp = float(np.clip(ambient_temp, -50.0, 60.0))
        density_ratio     = max(0.05, (1.0 - 2.25577e-5 * self.altitude) ** 4.25588)
        self.air_density  = round(1.225 * density_ratio, 3)
        self.cooling_factor = round(
            max(0.35, (1.0 + (15.0 - self.ambient_temp) * 0.012) * np.sqrt(density_ratio)), 2)

    def inject_fault(self, fault_type: Optional[str]):
        self.injected_fault = fault_type
        # Reset cycle so the new archetype trajectory starts from the beginning
        self.cycle = 0
        self._reset_state_attrs()

    def step(self) -> Dict[str, float]:
        self.cycle += 1

        # ---- EngineSim path -------------------------------------------------
        if self._using_enginesim:
            return self._step_enginesim()

        # ---- Fallback: hand-scripted path -----------------------------------
        return self._step_scripted()

    def _step_enginesim(self) -> Dict[str, float]:
        """Advance one cycle using the pre-computed EngineSim trajectory."""
        from sim.engine_sim import SENSOR_ORDER

        # Select active trajectory (fall back to nominal if archetype not cached)
        fault_key = self.injected_fault if self.injected_fault in self._trajs else None
        data, truth = self._trajs[fault_key]

        # Index: wrap around if we exceed the pre-simulated length
        idx = (self.cycle - 1) % self._TOTAL_CYCLES

        # Build sensor frame from pre-computed arrays
        frame = {k: float(data[k][idx]) for k in SENSOR_ORDER}

        # Apply throttle/mixture modulation from DRL (offset on top of trajectory)
        d_thr = (self.throttle - 0.72) * 15.0   # ~±15 RPM / °C scaling
        frame["rpm"]  = float(np.clip(frame["rpm"] + d_thr * 60.0, 3500.0, 6000.0))
        frame["cht"]  = float(np.clip(frame["cht"] + d_thr * 0.8,   90.0,  270.0))
        frame["egt"]  = float(np.clip(frame["egt"] + d_thr * 2.5,  500.0,  900.0))

        # Altitude/density corrections on top of base trajectory
        density_ratio = max(0.05, (1.0 - 2.25577e-5 * self.altitude) ** 4.25588)
        self.air_density    = round(1.225 * density_ratio, 3)
        self.cooling_factor = round(
            max(0.35, (1.0 + (15.0 - self.ambient_temp) * 0.012) * np.sqrt(density_ratio)), 2)
        cooling_penalty = (1.0 - self.cooling_factor) * 30.0
        frame["cht"] = float(np.clip(frame["cht"] + cooling_penalty, 90.0, 270.0))

        # Update exposed attributes for direct server.py access
        self.rpm          = frame["rpm"]
        self.cht          = frame["cht"]
        self.egt          = frame["egt"]
        self.oil_temp     = frame["oil_temp"]
        self.oil_pressure = frame["oil_pressure"]
        self.fuel_flow    = frame["fuel_flow"]
        self.vibration_rms = frame["vibration_rms"]
        self.map_kpa      = frame.get("map_kpa", 92.0)
        self.coolant_temp = frame.get("coolant_temp", 82.0)
        self.health       = float(truth["health"][idx])

        # sensor_dropout: inject NaN on top of any physics trajectory
        if self.injected_fault == "sensor_dropout":
            frame["cht"]          = float("nan")
            frame["oil_pressure"] = -999.0

        return {
            "rpm":           round(self.rpm, 1),
            "cht":           frame["cht"] if self.injected_fault == "sensor_dropout" else round(self.cht, 1),
            "egt":           round(self.egt, 1),
            "oil_temp":      round(self.oil_temp, 1),
            "oil_pressure":  frame["oil_pressure"] if self.injected_fault == "sensor_dropout" else round(self.oil_pressure, 1),
            "fuel_flow":     round(self.fuel_flow, 2),
            "vibration_rms": round(self.vibration_rms, 2),
            "map":           round(self.map_kpa, 1),
            "afr":           round(self.mixture, 2),
            "torque":        24.5,
            "crank_pos":     float((self.cycle * 35) % 360),
            "coolant_temp":  round(self.coolant_temp, 1),
            "altitude":      round(self.altitude, 1),
            "ambient_temp":  round(self.ambient_temp, 1),
            "air_density":   self.air_density,
            "cooling_factor": self.cooling_factor,
        }

    def _step_scripted(self) -> Dict[str, float]:
        """Fallback hand-scripted physics (identical to the original SimulationController)."""
        noise = lambda scale: float(np.random.normal(0, scale))
        density_ratio   = max(0.05, (1.0 - 2.25577e-5 * self.altitude) ** 4.25588)
        self.air_density    = round(1.225 * density_ratio, 3)
        self.cooling_factor = round(
            max(0.35, (1.0 + (15.0 - self.ambient_temp) * 0.012) * np.sqrt(density_ratio)), 2)

        temp_delta   = self.ambient_temp - 15.0
        cooling_loss = (1.0 - self.cooling_factor) * 35.0

        self.rpm      = 4535.4 + (self.throttle - 0.72) * 1200.0 + temp_delta * 2.0 + noise(8.0)
        base_cht      = 148.6  + (self.throttle - 0.72) * 35.0 + (13.8 - self.mixture) * 5.0 + temp_delta * 0.75 + cooling_loss
        base_egt      = 667.7  + (self.throttle - 0.72) * 80.0 + (self.mixture - 13.8) * 12.0 + temp_delta * 0.45 + cooling_loss * 0.35
        base_vib      = 1.41   + (self.throttle - 0.72) * 0.70 + (self.altitude / 10000.0) * 0.28 + max(0.0, (base_cht - 148.0) * 0.014)
        base_oil_temp = 85.0   + temp_delta * 0.45 + cooling_loss * 0.25
        base_oil_p    = 281.8  - (base_oil_temp - 85.0) * 1.3 - (self.altitude / 10000.0) * 8.0

        self.health   = max(0.85, 1.0 - (self.cycle * 0.00002))
        deg_factor    = 1.0 - self.health

        if self.injected_fault == "thermal_shock":
            self.cht          = self.cht + (212.0 - self.cht) * 0.08 + noise(0.2)
            self.egt          = self.egt + (730.0 - self.egt) * 0.04 + noise(0.3)
            self.vibration_rms = self.vibration_rms + (1.85 - self.vibration_rms) * 0.04 + noise(0.01)
        elif self.injected_fault == "sensor_dropout":
            return {
                "rpm": round(self.rpm, 1), "cht": float("nan"), "egt": round(self.egt, 1),
                "oil_temp": round(self.oil_temp, 1), "oil_pressure": -999.0,
                "fuel_flow": round(self.fuel_flow, 2), "vibration_rms": round(self.vibration_rms, 2),
                "map": round(self.map_kpa, 1), "afr": round(self.mixture, 2), "torque": 24.5,
                "crank_pos": float((self.cycle * 35) % 360), "coolant_temp": round(self.coolant_temp, 1),
                "altitude": round(self.altitude, 1), "ambient_temp": round(self.ambient_temp, 1),
                "air_density": self.air_density, "cooling_factor": self.cooling_factor,
            }
        elif self.injected_fault == "oil_leak":
            self.oil_pressure  = self.oil_pressure + (148.0 - self.oil_pressure) * 0.08 + noise(0.4)
            self.oil_temp      = self.oil_temp + (115.0 - self.oil_temp) * 0.04 + noise(0.1)
            self.vibration_rms = self.vibration_rms + (2.05 - self.vibration_rms) * 0.04 + noise(0.01)
        elif self.injected_fault == "vibration_spike":
            self.vibration_rms = self.vibration_rms + (3.25 - self.vibration_rms) * 0.08 + noise(0.02)
        else:
            phase = (self.cycle / 55.0) * 2.0 * np.pi
            self.cht          = self.cht + (base_cht + deg_factor * 15.0 + 0.85 * np.sin(phase) - self.cht) * 0.08
            self.egt          = self.egt + (base_egt + deg_factor * 20.0 + 2.10 * np.cos(phase) - self.egt) * 0.08
            self.vibration_rms = self.vibration_rms + (base_vib + deg_factor * 0.4 + 0.035 * np.sin(phase) - self.vibration_rms) * 0.08
            self.oil_pressure  = self.oil_pressure + (base_oil_p - deg_factor * 25.0 + 1.80 * np.cos(phase) - self.oil_pressure) * 0.08
            self.oil_temp      = self.oil_temp + (base_oil_temp + deg_factor * 12.0 + 0.45 * np.sin(phase) - self.oil_temp) * 0.08

        self.map_kpa  = round(max(35.0, 101.3 * density_ratio), 1)
        self.fuel_flow = round(max(4.0, (8.30 + (self.throttle - 0.72) * 6.0) * (self.map_kpa / 92.0) + noise(0.05)), 2)

        return {
            "rpm": round(self.rpm, 1), "cht": round(self.cht, 1), "egt": round(self.egt, 1),
            "oil_temp": round(self.oil_temp, 1), "oil_pressure": round(self.oil_pressure, 1),
            "fuel_flow": round(self.fuel_flow, 2), "vibration_rms": round(self.vibration_rms, 2),
            "map": round(self.map_kpa, 1), "afr": round(self.mixture, 2), "torque": 24.5,
            "crank_pos": float((self.cycle * 35) % 360), "coolant_temp": round(self.coolant_temp, 1),
            "altitude": round(self.altitude, 1), "ambient_temp": round(self.ambient_temp, 1),
            "air_density": self.air_density, "cooling_factor": self.cooling_factor,
        }


sim = SimulationController()


# Initialize AI Orchestrator with models
pinn_model = None
fault_classifier = None
drl_policy = None

# Attempt to load checkpoints if available
try:
    pinn_p = ROOT / "models" / "checkpoints" / "pinn_best.pt"
    if pinn_p.exists():
        from pinn.pinn_model import PINNRULModel
        data = torch.load(pinn_p, map_location="cpu", weights_only=False)
        pinn_model = PINNRULModel(in_features=12, hidden=64)
        pinn_model.load_state_dict(data["state_dict"])
        if "norm_min" in data and "norm_max" in data:
            pinn_model.set_norm_stats(data["norm_min"], data["norm_max"])
        pinn_model.eval()
        print("[Server] Loaded PINN checkpoint successfully")
except Exception as e:
    print(f"[Server] PINN load note: {e}")

try:
    cls_p = ROOT / "models" / "checkpoints" / "fault_classifier.pt"
    if cls_p.exists():
        from classifier.fault_classifier import FaultClassifierNet
        data = torch.load(cls_p, map_location="cpu", weights_only=False)
        fault_classifier = FaultClassifierNet(in_channels=12, num_classes=4, hidden=48)
        fault_classifier.load_state_dict(data["state_dict"])
        fault_classifier.eval()
        print("[Server] Loaded Fault Classifier checkpoint successfully")
except Exception as e:
    print(f"[Server] Classifier load note: {e}")

try:
    drl_p = ROOT / "models" / "checkpoints" / "drl_policy.pt"
    if drl_p.exists():
        from drl.drl_agent import ActorCritic
        data = torch.load(drl_p, map_location="cpu", weights_only=False)
        # state_dim=18: 14 sensor/PINN dims + 4-dim archetype one-hot
        loaded_dim = data.get("state_dim", 18)
        drl_policy = ActorCritic(state_dim=loaded_dim, action_dim=2, hidden_dim=64)
        drl_policy.load_state_dict(data["state_dict"])
        drl_policy.eval()
        print(f"[Server] Loaded DRL Policy checkpoint (state_dim={loaded_dim}) successfully")
except Exception as e:
    print(f"[Server] DRL load note: {e}")

# ---- Extract real benchmark metadata from checkpoints (single source of truth) ----
# These are the values stored by train_pinn.py and train_classifier.py at training time.
# /api/benchmarks serves these directly — no hardcoded numbers anywhere.
_ckpt_meta: Dict[str, Any] = {}

try:
    if pinn_p.exists():
        _d = torch.load(pinn_p, map_location="cpu", weights_only=False)
        _ckpt_meta["pinn_mae"]  = round(float(_d.get("test_mae",  float("nan"))), 2)
        _ckpt_meta["pinn_rmse"] = round(float(_d.get("test_rmse", float("nan"))), 2)
        _ckpt_meta["pinn_cfg"]  = _d.get("config", {})
except Exception as _e:
    print(f"[Server] Benchmark meta (PINN): {_e}")

try:
    _cnn_pca_p = ROOT / "models" / "checkpoints" / "cnn_pca_best.pt"
    if _cnn_pca_p.exists():
        _d = torch.load(_cnn_pca_p, map_location="cpu", weights_only=False)
        _ckpt_meta["cnn_pca_mae"]  = round(float(_d.get("test_mae",  float("nan"))), 2)
        _ckpt_meta["cnn_pca_rmse"] = round(float(_d.get("test_rmse", float("nan"))), 2)
except Exception as _e:
    print(f"[Server] Benchmark meta (CNN-PCA): {_e}")

try:
    _cnn_raw_p = ROOT / "models" / "checkpoints" / "cnn_raw_best.pt"
    if _cnn_raw_p.exists():
        _d = torch.load(_cnn_raw_p, map_location="cpu", weights_only=False)
        _ckpt_meta["cnn_raw_mae"]  = round(float(_d.get("test_mae",  float("nan"))), 2)
        _ckpt_meta["cnn_raw_rmse"] = round(float(_d.get("test_rmse", float("nan"))), 2)
except Exception as _e:
    print(f"[Server] Benchmark meta (CNN-Raw): {_e}")

try:
    if cls_p.exists():
        _d = torch.load(cls_p, map_location="cpu", weights_only=False)
        _ckpt_meta["clf_test_acc"] = round(float(_d.get("test_acc", float("nan"))) * 100, 1)
        _ckpt_meta["clf_val_acc"]  = round(float(_d.get("val_acc",  float("nan"))) * 100, 1)
except Exception as _e:
    print(f"[Server] Benchmark meta (Classifier): {_e}")

try:
    if drl_p.exists():
        _d = torch.load(drl_p, map_location="cpu", weights_only=False)
        _ckpt_meta["drl_avg_endurance"] = round(float(_d.get("avg_endurance", float("nan"))), 1)
except Exception as _e:
    print(f"[Server] Benchmark meta (DRL): {_e}")

print(f"[Server] Benchmark metadata loaded: {_ckpt_meta}")

from agent.orchestrator import DigitalTwinOrchestrator
orchestrator = DigitalTwinOrchestrator(
    pinn_model=pinn_model,
    fault_classifier=fault_classifier,
    drl_policy=drl_policy,
)


class FaultInjectionRequest(BaseModel):
    fault_type: Optional[str] = None  # "thermal_shock", "sensor_dropout", "oil_leak", "vibration_spike", or None


@app.get("/api/status")
async def get_system_status():
    """Returns runtime model status, system health, and benchmark scores."""
    return {
        "status": "ONLINE",
        "models": {
            "pinn_guardian": "LOADED" if pinn_model is not None else "ACTIVE_ANALYTICAL",
            "fault_classifier": "LOADED" if fault_classifier is not None else "ACTIVE_ANALYTICAL",
            "drl_strategist": "LOADED" if drl_policy is not None else "ACTIVE_ANALYTICAL",
            "langgraph_orchestrator": "ACTIVE",
        },
        "simulation": {
            "cycle": sim.cycle,
            "health": round(sim.health, 3),
            "active_fault": sim.injected_fault,
        },
        "self_correction_count": getattr(orchestrator, "_correction_count", 0),
    }


@app.get("/api/benchmarks")
async def get_benchmarks():
    """Returns comparative metrics sourced directly from trained checkpoint files.

    All numeric values trace back to fields written by train_pinn.py, train_classifier.py,
    and drl_agent.py at training time.  See /api/benchmarks/raw for the raw checkpoint
    metadata.  No hardcoded benchmark numbers are used anywhere in this endpoint.
    """
    pinn_mae     = _ckpt_meta.get("pinn_mae")
    pinn_rmse    = _ckpt_meta.get("pinn_rmse")
    cnn_pca_mae  = _ckpt_meta.get("cnn_pca_mae")
    cnn_raw_mae  = _ckpt_meta.get("cnn_raw_mae")
    clf_acc      = _ckpt_meta.get("clf_test_acc")
    drl_end      = _ckpt_meta.get("drl_avg_endurance")

    def _fmt(v, unit="", fallback="pending"):
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            return fallback
        return f"{v}{unit}"

    # MAE improvement vs CNN-PCA baseline (same dataset, apples-to-apples)
    mae_improvement = None
    if pinn_mae is not None and cnn_pca_mae is not None and cnn_pca_mae > 0:
        mae_improvement = round((cnn_pca_mae - pinn_mae) / cnn_pca_mae * 100, 1)

    return {
        "source": "checkpoint_metadata",
        "checkpoint_files": {
            "pinn":       str(ROOT / "models" / "checkpoints" / "pinn_best.pt"),
            "cnn_pca":    str(ROOT / "models" / "checkpoints" / "cnn_pca_best.pt"),
            "cnn_raw":    str(ROOT / "models" / "checkpoints" / "cnn_raw_best.pt"),
            "classifier": str(ROOT / "models" / "checkpoints" / "fault_classifier.pt"),
            "drl":        str(ROOT / "models" / "checkpoints" / "drl_policy.pt"),
        },
        "metrics": [
            {
                "metric": "RUL Prediction MAE (cycles)",
                "cnn_pca_baseline": _fmt(cnn_pca_mae, " cycles"),
                "cnn_raw_ablation": _fmt(cnn_raw_mae, " cycles"),
                "pinn_physics_informed": _fmt(pinn_mae, " cycles"),
                "pinn_rmse": _fmt(pinn_rmse, " cycles"),
                "mae_reduction_vs_baseline_pct": _fmt(mae_improvement, "%") if mae_improvement is not None else "pending",
                "note": "All three models trained on identical dataset split; see train_pinn.py ablation",
                "category": "Prognostics",
            },
            {
                "metric": "Fault Archetype Classification Accuracy",
                "traditional_dl": "Rule-based threshold (no ML)",
                "pinn_drl_stack": _fmt(clf_acc, "%"),
                "note": "4-class classifier (vibration_over / cht_over / egt_over / oil_starvation). "
                        "Trained on early-life windows only (first 60% of engine life) to avoid "
                        "per-archetype trajectory fingerprinting. see train_classifier.py.",
                "category": "Diagnostics",
            },
            {
                "metric": "DRL Mission Endurance (training)",
                "value": _fmt(drl_end, " cycles avg (last 20 episodes)"),
                "max_cycles": 300,
                "note": "Avg survival cycles during PPO training. Higher = agent learned to extend engine life.",
                "category": "Control",
            },
            {
                "metric": "Out-of-Distribution Safety",
                "description": "PINN Fourier thermal-balance loss enforces physical bounds on CHT predictions "
                               "even for unseen altitude/temperature combinations. Scripted fallback clearly "
                               "labelled DEMO_FALLBACK when trained model not loaded.",
                "category": "Reliability",
            },
            {
                "metric": "Edge Sensor Recovery",
                "description": "Sensor Auditor imputes NaN/out-of-range frames from rolling history buffer. "
                               "Tested against sensor_dropout fault injection.",
                "category": "Resilience",
            },
        ],
    }


@app.get("/api/benchmarks/raw")
async def get_benchmarks_raw():
    """Returns the raw checkpoint metadata dict for direct traceability verification."""
    return {"checkpoint_metadata": _ckpt_meta}


@app.post("/api/simulate/inject")
async def inject_fault(req: FaultInjectionRequest):
    """Trigger or clear a fault injection in the live telemetry stream."""
    sim.inject_fault(req.fault_type)
    if req.fault_type is None:
        orchestrator.reset_state()
    return {
        "success": True,
        "active_fault": sim.injected_fault,
        "message": f"Injected fault set to: {sim.injected_fault or 'CLEAR (NOMINAL)'}",
    }


@app.post("/api/simulate/environment")
async def update_environment(req: EnvironmentRequest):
    """Dynamically update flight altitude and ambient temperature in real time."""
    sim.set_environment(req.altitude, req.ambient_temp)
    return {
        "success": True,
        "altitude": sim.altitude,
        "ambient_temp": sim.ambient_temp,
        "air_density": sim.air_density,
        "cooling_factor": sim.cooling_factor,
        "message": f"Environment set to Alt: {sim.altitude}ft, Amb: {sim.ambient_temp}°C",
    }


@app.post("/api/simulate/reset")
async def reset_simulation():
    """Reset simulation to healthy cruise start."""
    sim.cycle = 0
    sim.health = 1.0
    sim.cht = 150.0
    sim.egt = 650.0
    sim.oil_pressure = 320.0
    sim.vibration_rms = 1.2
    sim.injected_fault = None
    orchestrator.reset_state()
    return {"success": True, "message": "Simulation reset"}


@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    """High-frequency (10 Hz) live telemetry and orchestrator dispatch stream."""
    await websocket.accept()
    print("[WebSocket] Client connected")

    try:
        while True:
            # 1. Step simulation
            raw_frame = sim.step()

            # 2. Process through LangGraph Agentic Loop (Auditor -> PINN -> Classifier -> DRL -> Dispatch)
            dispatch = orchestrator.process_telemetry_frame(raw_frame, cycle=sim.cycle)
            dispatch["active_injected_fault"] = sim.injected_fault

            # 3. Apply DRL policy feedback to simulator if shield or action active
            drl_act = dispatch.get("drl_action", {})
            if drl_act.get("shield_applied", False):
                sim.throttle = float(np.clip(sim.throttle + drl_act.get("delta_throttle", 0.0), 0.50, 0.90))
                sim.mixture = float(np.clip(sim.mixture + drl_act.get("delta_mixture", 0.0) * 1.5, 12.0, 15.0))

            # 4. Stream JSON payload to client
            await websocket.send_text(json.dumps(dispatch))
            await asyncio.sleep(0.1)  # 10 Hz rate

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        import traceback
        print(f"[WebSocket] Error: {e}")
        traceback.print_exc()


# Mount frontend static directory if exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(
                content=index_file.read_text(encoding="utf-8"),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        return HTMLResponse("<h3>Dashboard frontend directory found, index.html not found.</h3>")

    @app.get("/{file_name:path}")
    async def serve_static_root(file_name: str):
        target = (FRONTEND_DIR / file_name).resolve()
        if FRONTEND_DIR in target.parents and target.exists() and target.is_file():
            media_type = None
            if file_name.endswith(".css"):
                media_type = "text/css"
            elif file_name.endswith(".js"):
                media_type = "application/javascript"
            elif file_name.endswith(".png"):
                media_type = "image/png"
            elif file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
                media_type = "image/jpeg"
            return FileResponse(
                target,
                media_type=media_type,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        return JSONResponse(status_code=404, content={"detail": "File not found"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
FRONTEND_DIR = ROOT / "frontend"

from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from auth.routes import router as auth_router, get_current_user
from auth.security import COOKIE_NAME, decode_session_token
from db.models import User, Mission, FaultEvent
from db.session import get_session, get_sessionmaker

app = FastAPI(title="PRAHARI — SIH26054 MALE UAV Engine Digital Twin Server", version="1.0.0")
app.include_router(auth_router)

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


def _presim(fault_type: Optional[str], total_cycles: int, seed: int = 7,
            initial_health: float = 0.92, degradation_driver: float = 1.05) -> tuple:
    """Run EngineSim for one fault archetype and return (data_dict, truth_dict).

    EngineSim's health trajectory is *always* a run-to-failure curve that
    reaches health~0 at the end of `total_cycles`, by design (it's meant to
    generate RUL training data). For the injected fault archetypes that's
    exactly what we want — the fault should develop into a real failure
    within a demo-length window. But the nominal/no-fault baseline reused
    this same short 1200-cycle (120s @ 10Hz) horizon, so "NOMINAL CRUISE"
    silently degraded into a critical-looking state every ~2 minutes and
    wrapped back to healthy — see the caller for the actual fix (a much
    longer, near-1.0-health nominal trajectory).
    """
    try:
        from sim.engine_sim import EngineSim, MissionProfile
        bias = _FAULT_ARCHETYPES.get(fault_type, {})
        sim  = EngineSim(
            total_cycles=total_cycles,
            seed=seed,
            profile=MissionProfile.cruise_dominant(total_cycles),
            initial_health=initial_health,
            degradation_driver=degradation_driver,
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

    _TOTAL_CYCLES = 1200          # pre-simulated trajectory length for injected fault archetypes

    # Fault modes that do NOT use an EngineSim archetype trajectory as their
    # physics base — nominal cruise, and the two sensor-level faults that are
    # NaN/bias overlays rather than a mechanical failure archetype. All three
    # instead use the hand-scripted steady-cruise physics in _step_scripted(),
    # which is calibrated to sit at the app's advertised nominal baseline
    # (~148.6°C CHT, ~1.41g vibration) and decays health extremely slowly.
    #
    # Why not just precompute a "None" EngineSim trajectory like the other
    # archetypes? EngineSim's MissionProfile.cruise_dominant() includes
    # climb/descent power segments by design (it's meant to generate varied
    # RUL training data) — even at initial_health=1.0 its CHT swings well
    # into caution/critical territory (measured: 183-217°C across a 6000-step
    # run), which reads as a false alarm under a button literally labelled
    # "NOMINAL CRUISE". The hand-scripted path is the one actually designed
    # to hold steady.
    _NON_ARCHETYPE_FAULTS = (None, "sensor_dropout", "sensor_drift")

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

        # ---- Pre-compute trajectories for the mechanical fault archetypes ----
        print("[LiveEngineSim] Pre-computing fault trajectories (this takes ~2–5 s) …")
        self._trajs: Dict[Optional[str], tuple] = {}
        for fault in list(_FAULT_ARCHETYPES.keys()):
            if fault in self._NON_ARCHETYPE_FAULTS:
                continue
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
        self.throttle     = 0.72   # undo any DRL shield de-rate from a prior fault
        self.mixture      = 13.8
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

        # ---- Steady hand-scripted path: nominal cruise + sensor-level faults --
        if self.injected_fault in self._NON_ARCHETYPE_FAULTS or not self._using_enginesim:
            return self._step_scripted()

        # ---- EngineSim path: mechanical fault archetypes ---------------------
        return self._step_enginesim()

    def _step_enginesim(self) -> Dict[str, float]:
        """Advance one cycle using the pre-computed EngineSim trajectory."""
        from sim.engine_sim import SENSOR_ORDER

        # This path only runs for the mechanical archetype faults (step()
        # routes nominal/sensor_dropout/sensor_drift to _step_scripted()
        # instead), so the trajectory is always cached under the exact key.
        data, truth = self._trajs[self.injected_fault]

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

        # sensor_dropout/sensor_drift are handled entirely in _step_scripted()
        # (see step()'s dispatch) — this path only ever runs for the genuine
        # mechanical archetype faults, so no NaN/drift overlay is needed here.

        return {
            "rpm":           round(self.rpm, 1),
            "cht":           round(self.cht, 1),
            "egt":           round(self.egt, 1),
            "oil_temp":      round(self.oil_temp, 1),
            "oil_pressure":  round(self.oil_pressure, 1),
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
        elif self.injected_fault == "sensor_drift":
            drift_frac = min(1.0, self.cycle / 400.0)
            self.cht         = self.cht + drift_frac * 0.1
            self.oil_pressure = self.oil_pressure - drift_frac * 0.25
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

fault_classifiers = []  # Layer 2 ensemble: every independently-seeded checkpoint found
try:
    cls_p = ROOT / "models" / "checkpoints" / "fault_classifier.pt"
    from classifier.fault_classifier import FaultClassifierNet
    ensemble_paths = sorted((ROOT / "models" / "checkpoints").glob("fault_classifier*.pt"))
    for p in ensemble_paths:
        try:
            data = torch.load(p, map_location="cpu", weights_only=False)
            m = FaultClassifierNet(in_channels=12, num_classes=4, hidden=48)
            m.load_state_dict(data["state_dict"])
            m.eval()
            fault_classifiers.append(m)
            if p == cls_p:
                fault_classifier = m
        except Exception as _me:
            print(f"[Server] Classifier ensemble member {p.name} load note: {_me}")
    if fault_classifiers and fault_classifier is None:
        fault_classifier = fault_classifiers[0]
    print(f"[Server] Loaded {len(fault_classifiers)} Fault Classifier ensemble checkpoint(s): "
          f"{[p.name for p in ensemble_paths]}")
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
from agent.defense_layers import sign_packet, verify_packet
orchestrator = DigitalTwinOrchestrator(
    pinn_model=pinn_model,
    fault_classifier=fault_classifier,
    fault_classifiers=fault_classifiers,
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


# --------------------------------------------------------------------------- #
# Dashboard API — mission history + summary stats from Postgres              #
# --------------------------------------------------------------------------- #
from auth.routes import require_user  # noqa: E402  (after app/router setup above)


_ensemble_acc_cache: Optional[float] = None


@app.get("/api/dashboard/summary")
async def dashboard_summary(user: User = Depends(require_user), db: Session = Depends(get_session)):
    missions = db.query(Mission).all()
    total_missions = len(missions)
    total_seconds = sum((m.frame_count or 0) * 0.1 for m in missions)
    critical_events = db.query(FaultEvent).filter(FaultEvent.severity == "CRITICAL").count()

    global _ensemble_acc_cache
    ensemble_acc = _ensemble_acc_cache
    if ensemble_acc is None:
        try:
            from eval.benchmarks import evaluate_classifier_ensemble
            # Runs the ensemble against ~7k held-out windows once per server
            # process — slow enough that it must not block the 10 Hz WS loop,
            # and unnecessary to redo per request since checkpoints don't
            # change during a running server session.
            result = await asyncio.to_thread(evaluate_classifier_ensemble)
            if result.get("ensemble_accuracy") is not None:
                ensemble_acc = round(result["ensemble_accuracy"] * 100, 1)
                _ensemble_acc_cache = ensemble_acc
        except Exception:
            pass  # falls back to self-reported clf_test_acc below

    benchmarks = dict(_ckpt_meta)
    if ensemble_acc is not None:
        benchmarks["clf_ensemble_accuracy"] = ensemble_acc

    return {
        "total_missions": total_missions,
        "total_telemetry_seconds": total_seconds,
        "critical_events": critical_events,
        "benchmarks": benchmarks,
    }


@app.get("/api/dashboard/missions")
async def dashboard_missions(limit: int = 25, user: User = Depends(require_user), db: Session = Depends(get_session)):
    missions = (
        db.query(Mission)
        .order_by(Mission.started_at.desc())
        .limit(min(limit, 100))
        .all()
    )
    out = []
    for m in missions:
        duration = None
        if m.ended_at:
            duration = (m.ended_at - m.started_at).total_seconds()
        out.append({
            "id": m.id,
            "started_at": m.started_at.isoformat() if m.started_at else None,
            "ended_at": m.ended_at.isoformat() if m.ended_at else None,
            "duration_seconds": duration,
            "frame_count": m.frame_count,
            "final_fault_archetype": m.final_fault_archetype,
            "final_rul_cycles": m.final_rul_cycles,
            "had_critical_event": m.had_critical_event,
            "event_count": len(m.events),
        })
    return out


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


# --------------------------------------------------------------------------- #
# Mission/event persistence — one Mission row per WS connection, one          #
# FaultEvent row per fault/severity state transition. Every DB call runs in   #
# a thread (asyncio.to_thread) so a slow/unavailable Postgres never stalls    #
# the 10 Hz telemetry loop, and every call is wrapped so a DB error is        #
# logged but never breaks the live stream.                                    #
# --------------------------------------------------------------------------- #
def _create_mission(user_id: Optional[str]) -> Optional[str]:
    try:
        SessionLocal = get_sessionmaker()
        with SessionLocal() as db:
            mission = Mission(user_id=user_id)
            db.add(mission)
            db.commit()
            return mission.id
    except Exception as e:
        print(f"[Persist] Could not create mission row: {e}")
        return None


def _log_fault_event(mission_id: Optional[str], fault: str, severity: str, confidence: float, detail: str = ""):
    if not mission_id:
        return
    try:
        SessionLocal = get_sessionmaker()
        with SessionLocal() as db:
            db.add(FaultEvent(
                mission_id=mission_id, fault_archetype=fault,
                severity=severity, confidence=confidence, detail=detail,
            ))
            db.commit()
    except Exception as e:
        print(f"[Persist] Could not log fault event: {e}")


def _close_mission(mission_id: Optional[str], frame_count: int, final_fault: str,
                    final_rul: Optional[float], had_critical: bool):
    if not mission_id:
        return
    try:
        SessionLocal = get_sessionmaker()
        with SessionLocal() as db:
            mission = db.get(Mission, mission_id)
            if mission:
                mission.ended_at = datetime.now(timezone.utc)
                mission.frame_count = frame_count
                mission.final_fault_archetype = final_fault
                mission.final_rul_cycles = final_rul
                mission.had_critical_event = had_critical
                db.commit()
    except Exception as e:
        print(f"[Persist] Could not close mission row: {e}")


@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    """High-frequency (10 Hz) live telemetry and orchestrator dispatch stream.

    Auth-gated on the same session cookie the HTML pages use — WebSocket
    doesn't go through FastAPI's Depends() for cookies the way HTTP routes
    do, so the token is decoded directly from websocket.cookies.
    """
    token = websocket.cookies.get(COOKIE_NAME)
    session_payload = decode_session_token(token) if token else None
    if not session_payload:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    print(f"[WebSocket] Client connected (user={session_payload.get('email')})")

    mission_id = await asyncio.to_thread(_create_mission, session_payload.get("sub"))
    last_fault, last_severity = "nominal", "NOMINAL"
    frame_count, had_critical = 0, False
    final_fault, final_rul = "nominal", None

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

            # 4. Layer 3 — sign the outgoing packet (HMAC-SHA256) and self-verify
            #    before it leaves the server, demonstrating tamper/replay rejection
            #    on the transport boundary without needing a second trust domain.
            body_json = json.dumps(dispatch, sort_keys=True)
            send_ts = time.time()
            signature = sign_packet(body_json)
            dispatch["integrity"] = {
                "signature": signature,
                "timestamp": send_ts,
                "verified": verify_packet(body_json, signature, send_ts),
            }

            # 5. Persist a FaultEvent row whenever the fault archetype or its
            #    severity changes — a server-side mirror of telemetry_store.js's
            #    EventLog.checkTransitions, but durable across page refreshes.
            frame_count += 1
            fault = dispatch.get("fault_archetype", "nominal")
            confidence = dispatch.get("fault_confidence", 0.0)
            severity = "CRITICAL" if (fault != "nominal" and confidence > 0.8) else \
                       ("MODERATE" if fault != "nominal" else "NOMINAL")
            final_fault = fault
            final_rul = dispatch.get("adjusted_rul", dispatch.get("rul_cycles"))
            if severity == "CRITICAL":
                had_critical = True
            if fault != last_fault or severity != last_severity:
                asyncio.create_task(asyncio.to_thread(
                    _log_fault_event, mission_id, fault, severity, confidence,
                    dispatch.get("drl_action", {}).get("recommendation", ""),
                ))
                last_fault, last_severity = fault, severity

            # 6. Stream JSON payload to client
            await websocket.send_text(json.dumps(dispatch))
            await asyncio.sleep(0.1)  # 10 Hz rate

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        import traceback
        print(f"[WebSocket] Error: {e}")
        traceback.print_exc()
    finally:
        await asyncio.to_thread(_close_mission, mission_id, frame_count, final_fault, final_rul, had_critical)


# Mount frontend static directory if exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    def _serve_page(name: str) -> HTMLResponse:
        page_file = FRONTEND_DIR / name
        if page_file.exists():
            return HTMLResponse(
                content=page_file.read_text(encoding="utf-8"),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        return HTMLResponse(f"<h3>{name} not found.</h3>", status_code=404)

    @app.get("/")
    async def index(user: Optional[User] = Depends(get_current_user)):
        return RedirectResponse("/dashboard" if user else "/login")

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(user: Optional[User] = Depends(get_current_user)):
        if user:
            return RedirectResponse("/dashboard")
        return _serve_page("login.html")

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page(user: Optional[User] = Depends(get_current_user)):
        if not user:
            return RedirectResponse("/login")
        return _serve_page("dashboard.html")

    @app.get("/app", response_class=HTMLResponse)
    async def live_console_page(user: Optional[User] = Depends(get_current_user)):
        if not user:
            return RedirectResponse("/login")
        return _serve_page("index.html")

    @app.get("/{file_name:path}")
    async def serve_static_root(file_name: str):
        # .html pages are only ever served through the explicit, auth-gated
        # routes above (/login, /dashboard, /app) — block direct access to
        # the raw files here so those gates can't be bypassed by path.
        if file_name.endswith(".html"):
            return JSONResponse(status_code=404, content={"detail": "File not found"})
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

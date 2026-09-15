"""Defense-grade layered architecture for SIH26054 (SAE ARP4754A-style redundancy).

Real UCAV/MALE-UAV avionics (MQ-9, TB2, Wing Loong, Heron TP) achieve >90%
mission-capable rates not through exotic AI but disciplined redundancy: fused
sensor estimates, voted computation, authenticated links, and fail-safe
gating (see Executive Summary.pdf, "Fault-tolerant flight controls").  This
module ports that same discipline onto the PINN/classifier/DRL pipeline:

  Layer 1 - Sensor fusion          : fuse_sensor_and_physics()
  Layer 2 - Ensemble voting        : vote_fault_classification()
  Layer 3 - Comms integrity (HMAC) : sign_packet() / verify_packet()
  Layer 4 - Safe-mode gate         : decide_action_mode()
  Layer 5 - Trend / PHM risk       : compute_trend_risk()

Each function is independent and pure (no hidden state) so it can be unit
tested directly - see scripts/test_fault_injection.py.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Layer 1 - Sensor Fusion (confidence-weighted, EKF-style)
# --------------------------------------------------------------------------- #

def fuse_sensor_and_physics(
    raw_sensor_value: float,
    pinn_estimate: float,
    sensor_confidence: float,
    pinn_confidence: float,
) -> float:
    """Confidence-weighted fusion of a raw sensor reading and the PINN's
    physics-consistent estimate for the same channel.

    If the raw sensor is glitching (low sensor_confidence, e.g. from an
    imputation/corruption flag or a failed integrity check), the fused
    estimate leans on the physics estimate instead of a bad reading -
    exactly the technique real aircraft engine health monitoring uses.
    """
    total_confidence = sensor_confidence + pinn_confidence
    if total_confidence <= 1e-9:
        return pinn_estimate
    return (raw_sensor_value * sensor_confidence + pinn_estimate * pinn_confidence) / total_confidence


def sensor_confidence_from_audit(field: str, audit_results: Dict[str, Any], integrity_ok: bool) -> float:
    """Derives a [0, 1] trust weight for one telemetry field from Layer 3's
    sensor-auditor output plus the comms-integrity check (Layer 3)."""
    confidence = 1.0
    if field in (audit_results.get("corrupted_fields") or []):
        confidence *= 0.15
    if field in (audit_results.get("imputed_fields") or {}):
        confidence *= 0.35
    if not integrity_ok:
        confidence *= 0.5
    return max(0.05, min(1.0, confidence))


def pinn_confidence_from_residual(physics_residual: float, threshold: float = 2.5) -> float:
    """Inverse-scales the PINN's physics-consistency residual into a [0.1, 1]
    trust weight - a low residual (physically consistent) yields high trust."""
    ratio = max(0.0, physics_residual) / max(threshold, 1e-6)
    return max(0.1, min(1.0, 1.0 - min(1.0, ratio)))


# --------------------------------------------------------------------------- #
# Layer 2 - Ensemble Voting (2-of-3 / N-of-M fault-tolerant computation)
# --------------------------------------------------------------------------- #

def vote_fault_classification(predictions: Sequence[str]) -> Tuple[str, float]:
    """Majority-votes a list of independent classifier predictions for the
    same telemetry window.

    Returns (majority_result, agreement_score) where agreement_score is the
    fraction of members that agreed with the majority - the "escalate to
    human" signal when redundant models disagree (real triple-redundant
    flight computers isolate the outlier the same way).
    """
    if not predictions:
        return "nominal", 0.0
    counts = Counter(predictions)
    majority_result, majority_count = counts.most_common(1)[0]
    agreement_score = majority_count / len(predictions)
    return majority_result, agreement_score


# --------------------------------------------------------------------------- #
# Layer 3 - Communications / Data Integrity (HMAC-authenticated telemetry)
# --------------------------------------------------------------------------- #

_DEFAULT_SECRET = "sih26054-dev-telemetry-key-change-in-prod"
_HMAC_SECRET = os.environ.get("TELEMETRY_HMAC_SECRET", _DEFAULT_SECRET).encode("utf-8")


def sign_packet(payload: str) -> str:
    """Generates an HMAC-SHA256 signature for an outgoing telemetry packet."""
    return hmac.new(_HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_packet(payload: str, signature: str, timestamp: float, max_age_seconds: float = 2.0) -> bool:
    """Rejects a packet if the signature doesn't match (tampered) or if it's
    older than max_age_seconds (replayed/delayed - a common spoofing tactic)."""
    expected_signature = sign_packet(payload)
    signature_ok = hmac.compare_digest(expected_signature, signature)
    fresh_enough = (time.time() - timestamp) <= max_age_seconds
    return signature_ok and fresh_enough


# --------------------------------------------------------------------------- #
# Layer 4 - Autonomy / Fail-Safe Logic (safe-mode gate)
# --------------------------------------------------------------------------- #

def decide_action_mode(
    agreement_score: float,
    integrity_ok: bool,
    confidence: float,
    threshold: float = 0.8,
) -> str:
    """Gate for autonomous DRL action. Real fail-safe avionics call this a
    'safe state fallback' - never act on uncertain or unverified inputs.

    Returns "AUTONOMOUS_ACTION" only when the ensemble agrees (Layer 2), the
    telemetry passed its integrity check (Layer 3), AND classifier confidence
    clears the threshold; otherwise "SAFE_MODE_ESCALATE_TO_HUMAN".
    """
    if agreement_score >= threshold and integrity_ok and confidence >= threshold:
        return "AUTONOMOUS_ACTION"
    return "SAFE_MODE_ESCALATE_TO_HUMAN"


# --------------------------------------------------------------------------- #
# Layer 5 - Prognostics & Health Management (rolling-trend risk)
# --------------------------------------------------------------------------- #

_TREND_FIELDS = {
    "egt": "rising",
    "vibration_rms": "rising",
    "cht": "rising",
    "oil_pressure": "falling",
}


def compute_trend_risk(history: List[Dict[str, float]], window: int = 20) -> float:
    """Slope-checks the last `window` frames of the rolling telemetry buffer
    (already collected by the sensor auditor) and returns the fraction of
    monitored channels trending in a worsening direction together -
    condition-based-maintenance's core signal: compounding risk across
    multiple sensors is a stronger indicator than any single reading.
    """
    if len(history) < window:
        return 0.0

    recent = history[-window:]
    x = np.arange(window, dtype=np.float64)
    worsening = 0
    monitored = 0

    for field, direction in _TREND_FIELDS.items():
        values = np.array([frame.get(field, np.nan) for frame in recent], dtype=np.float64)
        if np.any(np.isnan(values)):
            continue
        monitored += 1
        slope = float(np.polyfit(x, values, 1)[0])
        if (direction == "rising" and slope > 0) or (direction == "falling" and slope < 0):
            worsening += 1

    return worsening / max(monitored, 1)

"""GLM (Zhipu AI) LLM advisory layer — the "orchestrator control" piece the
user asked for: an LLM that reads the current live telemetry/defense-layer
state and can decide to trigger one of the intent-routed diagnostic graph's
branches (src/agent/orchestrator.py::run_diagnostic), not just narrate.

Important, stated plainly rather than left implicit: this layer does NOT and
cannot change the fault classifier's trained accuracy — that's a property of
the model weights (see MODEL_CARD.md "Known limitations" #1, and the
separate background task fixing the training-data leakage). What this layer
adds is a reasoning/arbitration capability on top of the existing smart
layers (ensemble vote + agreement, PINN physics residual, trend/PHM risk,
sensor fusion confidence) — grounded strictly in their real computed values,
never inventing sensor readings or diagnoses of its own.

Hard rate-limited by design (explicit user requirement: "really really
limited in number") — a fixed-window counter, no per-user tracking needed
since this is a single-operator ground station.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

GLM_API_KEY = os.environ.get("GLM_API_KEY")
GLM_API_BASE = os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4.5-flash")

VALID_INTENTS = ("sensor_integrity_check", "thermal_stress_analysis", "root_cause_diagnostic", "none")

# --------------------------------------------------------------------------- #
# Rate limiter — fixed-window, in-process. Deliberately simple and strict.
# --------------------------------------------------------------------------- #
_MAX_CALLS_PER_WINDOW = int(os.environ.get("GLM_MAX_CALLS_PER_WINDOW", "8"))
_WINDOW_SECONDS = int(os.environ.get("GLM_WINDOW_SECONDS", "3600"))  # 1 hour
_call_timestamps: list[float] = []


def rate_limit_status() -> Dict[str, Any]:
    now = time.time()
    active = [t for t in _call_timestamps if now - t < _WINDOW_SECONDS]
    _call_timestamps[:] = active
    remaining = max(0, _MAX_CALLS_PER_WINDOW - len(active))
    reset_in = 0.0
    if active and remaining == 0:
        reset_in = _WINDOW_SECONDS - (now - min(active))
    return {"remaining": remaining, "limit": _MAX_CALLS_PER_WINDOW, "reset_in_seconds": round(max(0.0, reset_in), 1)}


def _consume_rate_limit() -> bool:
    status = rate_limit_status()
    if status["remaining"] <= 0:
        return False
    _call_timestamps.append(time.time())
    return True


# --------------------------------------------------------------------------- #
# GLM client
# --------------------------------------------------------------------------- #

def _call_glm(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
    if not GLM_API_KEY:
        raise RuntimeError("GLM_API_KEY is not set — copy .env.example to .env and fill it in.")

    resp = requests.post(
        f"{GLM_API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {GLM_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},  # keep latency/cost down — no chain-of-thought tokens
            "temperature": 0.2,
        },
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


_SYSTEM_PROMPT = """You are PRAHARI's diagnostic advisor for a MALE UAV aero-engine digital twin.

You will be given the CURRENT live telemetry/defense-layer state as JSON. You must:
1. Reason ONLY from the provided JSON — never invent sensor values, fault types, or numbers not present in it.
2. Write a short (2-4 sentence) plain-English synthesis of the current situation for a ground-station operator.
3. Decide whether one of the intent-routed diagnostic branches should run next, and which one:
   - "sensor_integrity_check" — if telemetry integrity looks questionable
   - "thermal_stress_analysis" — if CHT/EGT/physics-residual signals look concerning
   - "root_cause_diagnostic" — if a fault archetype other than nominal is active and needs classification/mitigation review
   - "none" — if the state is nominal and no further diagnostic is warranted
4. Respond with STRICT JSON only, no markdown, no prose outside the JSON, in exactly this shape:
   {"synthesis": "...", "recommended_intent": "sensor_integrity_check|thermal_stress_analysis|root_cause_diagnostic|none", "reasoning": "one sentence"}
"""


def advise(state_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Rate-limited. Raises RuntimeError('rate_limited') if the window is exhausted."""
    if not _consume_rate_limit():
        raise RuntimeError("rate_limited")

    user_prompt = "Current PRAHARI state:\n" + json.dumps(state_snapshot, default=str)
    raw = _call_glm(_SYSTEM_PROMPT, user_prompt)

    try:
        # Strip accidental markdown code fences if the model adds them anyway.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {"synthesis": raw.strip(), "recommended_intent": "none", "reasoning": "(response was not strict JSON — shown as-is)"}

    intent = parsed.get("recommended_intent", "none")
    if intent not in VALID_INTENTS:
        intent = "none"
    return {
        "synthesis": parsed.get("synthesis", ""),
        "recommended_intent": intent,
        "reasoning": parsed.get("reasoning", ""),
    }

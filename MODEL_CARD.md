# Model Card — SIH26054 Aero-Engine Digital Twin

This card documents every threshold, label, and defense-layer behavior in the
live pipeline (`src/agent/orchestrator.py` + `src/agent/defense_layers.py`),
in the spirit real UAV maintenance crews expect: every alert traces back to a
specific threshold and sensor reading a technician can verify (see
`Executive Summary.pdf`, "Maintenance Documentation" / real UCAV MTBF
practice). Numbers below are sourced live from `/api/benchmarks` and
`python -m src.eval.benchmarks` — never hand-typed.

## 1. Operational limits (Rotax 914 F)

Single source of truth, mirrored between `frontend/telemetry_store.js`
(`LIMITS`) and `src/agent/orchestrator.py` (`NOMINAL_LIMITS`):

| Channel | Caution | Critical | Unit |
|---|---|---|---|
| CHT | 185 | 210 | °C |
| EGT | 740 | 800 | °C |
| Vibration RMS | 2.5 | 3.5 | g |
| Oil Pressure (low) | 220 | 180 | kPa |
| Oil Temp | 120 | 140 | °C |
| RPM | 5,600 | 5,800 | RPM |

## 2. Fault archetypes

| Label | Driven by | Description |
|---|---|---|
| `cht_over` | CHT | Cooling boundary-layer breakdown / thermal stress |
| `egt_over` | EGT | Abnormal combustion, lean-mixture thermal runaway |
| `oil_starvation` | Oil pressure | Lubrication circuit pressure decay |
| `vibration_over` | Vibration RMS | Crankshaft/bearing mechanical degradation |
| `nominal` | — | No channel anomalous |

## 3. Defense-grade layered architecture

Each layer is a pure function in `src/agent/defense_layers.py`, unit-tested
in `scripts/test_fault_injection.py`.

- **Layer 1 — Sensor fusion** (`fuse_sensor_and_physics`): CHT/EGT are fused
  from the raw audited reading and a short-horizon trend-continuity estimate
  (process-model prior), confidence-weighted by (a) whether the sensor auditor
  flagged the field corrupted/imputed and (b) the PINN's physics-consistency
  residual. A glitching sensor is discounted, not trusted.
- **Layer 2 — Ensemble voting** (`vote_fault_classification`): every loaded
  `fault_classifier*.pt` checkpoint votes independently; the majority wins and
  `fault_agreement_score` (fraction agreeing) becomes the escalation signal.
  Degrades gracefully to single-model behavior with 0-1 checkpoints loaded.
- **Layer 3 — Comms integrity** (`sign_packet`/`verify_packet`): every
  outgoing `/ws/telemetry` frame is HMAC-SHA256 signed
  (`TELEMETRY_HMAC_SECRET` env var) and self-verified with a 2s freshness
  window before dispatch; the result rides in `payload.integrity`.
- **Layer 4 — Safe-mode gate** (`decide_action_mode`): the DRL's autonomous
  action only proceeds when ensemble agreement ≥0.8 **and** the sensor audit
  passed **and** classifier confidence ≥0.8; otherwise the action mode flips
  to `SAFE_MODE_ESCALATE_TO_HUMAN` and the last known-good throttle/mixture
  setting is held instead of guessing.
- **Layer 5 — Trend/PHM risk** (`compute_trend_risk`): slope-checks EGT,
  vibration RMS, CHT, and oil pressure over the last 20 audited frames;
  `trend_risk_score` is the fraction trending the wrong way *together* —
  catches slow degradation a single-frame threshold would miss.
- **Layer 6 — Fault-injection harness**: `scripts/test_fault_injection.py`
  (18 tests) exercises all defense-layer functions in isolation plus the full
  pipeline under every injectable fault, including the new `sensor_drift`
  slow-drift scenario (a gradually-ramping ~40s calibration bias, distinct
  from the instantaneous `sensor_dropout` NaN injection).
- **Layer 7 — This document**, plus `src/eval/benchmarks.py` for the numbers.

## 4. Benchmark numbers (live, `python -m src.eval.benchmarks`)

| Metric | Value | Source |
|---|---|---|
| PINN test MAE / RMSE | 37.07 / 66.95 cycles | `models/checkpoints/pinn_best.pt` |
| Classifier self-reported test acc | 100.0% | `models/checkpoints/fault_classifier.pt` (see limitation below) |
| Classifier **independently re-verified** test acc (each of 3 seeds) | **46.7%** | `src/eval/benchmarks.py::evaluate_classifier_ensemble()` |
| Ensemble (3 members) majority-vote accuracy | 46.7% | same |
| Ensemble mean agreement | 100.0% | same — all 3 independently-seeded members agree with *each other* on every window, not with ground truth: strong evidence they converged to the same leakage shortcut rather than 3 diverse hypotheses |
| DRL avg endurance extension | 7.2 cycles | `models/checkpoints/drl_policy.pt` |

## 5. Known limitations

1. **Classifier train/test leakage.** `data/processed/clf_early_life_mask.npy`
   is missing from the current processed dataset, so `fault_classifier.pt`'s
   self-reported 100% test accuracy was computed without the early-life
   restriction `scripts/train_classifier.py` warns about — the model likely
   fingerprints per-engine identity rather than genuine fault signature.
   Independently re-evaluating the same checkpoint against its own held-out
   split via `src/eval/benchmarks.py` (which reconstructs labels the same way
   `classifier/fault_classifier.py::load_dataset()` does) measures **46.7%**
   accuracy for all 3 independently-seeded checkpoints — and they agree with
   *each other* 100% of the time while each being wrong on the same ~53% of
   windows, which is strong evidence all 3 converged to the same leakage
   shortcut (most likely per-engine identity fingerprinting) rather than 3
   genuinely diverse hypotheses. Ensemble voting cannot rescue this: N copies
   of the same systematic error still vote for the same wrong answer. This is
   why `scripts/test_fault_injection.py`'s integration tests assert on the
   *physical channel* (CHT/oil-pressure/vibration excursions) rather than the
   classifier's specific archetype label: the sensor/fusion/PHM pipeline is
   verified correct, but the classifier's label should not yet be trusted for
   root-cause attribution. **Fix**: re-run
   `python scripts/build_dataset.py --early-life-frac 0.60` then retrain all
   ensemble members (tracked as a follow-up task).
2. **Ensemble size.** All 3 intended independently-seeded members are now
   trained (`fault_classifier.pt`, `fault_classifier_seed1.pt`,
   `fault_classifier_seed2.pt`). `agreement_score` and the safe-mode gate
   both degrade gracefully with fewer members if any checkpoint is later
   removed or fails to load.
3. **Layer 3 trust boundary.** The HMAC signature is generated and verified
   server-side in the same process (no separate untrusted network hop exists
   in this single-machine demo), so it demonstrates the *mechanism*
   end-to-end rather than defending a real network boundary. A production
   deployment would sign at the edge device and verify at the ground-station
   ingest point, with the key never present on the untrusted link.
4. **PINN fallback labeling.** When no trained PINN model is loaded (or the
   inference buffer is too short), `pinn_results.fourier_law_adherence` is
   explicitly labeled `DEMO_FALLBACK` — never call the scripted Arrhenius
   approximation "physically validated by Fourier's law."

## 6. Scope & Ethics — Mission Map content boundary

The Mission Map (`/mission-map`) visualizes a UAV patrol trajectory, DRDO
laboratory locations, ground-station markers, and a simulated
sensor/EW-denial fault encounter, plus a Regional Globe view for geographic
context. Deliberately **out of scope**, by design decision rather than
oversight:

- **No cross-border strike depiction.** SIH26054's problem statement is
  aero-engine health monitoring and predictive maintenance — not weapons
  targeting. Rendering realistic attacks against named neighboring countries
  doesn't serve that problem statement and risks reading as inappropriate for
  a student hackathon submission being judged by DRDO-adjacent evaluators.
- **No missile-compatibility claims.** PRAHARI is a health/prognostics
  digital twin for a MALE UAV's Rotax 914F piston engine; it makes no claims
  about weapons payload integration of any kind.
- **Neighboring-country markers are geographic reference only.** The Regional
  Globe view labels a handful of neighboring capitals (Islamabad, Kathmandu,
  Dhaka, Colombo, Naypyidaw, Lhasa) purely for spatial context — plain dots,
  no connecting lines, no activity or status implied, the same treatment an
  atlas gives any city label.
- **All trajectory/attack/ground-station content is fictional**, explicitly
  disclaimed in-page (`frontend/mission-map.html`) — only the India boundary
  geodata and the DRDO laboratory city locations are real, public, sourced
  data (see `scripts/build_india_map.py` and
  `frontend/assets/mission_map_data.js`).

This boundary was held across multiple follow-up requests during development
to extend the map toward cross-border strike scenarios and missile
compatibility — declined for the reasons above rather than silently
implemented.

## 7. Intent-Routed Diagnostic Agent & Hash-Chain Audit Log

`src/agent/orchestrator.py::run_diagnostic()` is a **second**, separate
LangGraph `StateGraph` from the continuous 10 Hz pipeline — it uses genuine
conditional routing (`add_conditional_edges`/`set_conditional_entry_point`,
which the continuous graph never needs) to dispatch one of three ground-
station-triggered intents (`sensor_integrity_check`, `thermal_stress_analysis`,
`root_cause_diagnostic`) to the matching existing tool nodes, gated by a
physical-feasibility check (Fourier conservation-of-energy) or a safety-shield
check (thermal/mechanical redline). It reuses the exact same node methods as
the continuous graph — no duplicated math — and never runs per-frame; it's
invoked on demand via `POST /api/agent/diagnose` from the Live Ops Console's
"AI Diagnostic Router" panel.

**Hash-chain audit log** (`src/agent/audit_chain.py`) — an honest substitute
for a "blockchain node." Real guarantee: each entry's hash is computed over
the previous entry's hash plus this entry's payload
(`sha256(prev_hash + timestamp + kind + payload)`), so tampering with any
past entry breaks every hash after it — the actual tamper-evidence property
a blockchain provides, verifiable via `audit_chain.verify_chain()`. What it
deliberately does **not** claim: distributed consensus, multiple independent
nodes, or Byzantine fault tolerance. It's a single-database hash chain, not
a blockchain.

**Simulated autopilot uplink** — the DRL shield's `delta_throttle`/
`delta_mixture` dispatch (`src/server/server.py`'s WS loop) writes directly
to the in-process simulation state and is labeled in both the payload
(`dispatch["autopilot_uplink"]`) and the UI as a simulated STANAG 4586-style
closed-loop de-rate dispatch — no real flight-control protocol is
implemented, and none is claimed.

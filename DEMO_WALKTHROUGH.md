# PRAHARI — 4-Minute Live Demo Walkthrough

A tight, rehearsed click-path for judges. Total run time ~3.5-4 minutes.
Practice it end-to-end at least twice before presenting — the timing notes
below assume you're not improvising.

**Before you start:** open the app, log in once, and leave the browser on
`/dashboard` so you're not fumbling with the login screen live. Have a
second browser tab pre-opened to `/app` (don't navigate yet) so the "open
Live Ops in a second tab" beat is instant.

---

## Beat 1 — The Dashboard (30s)

Land on `/dashboard`. Say:

> "This is PRAHARI — a predictive-maintenance ground station for a MALE UAV
> aero engine. Everything you're about to see is backed by a real physics-
> informed neural net, a 3-model classifier ensemble, a DRL safety shield,
> and it's all persisted to Postgres — not just a live demo that resets on
> refresh."

Point at the stat cards: total missions, classifier ensemble accuracy
(**say the honest number out loud — don't dodge it**, see note below),
PINN MAE. Point at "Why PRAHARI" — this is your one-line pitch against
generic statistical-only ML.

**⚠️ Say the accuracy number honestly.** If the classifier retrain hasn't
landed, the card will show ~47%. Own it: *"We found and are actively fixing
a real data-leakage bug in our own training pipeline — most teams wouldn't
catch this, and MODEL_CARD.md documents exactly what's wrong and why."*
This reads as rigor, not weakness, if you say it before a judge finds it.

## Beat 2 — Live Ops Console, nominal state (45s)

Click "LAUNCH LIVE OPS CONSOLE." Let the 3D twin load. Say:

> "This is the actual digital twin — Rotax 914F aero engine, live 10 Hz
> telemetry, physics-informed RUL prediction on the left, the fault
> classifier ensemble and DRL prognostic strategist on the right."

Rotate the 3D model once with the mouse. Point at the Parameter Rail
sparklines actively updating. Point at "Defense-Layer Status" — ensemble
agreement, action mode, comms integrity, trend risk — all real, all live.

## Beat 3 — Inject a fault, show the safe-mode gate (60s)

Scroll to the Fault Injection Testbed at the bottom. Click **"SENSOR SIGNAL
CORRUPTION."** Narrate as it happens:

> "I'm injecting a sensor dropout — a real fault, not a canned animation."

Watch for (~2-3s): the parameter rail shows `[IMP]` imputation tags, the
Defense-Layer Status panel flips **ACTION MODE → SAFE MODE**, and the DRL
panel shows the "holding last known-good setting, escalated to operator"
message. Say:

> "The system just refused to act on unverified telemetry — that's the
> safe-mode gate, not a scripted response. It's reading real ensemble
> agreement and sensor integrity right now."

Click **"NOMINAL CRUISE"** to clear it before moving on.

## Beat 4 — Mission Map + threat scenario (60s)

Navigate to `/mission-map`. Say:

> "Same mission, geographic context — real DRDO lab locations, a patrol
> trajectory, and simulated threats to the aircraft's own systems."

Click one threat button (**"EW / GPS Denial"** is the most visually clear —
it reliably trips safe mode). Point at the **live-monitor-strip at the top**
(not buried in a sidebar) updating in real time. If you have the `/app` tab
open, alt-tab to it for 2 seconds to show the *same* fault reacting there —
this is the strongest "it's not two separate demos" beat in the whole walk.
Click **"CLEAR / RETURN TO NOMINAL"** before moving on.

## Beat 5 — AI Advisor / orchestrator control (45s)

Back on `/app`, scroll the right sidebar to "AI Advisor (GLM)." Say:

> "This isn't a chatbot bolted on for show — it reads the real computed
> defense-layer state and can trigger its own diagnostic."

Click **"ASK AI ADVISOR."** It takes ~10-20s — narrate through the wait:

> "It's reasoning over live PINN residual, ensemble agreement, and trend
> risk right now — not calling out to fabricate an answer."

When it returns, point at the synthesis text, the recommended intent, and
**"AUTO-TRIGGERED"** if one fired — that's the orchestrator-control part:
the LLM decided *and acted*, not just suggested. Point at the audit-chain
hash underneath: *"Every action is hash-chained for tamper-evidence — not a
real blockchain, we're upfront about that in the model card, but the
tamper-evidence property is real and verifiable."*

## Beat 6 — Close on the Dashboard (20s)

Navigate back to `/dashboard`. Point at the missions table — the session you
just ran is already there with its fault-event timeline. Say:

> "Every session, every fault, every AI-triggered diagnostic — persisted,
> queryable, and this whole environment bootstraps with one script if you
> want to run it yourselves."

---

## If something breaks live

- **Wifi drops:** the app falls back to local SQLite automatically (see
  `src/db/session.py`) — auth/dashboard/missions keep working, just not
  synced to the shared cloud DB. The AI Advisor panel auto-hides if GLM is
  unreachable instead of showing a broken button. You can say this out loud
  if it happens — it's a feature, not a bug you're covering for.
- **AI Advisor quota hits 0/8:** it's hard rate-limited on purpose (cost
  control). If you've already used your quota in rehearsal, either skip
  Beat 5 or mention the rate limit is deliberate: *"We cap this hard —
  it's a cost-conscious design choice, not a technical ceiling."*
- **A fault doesn't clear cleanly:** click "NOMINAL CRUISE" on the Live Ops
  Console fault bar — it's the single source of truth and clears state
  everywhere (Mission Map, dashboard) since they all read the same backend.

## What NOT to claim

- Don't say the classifier is highly accurate unless you've re-checked
  `/dashboard`'s stat card that day — it's pulled live from
  `src/eval/benchmarks.py`, not hardcoded, so it'll tell the truth even if
  you forget to check.
- Don't call the hash-chain audit log a "blockchain" — say "hash-chained
  audit log" and let MODEL_CARD.md's honesty back you up if asked.
- Don't claim the STANAG 4586 uplink is a real flight-control integration —
  it's labeled "simulated" in the UI for exactly this reason.

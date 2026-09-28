const pptxgen = require("pptxgenjs");

const NAVY = "0B1220";
const PANEL = "141B2E";
const PANEL2 = "1B2440";
const INK = "0B1220";
const TEXT_LIGHT = "E7ECF5";
const TEXT_MUTED = "8B98B5";
const WHITE = "FFFFFF";
const TEAL = "22D3C7";
const MINT = "34D399";
const AMBER = "F5A623";
const RED = "E5484D";
const ICEBLUE = "CADCFC";

function fresh(base) { return Object.assign({}, base); }

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };

function footer(slide, num, title) {
  slide.addText(`SIH26054 · PRAHARI`, {
    x: 0.5, y: 7.16, w: 4, h: 0.28, fontFace: "Calibri", fontSize: 9,
    color: TEXT_MUTED, isTextBox: true, margin: 0,
  });
  slide.addText(String(num), {
    x: 12.6, y: 7.16, w: 0.5, h: 0.28, fontFace: "Calibri", fontSize: 9,
    color: TEXT_MUTED, align: "right", isTextBox: true, margin: 0,
  });
}

function lightTitle(slide, kicker, title) {
  slide.addText(kicker.toUpperCase(), {
    x: 0.6, y: 0.35, w: 10, h: 0.3, fontFace: "Calibri", fontSize: 12,
    color: TEAL, bold: true, charSpacing: 2, isTextBox: true, margin: 0,
  });
  slide.addText(title, {
    x: 0.6, y: 0.62, w: 12.1, h: 0.75, fontFace: "Cambria", fontSize: 30,
    color: NAVY, bold: true, isTextBox: true, margin: 0,
  });
}

function iconCircle(slide, x, y, d, color, letter) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color }, line: { type: "none" } });
  slide.addText(letter, {
    x, y, w: d, h: d, align: "center", valign: "middle", fontFace: "Calibri",
    fontSize: d * 26, bold: true, color: NAVY, isTextBox: true, margin: 0,
  });
}

// ============================================================ SLIDE 0 — COVER
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // subtle grid-ish accent shapes (motif: rounded chips, not stripes)
  s.addShape("roundRect", { x: 9.6, y: -1.2, w: 6, h: 6, rectRadius: 3, fill: { color: PANEL2, transparency: 40 }, line: { type: "none" }, rotate: 20 });

  iconCircle(s, 0.7, 0.65, 0.62, TEAL, "P");

  s.addText("PRAHARI", {
    x: 1.5, y: 0.55, w: 8, h: 0.8, fontFace: "Cambria", fontSize: 40, bold: true,
    color: WHITE, isTextBox: true, margin: 0,
  });
  s.addText("Predictive Reliability & Health Assessment for Rotary Intelligence", {
    x: 1.5, y: 1.28, w: 9.5, h: 0.4, fontFace: "Calibri", fontSize: 14, italic: true,
    color: ICEBLUE, isTextBox: true, margin: 0,
  });

  s.addText("An Agentic, Physics-Informed Digital Twin for\nMALE UAV Aero Piston Engines", {
    x: 0.7, y: 2.9, w: 8.6, h: 1.7, fontFace: "Cambria", fontSize: 30, bold: true,
    color: WHITE, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
  });

  s.addText("TAPAS-BH-201 MALE UAV  ·  Rotax 914F Aero Piston Engine  ·  DRDO Reliability & Mission Assurance", {
    x: 0.7, y: 4.55, w: 9, h: 0.4, fontFace: "Calibri", fontSize: 13,
    color: TEXT_MUTED, isTextBox: true, margin: 0,
  });

  const chips = [
    ["Problem Statement", "SIH26054"],
    ["Theme", "Robotics & Drones"],
    ["Category", "Software"],
    ["Team", "[ Team Name ]"],
  ];
  let cx = 0.7;
  chips.forEach(([label, val]) => {
    s.addShape("roundRect", { x: cx, y: 6.15, w: 2.85, h: 0.85, rectRadius: 0.08, fill: { color: PANEL }, line: { color: "26314F", width: 1 } });
    s.addText(label.toUpperCase(), { x: cx + 0.18, y: 6.25, w: 2.5, h: 0.25, fontFace: "Calibri", fontSize: 9, color: TEXT_MUTED, bold: true, charSpacing: 1, isTextBox: true, margin: 0 });
    s.addText(val, { x: cx + 0.18, y: 6.5, w: 2.5, h: 0.4, fontFace: "Calibri", fontSize: 14, color: WHITE, bold: true, isTextBox: true, margin: 0 });
    cx += 3.0;
  });
}

// ============================================================ SLIDE 1 — LITERATURE GAP
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Literature Review & Research Gap", "Why Existing Approaches Fail Military-Grade MALE UAV Engines");
  footer(s, 1);

  const rows = [
    { n: "1", name: "Pure Data-Driven RUL", sub: "LSTM / Transformer on C-MAPSS-style data", strength: "Captures complex non-linear sensor correlations without needing engine thermodynamics", flaw: "Can output thermodynamically impossible predictions under unseen conditions (e.g. sudden high-altitude pressure drop)", color: "5B7FDE" },
    { n: "2", name: "Traditional PINNs", sub: "Physics-Informed Neural Networks", strength: "High interpretability, strictly obeys thermal/structural laws", flaw: "Rigid & compute-heavy online; solves forward equations well but can't do active, multi-objective, closed-loop decisions", color: TEAL },
    { n: "3", name: "Standard Deep RL", sub: "For control / prognostics", strength: "Excellent sequential decision-making, adaptive optimization", flaw: "Black-box exploration — trial-and-error behavior is unacceptable in a live military UAV engine", color: MINT },
    { n: "4", name: "Modular Digital Twins", sub: "Isolated state-estimation / anomaly / RUL blocks", strength: "Clean, debuggable software engineering", flaw: "Silent failure propagation — a null/delayed payload from one module is silently treated as “nominal” downstream", color: AMBER },
  ];

  let y = 1.55;
  rows.forEach((r) => {
    s.addShape("roundRect", { x: 0.6, y, w: 12.1, h: 1.12, rectRadius: 0.06, fill: { color: "F7F9FC" }, line: { color: "E3E8F2", width: 1 } });
    iconCircle(s, 0.82, y + 0.24, 0.62, r.color, r.n);
    s.addText(r.name, { x: 1.65, y: y + 0.1, w: 3.0, h: 0.35, fontFace: "Calibri", fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(r.sub, { x: 1.65, y: y + 0.44, w: 3.0, h: 0.55, fontFace: "Calibri", fontSize: 9.5, italic: true, color: TEXT_MUTED, isTextBox: true, margin: 0 });
    s.addText([{ text: "Strength: ", options: { bold: true, color: "2C5F2D" } }, { text: r.strength, options: { color: "303A4E" } }], { x: 4.85, y: y + 0.08, w: 3.75, h: 1.0, fontFace: "Calibri", fontSize: 10, isTextBox: true, margin: 0, valign: "middle" });
    s.addText([{ text: "Fatal flaw: ", options: { bold: true, color: RED } }, { text: r.flaw, options: { color: "303A4E" } }], { x: 8.75, y: y + 0.08, w: 3.85, h: 1.0, fontFace: "Calibri", fontSize: 10, isTextBox: true, margin: 0, valign: "middle" });
    y += 1.18;
  });

  s.addShape("roundRect", { x: 0.6, y: y + 0.02, w: 12.1, h: 0.55, rectRadius: 0.06, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("No single paradigm gives physical trustworthiness AND adaptive decision-making AND fault-tolerant orchestration at once — PRAHARI is the first to combine all three.", {
    x: 0.85, y: y + 0.02, w: 11.6, h: 0.55, fontFace: "Calibri", fontSize: 12, italic: true, bold: true,
    color: WHITE, valign: "middle", isTextBox: true, margin: 0,
  });
}

// ============================================================ SLIDE 2 — SOLUTION OVERVIEW
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Proposed Solution", "PRAHARI: An Agentic Physics-Informed Digital Twin");
  footer(s, 2);

  s.addShape("roundRect", { x: 0.6, y: 1.55, w: 12.1, h: 0.98, rectRadius: 0.06, fill: { color: "EEF4FF" }, line: { type: "none" } });
  s.addText("A real-time digital twin that fuses a Physics-Informed Neural Network, an ensemble fault classifier, safety-shielded Deep RL, and a LangGraph-orchestrated AI agent — with an autonomous LLM advisor layer — to predict, explain, and safely act on aero piston engine health, with zero tolerance for silent failure.", {
    x: 0.9, y: 1.55, w: 11.5, h: 0.98, fontFace: "Calibri", fontSize: 13, italic: true, bold: true,
    color: NAVY, valign: "middle", isTextBox: true, margin: 0,
  });

  const tiles = [
    ["1", "Real-Time Telemetry Ingestion", "10 Hz stream — RPM, CHT, EGT, oil pressure, fuel flow, vibration, MAP, AFR", "5B7FDE"],
    ["2", "Physics-Constrained Anomaly Detection", "PINN layer enforces thermodynamic law inside the loss function itself", TEAL],
    ["3", "Ensemble Fault Classification + DRL RUL", "Multi-model voting classifier + safety-shielded DRL prognostic strategist", MINT],
    ["4", "Intent-Routed Diagnostic Agent", "LangGraph router with a hash-chained audit log for tamper-evident traceability", AMBER],
    ["5", "3D Digital Twin + Mission Map", "Exploded engine view with live heatmaps, plus a geospatial mission command view", "9B6BDB"],
    ["6", "Autonomous AI Advisor", "LLM reads live defense-layer state and can trigger its own diagnostic branch — orchestrator-control, not just chat", RED],
  ];
  let tx = 0.6, ty = 2.75;
  tiles.forEach((t, i) => {
    if (i === 3) { tx = 0.6; ty += 1.75; }
    s.addShape("roundRect", { x: tx, y: ty, w: 3.9, h: 1.55, rectRadius: 0.08, fill: { color: "FFFFFF" }, line: { color: "E3E8F2", width: 1.25 }, shadow: { type: "outer", color: "1B2440", opacity: 0.12, blur: 6, offset: 2, angle: 90 } });
    iconCircle(s, tx + 0.22, ty + 0.22, 0.5, t[3], t[0]);
    s.addText(t[1], { x: tx + 0.85, y: ty + 0.16, w: 2.9, h: 0.6, fontFace: "Calibri", fontSize: 12, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(t[2], { x: tx + 0.22, y: ty + 0.82, w: 3.5, h: 0.65, fontFace: "Calibri", fontSize: 9.5, color: TEXT_MUTED, isTextBox: true, margin: 0 });
    tx += 4.15;
  });
}

// ============================================================ SLIDE 3 — TECHNICAL ARCHITECTURE
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Technical Approach, Architecture & Math", "PINN → Ensemble → DRL → Agentic Pipeline");
  footer(s, 3);

  // Flow diagram
  const boxes = [
    { t: "Engine /\nSimulator", c: "5B7FDE" },
    { t: "Sensor\nAuditor", c: TEAL },
    { t: "PINN\nPhysics Check", c: TEAL },
    { t: "Ensemble\nClassifier + DRL", c: MINT },
    { t: "Safe-Mode\nGate", c: AMBER },
    { t: "3D Twin + Map\n+ AI Advisor", c: RED },
  ];
  let bx = 0.6;
  const by = 1.6, bw = 1.85, bh = 0.85, gap = 0.22;
  boxes.forEach((b, i) => {
    s.addShape("roundRect", { x: bx, y: by, w: bw, h: bh, rectRadius: 0.07, fill: { color: b.c }, line: { type: "none" } });
    s.addText(b.t, { x: bx, y: by, w: bw, h: bh, align: "center", valign: "middle", fontFace: "Calibri", fontSize: 10.5, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    if (i < boxes.length - 1) {
      s.addText("→", { x: bx + bw, y: by, w: gap, h: bh, align: "center", valign: "middle", fontFace: "Calibri", fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    }
    bx += bw + gap;
  });
  s.addText("Continuous 10 Hz safety loop (above) runs every frame. A second, on-demand LangGraph intent router (sensor integrity / thermal stress / root-cause diagnostic) is directly triggerable from the ground station for deep-dive diagnostics.", {
    x: 0.6, y: 2.55, w: 12.1, h: 0.4, fontFace: "Calibri", fontSize: 9.5, italic: true, color: TEXT_MUTED, isTextBox: true, margin: 0,
  });

  // Three layers
  const layers = [
    ["PINN Layer — “Physics Guardian”", "Continuous estimator of structural/thermal stress; forces predictions to obey Fourier's law of heat conduction and gas-law transformations.", TEAL],
    ["Ensemble + DRL — “Prognostic Strategist”", "Multi-model voting classifier feeds a safety-shielded DRL policy; action space is boundary-constrained by the PINN's physical residual.", MINT],
    ["LangGraph Agent — “Orchestrator”", "Routes between tools with conditional edges, not a one-way pipeline — catches and halts anomalies mid-process instead of passing them downstream.", AMBER],
  ];
  let ly = 3.1;
  layers.forEach(([title, desc, color]) => {
    s.addShape("rect", { x: 0.6, y: ly + 0.05, w: 0.06, h: 0.62, fill: { color }, line: { type: "none" } });
    s.addText(title, { x: 0.82, y: ly, w: 5.6, h: 0.32, fontFace: "Calibri", fontSize: 11.5, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(desc, { x: 0.82, y: ly + 0.32, w: 5.6, h: 0.42, fontFace: "Calibri", fontSize: 9, color: TEXT_MUTED, isTextBox: true, margin: 0 });
    ly += 0.78;
  });

  // Math card
  s.addShape("roundRect", { x: 6.7, y: 3.1, w: 6.0, h: 1.55, rectRadius: 0.08, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("KEY MATHEMATICS — PINN LOSS", { x: 6.95, y: 3.22, w: 5.5, h: 0.28, fontFace: "Calibri", fontSize: 9.5, bold: true, color: TEAL, charSpacing: 1, isTextBox: true, margin: 0 });
  s.addText("L_Total = L_Data + λ · L_Physics", { x: 6.95, y: 3.52, w: 5.5, h: 0.4, fontFace: "Cambria", fontSize: 17, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addText("L_Physics = (1/N) Σ | ∂T/∂t − α∇²T − q_combustion |²", { x: 6.95, y: 3.9, w: 5.5, h: 0.35, fontFace: "Cambria", fontSize: 13, color: ICEBLUE, isTextBox: true, margin: 0 });
  s.addText("Penalizes any predicted temperature trajectory that violates real heat-transfer physics — the network literally cannot output an impossible reading.", { x: 6.95, y: 4.28, w: 5.5, h: 0.34, fontFace: "Calibri", fontSize: 8.5, italic: true, color: TEXT_MUTED, isTextBox: true, margin: 0 });

  // Tech stack table
  s.addText("TECH STACK", { x: 0.6, y: 5.5, w: 4, h: 0.3, fontFace: "Calibri", fontSize: 11, bold: true, color: NAVY, charSpacing: 1, isTextBox: true, margin: 0 });
  const stack = [
    ["AI/ML", "PyTorch, PINN + ensemble classifier + safety-shielded DRL (PPO)"],
    ["Orchestration & Agent", "LangGraph (conditional routing) · GLM-4.5 LLM advisor (orchestrator-control, rate-limited)"],
    ["Backend & Data", "FastAPI, WebSockets · SQLAlchemy on Neon Postgres, SQLite offline fallback"],
    ["Auth & Integrity", "JWT (httpOnly cookie), bcrypt · SHA-256 hash-chained audit log"],
    ["Visualization", "Three.js / WebGL 3D twin (custom shaders) · SVG geospatial Mission Map"],
  ];
  let sty = 5.82;
  stack.forEach(([k, v]) => {
    s.addText(k, { x: 0.6, y: sty, w: 2.6, h: 0.27, fontFace: "Calibri", fontSize: 9.5, bold: true, color: TEAL, isTextBox: true, margin: 0 });
    s.addText(v, { x: 3.25, y: sty, w: 9.45, h: 0.27, fontFace: "Calibri", fontSize: 9.5, color: "303A4E", isTextBox: true, margin: 0 });
    sty += 0.265;
  });
}

// ============================================================ SLIDE 4 — FEASIBILITY & VIABILITY
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Feasibility & Viability", "Built, Running, and Demo-Safe Today");
  footer(s, 4);

  const cols = [
    ["Technical", TEAL, [
      "Built entirely on open-source frameworks (PyTorch, FastAPI, LangGraph, Three.js) — no proprietary dependency risk",
      "PINN and DRL are established techniques individually; novelty is in the orchestrated combination",
      "Public C-MAPSS-style datasets allow development/validation without live UAV hardware access",
      "One-command bootstrap script (setup.sh / setup.ps1) — reproducible in minutes, not a fragile local setup",
    ]],
    ["Operational (DRDO context)", MINT, [
      "Directly addresses near-zero downtime need for military-grade MALE UAV engines",
      "Sensor Auditor is specifically designed to prevent silent “nominal” misreporting",
      "Modular — PINN / classifier / DRL / agent tools upgrade independently",
      "Demo-safety fallback: SQLite if Postgres is unreachable, AI Advisor auto-hides if GLM is unreachable — a venue wifi drop never kills a live run",
    ]],
    ["Economic", AMBER, [
      "Open-source stack keeps licensing cost near zero",
      "Compute cost (PINN + DRL) is manageable at prototype/demo scale; edge-inference optimization is a clear future-work item",
      "AI Advisor is hard rate-limited (8 calls/hour) — a deliberate cost control, not a technical ceiling",
      "Catching failures early reduces long-term cost vs. unscheduled field maintenance",
    ]],
  ];
  let fx = 0.6;
  cols.forEach(([title, color, items]) => {
    s.addShape("roundRect", { x: fx, y: 1.55, w: 3.97, h: 3.0, rectRadius: 0.08, fill: { color: "F7F9FC" }, line: { color: "E3E8F2", width: 1 } });
    s.addShape("roundRect", { x: fx + 0.2, y: 1.75, w: 0.34, h: 0.34, rectRadius: 0.06, fill: { color }, line: { type: "none" } });
    s.addText(title, { x: fx + 0.65, y: 1.75, w: 3.1, h: 0.34, fontFace: "Calibri", fontSize: 13, bold: true, color: NAVY, valign: "middle", isTextBox: true, margin: 0 });
    const bullets = items.map((it, i) => ({ text: it, options: { bullet: { code: "2726" }, color: "303A4E", breakLine: i < items.length - 1, paraSpaceAfter: 8 } }));
    s.addText(bullets, { x: fx + 0.2, y: 2.25, w: 3.57, h: 2.2, fontFace: "Calibri", fontSize: 9, isTextBox: true, margin: 0 });
    fx += 4.13;
  });

  s.addText("RISKS & MITIGATION", { x: 0.6, y: 4.75, w: 4, h: 0.3, fontFace: "Calibri", fontSize: 11, bold: true, color: NAVY, charSpacing: 1, isTextBox: true, margin: 0 });
  const risks = [
    ["Classifier real-world accuracy is 46.7% — a leakage bug had inflated earlier self-reported numbers to 100%", "Root-caused and documented in MODEL_CARD.md; dashboard shows the honest number; retraining fix in progress"],
    ["PINN training instability on real noisy telemetry", "Start with simulator-validated data, progressively introduce noise"],
    ["DRL reward mis-specification leading to unsafe suggestions", "Hard physics-based action boundary from PINN (already part of design)"],
    ["Network dependency (Postgres / GLM API) during a live demo", "SQLite fallback + AI Advisor auto-hide — implemented and tested end-to-end"],
  ];
  let ry = 5.06;
  risks.forEach(([risk, mit], i) => {
    const bg = i === 0 ? "FFF4E5" : "F7F9FC";
    const bd = i === 0 ? AMBER : "E3E8F2";
    s.addShape("roundRect", { x: 0.6, y: ry, w: 12.1, h: 0.44, rectRadius: 0.05, fill: { color: bg }, line: { color: bd, width: 1 } });
    s.addText([{ text: "Risk: ", options: { bold: true, color: RED } }, { text: risk, options: { color: "303A4E" } }], { x: 0.78, y: ry, w: 5.9, h: 0.44, fontFace: "Calibri", fontSize: 8.3, valign: "middle", isTextBox: true, margin: 0 });
    s.addText([{ text: "Mitigation: ", options: { bold: true, color: "2C5F2D" } }, { text: mit, options: { color: "303A4E" } }], { x: 6.8, y: ry, w: 5.75, h: 0.44, fontFace: "Calibri", fontSize: 8.3, valign: "middle", isTextBox: true, margin: 0 });
    ry += 0.5;
  });
}

// ============================================================ SLIDE 5 — IMPACT & BENEFITS
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Impact & Benefits", "From Reactive Maintenance to Predictive Mission Assurance");
  footer(s, 5);

  s.addText("DIRECT IMPACT", { x: 0.6, y: 1.55, w: 6, h: 0.3, fontFace: "Calibri", fontSize: 11, bold: true, color: TEAL, charSpacing: 1, isTextBox: true, margin: 0 });
  const direct = [
    "Prevents mid-flight engine failure by catching anomalies before they cascade",
    "Gives ground stations actionable RUL data for real-time go/no-go and recovery-maneuver decisions",
    "Removes “silent failure” risk that plagues conventional modular digital twins — a safety improvement, not just a performance one",
    "Mission Map ties engine health to geographic/mission context — the same fault reacting live across dashboard, live-ops console and map",
  ];
  s.addText(direct.map((t, i) => ({ text: t, options: { bullet: { code: "2726" }, color: "303A4E", breakLine: i < direct.length - 1, paraSpaceAfter: 10 } })), { x: 0.6, y: 1.9, w: 6.0, h: 2.5, fontFace: "Calibri", fontSize: 10.5, isTextBox: true, margin: 0 });

  s.addText("STRATEGIC / DEFENCE IMPACT", { x: 6.9, y: 1.55, w: 6, h: 0.3, fontFace: "Calibri", fontSize: 11, bold: true, color: TEAL, charSpacing: 1, isTextBox: true, margin: 0 });
  const strat = [
    "Reduces unplanned downtime → higher UAV fleet availability for missions",
    "Physics-constrained predictions build trust for safety-critical, high-stakes defence systems, unlike black-box-only ML",
    "Every AI-triggered diagnostic is hash-chain audited — tamper-evident, traceable decision history for post-mission review",
    "Extensible to other DRDO aero-propulsion systems (turboprop, hybrid-electric UAV powertrains) as future scope",
  ];
  s.addText(strat.map((t, i) => ({ text: t, options: { bullet: { code: "2726" }, color: "303A4E", breakLine: i < strat.length - 1, paraSpaceAfter: 10 } })), { x: 6.9, y: 1.9, w: 5.8, h: 2.5, fontFace: "Calibri", fontSize: 10.5, isTextBox: true, margin: 0 });

  // before/after split
  s.addShape("roundRect", { x: 0.6, y: 4.55, w: 5.9, h: 1.95, rectRadius: 0.08, fill: { color: "FDF3F3" }, line: { color: "F3D3D4", width: 1 } });
  s.addText("TRADITIONAL MODULAR PIPELINE", { x: 0.85, y: 4.7, w: 5.4, h: 0.3, fontFace: "Calibri", fontSize: 10, bold: true, color: RED, isTextBox: true, margin: 0 });
  s.addText("Sensor → Anomaly → RUL   ✗ module fails silently → downstream treats it as “nominal” → mid-flight surprise", { x: 0.85, y: 5.05, w: 5.4, h: 1.3, fontFace: "Calibri", fontSize: 10, color: "6E2A2C", isTextBox: true, margin: 0, valign: "top" });

  s.addShape("roundRect", { x: 6.8, y: 4.55, w: 5.9, h: 1.95, rectRadius: 0.08, fill: { color: "EAFBF6" }, line: { color: "BEEDDD", width: 1 } });
  s.addText("PRAHARI AGENTIC DIGITAL TWIN", { x: 7.05, y: 4.7, w: 5.4, h: 0.3, fontFace: "Calibri", fontSize: 10, bold: true, color: "0E7C5A", isTextBox: true, margin: 0 });
  s.addText("Sensor → Auditor catches malformed payload → halts & flags → safe-mode gate engages → operator alerted with traceable reason", { x: 7.05, y: 5.05, w: 5.4, h: 1.3, fontFace: "Calibri", fontSize: 10, color: "0E4A38", isTextBox: true, margin: 0, valign: "top" });

  s.addText("Benefit framing below is presented as directional design targets validated in our own test harness — not claimed field-validated results.", {
    x: 0.6, y: 6.65, w: 12.1, h: 0.3, fontFace: "Calibri", fontSize: 8.5, italic: true, color: TEXT_MUTED, isTextBox: true, margin: 0,
  });
}

// ============================================================ SLIDE 6 — REFERENCES
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "References", "Literature, Datasets & Tooling");
  footer(s, 6);

  const refs = [
    "Deep learning (LSTM/Attention) approaches to Remaining Useful Life prediction using C-MAPSS-style aero-propulsion prognostics benchmarks (MDPI/IEEE shared benchmarks).",
    "Hybrid digital twin approaches combining Kalman filtering with multi-layer perceptrons for electric/piston aircraft engines.",
    "Online-adaptive Physics-Informed Neural Network frameworks for remaining-useful-life prediction.",
    "Digital twin frameworks for UAV-assisted edge environments (IEEE Transactions), covering modular health/sensing/trajectory pipelines.",
    "NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset — reference dataset class for engine degradation simulation.",
    "IEEE Dataport — supplementary aero-engine sensor data profiles.",
    "LangGraph (LangChain AI) — stateful multi-agent orchestration framework used for the conditional diagnostic router.",
    "GLM-4.5 (Zhipu AI) — LLM backing the rate-limited AI Advisor orchestrator-control layer.",
  ];
  const colA = refs.slice(0, 4), colB = refs.slice(4);
  [colA, colB].forEach((col, ci) => {
    const x = 0.6 + ci * 6.2;
    const items = col.map((r, i) => ({ text: `${ci * 4 + i + 1}. ${r}`, options: { color: "303A4E", breakLine: i < col.length - 1, paraSpaceAfter: 14 } }));
    s.addText(items, { x, y: 1.7, w: 5.9, h: 4.6, fontFace: "Calibri", fontSize: 10, isTextBox: true, margin: 0 });
  });

  s.addShape("roundRect", { x: 0.6, y: 6.4, w: 12.1, h: 0.55, rectRadius: 0.05, fill: { color: "FFF4E5" }, line: { color: AMBER, width: 1 } });
  s.addText("Exact author names, years and full IEEE-style citation formatting to be finalized against source papers before submission.", {
    x: 0.85, y: 6.4, w: 11.6, h: 0.55, fontFace: "Calibri", fontSize: 9.5, italic: true, color: "6E4A00", valign: "middle", isTextBox: true, margin: 0,
  });
}

pres.writeFile({ fileName: "PRAHARI_SIH26054_Pitch_Deck.pptx" }).then(() => {
  console.log("done");
});

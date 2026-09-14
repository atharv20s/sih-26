/**
 * SIH26054 Digital Twin Front-End Dashboard Controller
 *
 * Coordinates:
 *   - Real-time WebSocket connection to FastAPI backend (10 Hz)
 *   - Three.js 3D viewport synchronization
 *   - Telemetry gauge updates & color warnings
 *   - PINN thermodynamic compliance display
 *   - DRL Prognostic recommendation & safety shield alerts
 *   - Fault injection controls & benchmark modal
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize 3D Engine
  const engine3D = new AeroEngine3D('webgl-canvas');
  window._engine3D = engine3D;
  window.dispatchEvent(new CustomEvent('engine3dReady', { detail: engine3D }));

  // DOM Elements
  const explodedSlider = document.getElementById('exploded-slider');
  const explodedVal = document.getElementById('exploded-val');
  const btnHeatmap = document.getElementById('btn-heatmap');
  const btnZoomIn = document.getElementById('btn-zoom-in');
  const btnZoomOut = document.getElementById('btn-zoom-out');
  const btnResetView = document.getElementById('btn-reset-view');

  const rulNumber = document.getElementById('rul-number');
  const rulExtension = document.getElementById('rul-extension');
  const rulTimeVal = document.getElementById('rul-time-val');
  const rulTimeExt = document.getElementById('rul-time-ext');
  const rulStatusBadge = document.getElementById('rul-status-badge');
  const rulHeroCard = document.getElementById('rul-hero-card');
  const fourierResidual = document.getElementById('fourier-residual');
  const physicalGradient = document.getElementById('physical-gradient');
  const fourierAdherence = document.getElementById('fourier-adherence');

  const faultName = document.getElementById('fault-name');
  const faultConf = document.getElementById('fault-conf');
  const drlText = document.getElementById('drl-text');
  const drlShieldBadge = document.getElementById('drl-shield-badge');
  const auditRecoveries = document.getElementById('audit-recoveries');
  const wsStatus = document.getElementById('ws-status');

  // Benchmark Modal Elements
  const btnBenchmarks = document.getElementById('btn-benchmarks');
  const modalBackdrop = document.getElementById('modal-backdrop');
  const modalClose = document.getElementById('modal-close');
  const benchmarkTbody = document.getElementById('benchmark-tbody');

  // Architecture Modal Elements
  const btnArchitecture = document.getElementById('btn-architecture');
  const modalArchitecture = document.getElementById('modal-architecture');
  const modalArchClose = document.getElementById('modal-arch-close');
  const archTabBtns = document.querySelectorAll('.arch-tab-btn');
  const archTabPanes = document.querySelectorAll('.arch-tab-pane');

  // Day / Night Tactical Mode Toggle
  const btnThemeToggle = document.getElementById('btn-theme-toggle');
  const themeIcon = document.getElementById('theme-icon');
  const themeText = document.getElementById('theme-text');

  function updateTheme(isLight) {
    if (isLight) {
      document.body.classList.add('light-theme');
      if (themeIcon) themeIcon.textContent = '🌙';
      if (themeText) themeText.textContent = 'NIGHT MODE';
      if (window.engine3D) window.engine3D.setTheme('day');
      localStorage.setItem('tapas_theme', 'light');
    } else {
      document.body.classList.remove('light-theme');
      if (themeIcon) themeIcon.textContent = '☀️';
      if (themeText) themeText.textContent = 'DAY MODE';
      if (window.engine3D) window.engine3D.setTheme('night');
      localStorage.setItem('tapas_theme', 'dark');
    }
  }

  if (btnThemeToggle) {
    const savedTheme = localStorage.getItem('tapas_theme') || 'dark';
    if (savedTheme === 'light') updateTheme(true);

    btnThemeToggle.addEventListener('click', () => {
      const isLight = document.body.classList.contains('light-theme');
      updateTheme(!isLight);
    });
  }

  // Exploded View Slider Listener
  // Exploded View Slider Listener (0% - 100%)
  if (explodedSlider && explodedVal) {
    explodedSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      explodedVal.textContent = `${Math.round(val)}%`;
      engine3D.setExplodedView(val);
    });
  }

  // Toggle Heatmap Button (GLSL Volumetric Shader)
  if (btnHeatmap) {
    btnHeatmap.addEventListener('click', () => {
      const active = engine3D.toggleHeatmap();
      btnHeatmap.classList.toggle('active', active);
    });
  }

  // Zoom In / Out Buttons
  if (btnZoomIn) {
    btnZoomIn.addEventListener('click', () => {
      engine3D.zoomIn(0.25);
    });
  }
  if (btnZoomOut) {
    btnZoomOut.addEventListener('click', () => {
      engine3D.zoomOut(0.25);
    });
  }

  // Reset Camera View
  if (btnResetView) {
    btnResetView.addEventListener('click', () => {
      engine3D.resetView();
    });
  }

  // View Manager Mode Switcher (Syncs both viewport and mission view manager buttons)
  const viewBtns = document.querySelectorAll('.vm-btn, .mvm-btn');
  viewBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-view');
      viewBtns.forEach(b => {
        if (b.getAttribute('data-view') === mode) {
          b.classList.add('active');
        } else {
          b.classList.remove('active');
        }
      });
      engine3D.setViewMode(mode);
    });
  });

  // -------------------------------------------------------------------------
  // Feature 3: Mission Replay & Environmental Simulation Controls
  // -------------------------------------------------------------------------
  const sliderAltitude = document.getElementById('slider-altitude');
  const valAltitude = document.getElementById('val-altitude');
  const envDensity = document.getElementById('env-density');
  const envPressure = document.getElementById('env-pressure');

  const sliderAmbientTemp = document.getElementById('slider-ambient-temp');
  const valAmbientTemp = document.getElementById('val-ambient-temp');
  const envCondition = document.getElementById('env-condition');
  const envConvection = document.getElementById('env-convection');

  const replayClock = document.getElementById('replay-clock');
  const scrubberCurrentTime = document.getElementById('scrubber-current-time');
  const replayTimeline = document.getElementById('replay-timeline');
  const btnReplayPlay = document.getElementById('btn-replay-play');
  const btnReplayStepBack = document.getElementById('btn-replay-step-back');
  const btnReplayStepFwd = document.getElementById('btn-replay-step-fwd');
  const selectReplaySpeed = document.getElementById('select-replay-speed');
  const phasePills = document.querySelectorAll('.phase-pill');

  let replaySeconds = 2535; // Default: T+00:42:15
  let isReplayPlaying = false;
  let replayInterval = null;
  let replaySpeed = 1.0;

  function formatTime(totalSec) {
    const h = Math.floor(totalSec / 3600).toString().padStart(2, '0');
    const m = Math.floor((totalSec % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(totalSec % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
  }

  // -------------------------------------------------------------------------
  // Subassembly Health Matrix Controller (High-Contrast Real-Time Cards)
  // -------------------------------------------------------------------------
  function updateSubassemblyHealth(healthData, telemetry) {
    if (!healthData) return;

    const components = [
      {
        id: 'cylinder_head',
        metric: `${(telemetry && telemetry.cht !== undefined && !isNaN(telemetry.cht) ? telemetry.cht : 148.6).toFixed(1)}°C CHT`,
      },
      {
        id: 'crankshaft',
        metric: `${(telemetry && telemetry.vibration_rms !== undefined && !isNaN(telemetry.vibration_rms) ? telemetry.vibration_rms : 1.41).toFixed(2)} g VIB`,
      },
      {
        id: 'lubrication_system',
        metric: `${(telemetry && telemetry.oil_pressure !== undefined && !isNaN(telemetry.oil_pressure) && telemetry.oil_pressure > 0 ? telemetry.oil_pressure : 281.8).toFixed(1)} kPa`,
      },
      {
        id: 'exhaust_manifold',
        metric: `${(telemetry && telemetry.egt !== undefined && !isNaN(telemetry.egt) ? telemetry.egt : 667.7).toFixed(1)}°C EGT`,
      }
    ];

    components.forEach(comp => {
      const hVal = healthData[comp.id] !== undefined ? healthData[comp.id] : 0.91;
      const pct = Math.max(0, Math.min(100, Math.round(hVal * 100)));

      const statusEl = document.getElementById(`comp-status-${comp.id}`);
      const barEl = document.getElementById(`comp-bar-${comp.id}`);
      const metricEl = document.getElementById(`comp-metric-${comp.id}`);
      const condEl = document.getElementById(`comp-cond-${comp.id}`);

      if (statusEl) {
        statusEl.textContent = `${pct}%`;
        if (pct >= 80) {
          statusEl.style.color = '#34d399';
          statusEl.style.textShadow = '0 0 10px rgba(52, 211, 153, 0.45)';
        } else if (pct >= 55) {
          statusEl.style.color = '#fbbf24';
          statusEl.style.textShadow = '0 0 10px rgba(251, 191, 36, 0.45)';
        } else {
          statusEl.style.color = '#f43f5e';
          statusEl.style.textShadow = '0 0 10px rgba(244, 63, 94, 0.55)';
        }
      }

      if (barEl) {
        barEl.style.width = `${pct}%`;
        barEl.classList.remove('warning', 'critical');
        if (pct < 55) {
          barEl.classList.add('critical');
        } else if (pct < 80) {
          barEl.classList.add('warning');
        }
      }

      if (metricEl) {
        metricEl.textContent = comp.metric;
      }

      if (condEl) {
        condEl.classList.remove('warning', 'critical');
        if (pct >= 85) {
          condEl.textContent = 'NOMINAL';
        } else if (pct >= 70) {
          condEl.textContent = 'DEGRADED';
          condEl.classList.add('warning');
        } else if (pct >= 50) {
          condEl.textContent = 'ELEVATED WEAR';
          condEl.classList.add('warning');
        } else {
          condEl.textContent = 'CRITICAL';
          condEl.classList.add('critical');
        }
      }
    });
  }

  // -------------------------------------------------------------------------
  // Remaining Useful Life (RUL) & Flight Sustainment Display Controller
  // -------------------------------------------------------------------------
  function updateRulDisplay(rulCycles, extensionCycles, isPhysicallyValid, sustainStr, missionStatus) {
    const rawCycles = rulCycles !== undefined && !isNaN(rulCycles) ? rulCycles : 485.0;
    const cycles = Math.round(rawCycles);
    const ext = extensionCycles !== undefined && !isNaN(extensionCycles) ? extensionCycles : 0.0;

    if (rulNumber) {
      rulNumber.textContent = cycles;
    }

    // 1 cycle ≈ 0.1 flight hours (6 minutes) on Rotax 914 F powered TAPAS UAV
    let flightHours = rawCycles * 0.1;
    let timeStr = sustainStr;
    if (!timeStr) {
      const hrs = Math.floor(flightHours);
      const mins = Math.round((flightHours - hrs) * 60);
      const paddedMins = mins.toString().padStart(2, '0');
      if (cycles <= 35) {
        timeStr = `⚠️ ${hrs}h ${paddedMins}m EMERGENCY RTB`;
      } else {
        timeStr = `${hrs}h ${paddedMins}m Mission Endurance`;
      }
    }

    // Extension time saved calculation
    const extHours = ext * 0.1;
    const extHrs = Math.floor(extHours);
    const extMins = Math.round((extHours - extHrs) * 60);
    const extTimeStr = ext > 1.0 ? `(+${extHrs > 0 ? extHrs + 'h ' : ''}${extMins}m Saved)` : '';

    if (rulExtension) {
      rulExtension.textContent = ext > 1.0 ? `+${ext.toFixed(1)} cyc (DRL Protected)` : `Nominal Cruise`;
    }
    if (rulTimeExt) {
      rulTimeExt.textContent = extTimeStr;
      rulTimeExt.style.display = ext > 1.0 ? 'inline' : 'none';
    }

    // Alert thresholds and status pills
    let statusClass = 'status-ok';
    let badgeText = '🟢 OPTIMAL';
    let cardClass = '';
    let timeValClass = '';

    if (missionStatus === 'CRITICAL_RTB' || cycles <= 35 || (isPhysicallyValid === false && cycles <= 60)) {
      statusClass = 'status-crit';
      badgeText = '🔴 CRITICAL RTB';
      cardClass = 'critical';
      timeValClass = 'critical';
      if (!timeStr.includes('⚠️')) {
        timeStr = `⚠️ ${timeStr.replace('Mission Endurance', 'EMERGENCY RTB WINDOW')}`;
      }
    } else if (missionStatus === 'ELEVATED_WEAR' || cycles <= 180) {
      statusClass = 'status-warn';
      badgeText = '🟡 ELEVATED WEAR';
      cardClass = 'warning';
      timeValClass = 'warning';
    } else {
      statusClass = 'status-ok';
      badgeText = '🟢 OPTIMAL';
      cardClass = '';
      timeValClass = '';
    }

    if (rulTimeVal) {
      rulTimeVal.textContent = timeStr;
      rulTimeVal.className = 'rul-time-val ' + timeValClass;
    }

    if (rulStatusBadge) {
      rulStatusBadge.className = 'rul-status-pill ' + statusClass;
      rulStatusBadge.textContent = badgeText;
    }

    if (rulHeroCard) {
      rulHeroCard.className = 'rul-hero-card ' + cardClass;
    }
  }

  let envDebounceTimer = null;

  function updateEnvironmentalState() {
    if (!sliderAltitude || !sliderAmbientTemp) return;
    const altFt = parseFloat(sliderAltitude.value);
    const ambC = parseFloat(sliderAmbientTemp.value);

    // Standard atmospheric barometric lapse calculations
    const densityRatio = Math.pow(Math.max(0.01, 1 - 2.25577e-5 * altFt), 4.25588);
    const airDensity = (1.225 * densityRatio).toFixed(3);
    const pressureInHg = (29.92 * Math.pow(Math.max(0.01, 1 - 6.8756e-6 * altFt), 5.2559)).toFixed(2);

    valAltitude.textContent = `${altFt.toLocaleString()} ft`;
    envDensity.textContent = `Air Density: ${airDensity} kg/m³`;
    envPressure.textContent = `Pressure: ${pressureInHg} inHg`;

    valAmbientTemp.textContent = `${ambC > 0 ? '+' : ''}${ambC.toFixed(1)} °C`;
    const coolingFactor = Math.max(0.35, (1.0 + (15.0 - ambC) * 0.012) * Math.sqrt(densityRatio)).toFixed(2);
    envConvection.textContent = `Cooling Factor: ${coolingFactor}x`;

    let condition = 'ISA Standard Day';
    if (ambC > 35) condition = 'High Thermal Heat Soak';
    else if (ambC < -10) condition = 'Sub-Zero Freezing Airframe';
    if (altFt > 15000) condition += ' (Hypobaric)';
    envCondition.textContent = `Condition: ${condition}`;

    // Immediate dynamic physical response across all engine subsystems
    const tempDelta = ambC - 15.0;
    const coolingLoss = (1.0 - parseFloat(coolingFactor)) * 35.0;

    const estCht = Math.max(105.0, 148.6 + tempDelta * 0.75 + coolingLoss);
    const estEgt = Math.max(500.0, 667.7 + tempDelta * 0.45 + coolingLoss * 0.35);
    const altBuffet = (altFt / 10000.0) * 0.28;
    const thermalStressVib = Math.max(0.0, (estCht - 148.0) * 0.014);
    const estVib = Math.max(0.9, 1.41 + altBuffet + thermalStressVib);
    const estOilTemp = Math.max(60.0, 85.0 + tempDelta * 0.45 + coolingLoss * 0.25);
    const estOilP = Math.max(110.0, 281.8 - (estOilTemp - 85.0) * 1.3 - (altFt / 10000.0) * 8.0);
    const estMap = Math.max(35.0, 101.3 * densityRatio);
    const estFuelFlow = Math.max(4.0, 8.30 * (estMap / 92.0));

    // 1. Immediately update 3D WebGL engine & Pinned 3D Callouts
    engine3D.updateEnvironment(altFt, ambC);

    // 2. Immediately update telemetry gauges on the left
    updateGauge('cht', estCht);
    updateGauge('egt', estEgt);
    updateGauge('vibration_rms', estVib);
    updateGauge('oil_temp', estOilTemp);
    updateGauge('oil_pressure', estOilP);
    updateGauge('fuel_flow', estFuelFlow);

    // 3. Immediately calculate and update Subassembly Health
    const cylHealth = Math.max(0.05, Math.min(1.0, 1.0 - Math.max(0.0, estCht - 150.0) / 65.0));
    const crankHealth = Math.max(0.10, Math.min(1.0, 1.0 - Math.max(0.0, estVib - 1.2) / 2.3));
    const lubeHealth = Math.max(0.10, Math.min(1.0, (estOilP - 130.0) / 180.0));
    const exhaustHealth = Math.max(0.10, Math.min(1.0, 1.0 - Math.max(0.0, estEgt - 650.0) / 200.0));

    updateSubassemblyHealth({
      cylinder_head: cylHealth,
      crankshaft: crankHealth,
      lubrication_system: lubeHealth,
      exhaust_manifold: exhaustHealth,
    }, {
      cht: estCht,
      egt: estEgt,
      vibration_rms: estVib,
      oil_temp: estOilTemp,
      oil_pressure: estOilP,
    });

    // 4. Update dynamic physical Arrhenius RUL & flight sustainment
    let envRul = 485.0;
    if (estCht > 155.0) {
      const excess = (estCht - 155.0) / 22.0;
      envRul *= Math.max(0.028, Math.exp(-excess * 0.95));
    }
    if (estOilP < 220.0) {
      const pLoss = Math.max(0.0, 220.0 - estOilP) / 120.0;
      envRul *= Math.max(0.045, 1.0 - pLoss * 0.92);
    }
    if (estVib > 2.2) {
      const vExcess = Math.max(0.0, estVib - 2.2) / 1.5;
      envRul *= Math.max(0.05, 1.0 - vExcess * 0.88);
    }
    envRul = Math.max(12.0, Math.min(520.0, envRul));
    const envStatus = envRul < 45 ? 'CRITICAL_RTB' : (envRul < 200 ? 'ELEVATED_WEAR' : 'OPTIMAL');
    const isCompliant = envRul >= 200;
    updateRulDisplay(envRul, envRul < 50 ? 35.0 : 0.0, isCompliant, null, envStatus);

    // 5. Send environment update to server backend (debounced 50ms)
    clearTimeout(envDebounceTimer);
    envDebounceTimer = setTimeout(() => {
      fetch('/api/simulate/environment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ altitude: altFt, ambient_temp: ambC })
      }).catch(err => console.error('Environment sync error:', err));
    }, 50);
  }

  if (sliderAltitude) sliderAltitude.addEventListener('input', updateEnvironmentalState);
  if (sliderAmbientTemp) sliderAmbientTemp.addEventListener('input', updateEnvironmentalState);

  // Helper for smooth linear interpolation
  function lerp(a, b, t) {
    return a + (b - a) * Math.max(0, Math.min(1, t));
  }

  // Continuous, realistic mission profile keyframes (STANAG 4586 Sortie Replay)
  const MISSION_KEYFRAMES = [
    { t: 0,    rpm: 1900, cht: 102.0, egt: 510.0, oil_p: 345, oil_t: 72, vib: 0.90, ff: 4.5, afr: 13.8, rul: 495, ext: 0, arch: 'nominal', conf: 0.99 },
    { t: 300,  rpm: 2350, cht: 118.0, egt: 560.0, oil_p: 338, oil_t: 79, vib: 1.08, ff: 5.8, afr: 13.8, rul: 490, ext: 0, arch: 'nominal', conf: 0.98 },
    { t: 900,  rpm: 5450, cht: 166.0, egt: 715.0, oil_p: 310, oil_t: 93, vib: 1.82, ff: 15.2, afr: 13.5, rul: 450, ext: 0, arch: 'nominal', conf: 0.96 },
    { t: 1500, rpm: 4800, cht: 148.9, egt: 666.1, oil_p: 320, oil_t: 86, vib: 1.20, ff: 9.5, afr: 13.8, rul: 420, ext: 0, arch: 'nominal', conf: 0.97 },
    { t: 4200, rpm: 4800, cht: 151.5, egt: 671.0, oil_p: 318, oil_t: 88, vib: 1.24, ff: 9.6, afr: 13.8, rul: 380, ext: 0, arch: 'nominal', conf: 0.95 },
    { t: 4600, rpm: 5200, cht: 214.5, egt: 792.0, oil_p: 285, oil_t: 112, vib: 2.45, ff: 11.2, afr: 15.2, rul: 18, ext: 0, arch: 'thermal_runaway', conf: 0.93 },
    { t: 5100, rpm: 5050, cht: 202.0, egt: 775.0, oil_p: 282, oil_t: 114, vib: 2.25, ff: 10.6, afr: 14.6, rul: 14, ext: 0, arch: 'thermal_runaway', conf: 0.94 },
    { t: 5400, rpm: 4450, cht: 172.0, egt: 680.0, oil_p: 310, oil_t: 95, vib: 1.55, ff: 8.8, afr: 12.4, rul: 135, ext: 40, arch: 'thermal_runaway', conf: 0.90 },
    { t: 5800, rpm: 4380, cht: 166.0, egt: 662.0, oil_p: 315, oil_t: 89, vib: 1.42, ff: 8.2, afr: 12.6, rul: 155, ext: 40, arch: 'thermal_runaway', conf: 0.88 },
    { t: 6300, rpm: 2050, cht: 110.0, egt: 505.0, oil_p: 330, oil_t: 76, vib: 0.88, ff: 4.2, afr: 13.8, rul: 148, ext: 40, arch: 'nominal', conf: 0.97 }
  ];

  function getInterpolatedMissionData(sec) {
    let k0 = MISSION_KEYFRAMES[0];
    let k1 = MISSION_KEYFRAMES[MISSION_KEYFRAMES.length - 1];

    for (let i = 0; i < MISSION_KEYFRAMES.length - 1; i++) {
      if (sec >= MISSION_KEYFRAMES[i].t && sec <= MISSION_KEYFRAMES[i + 1].t) {
        k0 = MISSION_KEYFRAMES[i];
        k1 = MISSION_KEYFRAMES[i + 1];
        break;
      }
    }

    const span = Math.max(1, k1.t - k0.t);
    const progress = Math.max(0, Math.min(1, (sec - k0.t) / span));

    // Smooth hermite s-curve for organic aesthetic transitions
    const t = progress * progress * (3 - 2 * progress);

    return {
      telemetry: {
        rpm: lerp(k0.rpm, k1.rpm, t),
        cht: lerp(k0.cht, k1.cht, t),
        egt: lerp(k0.egt, k1.egt, t),
        oil_pressure: lerp(k0.oil_p, k1.oil_p, t),
        oil_temp: lerp(k0.oil_t, k1.oil_t, t),
        vibration_rms: lerp(k0.vib, k1.vib, t),
        fuel_flow: lerp(k0.ff, k1.ff, t),
        afr: lerp(k0.afr, k1.afr, t),
      },
      rul_cycles: Math.round(lerp(k0.rul, k1.rul, t)),
      adjusted_rul: Math.round(lerp(k0.rul, k1.rul, t) + lerp(k0.ext, k1.ext, t)),
      extension_cycles: Math.round(lerp(k0.ext, k1.ext, t)),
      fault_archetype: t > 0.5 ? k1.arch : k0.arch,
      fault_confidence: lerp(k0.conf, k1.conf, t)
    };
  }

  function applyMissionTimelineStep(sec) {
    replaySeconds = Math.max(0, Math.min(6300, sec));
    if (replayTimeline) replayTimeline.value = replaySeconds;

    const timeStr = formatTime(replaySeconds);
    if (replayClock) replayClock.textContent = `T+${timeStr}`;
    if (scrubberCurrentTime) scrubberCurrentTime.textContent = timeStr;

    // Highlight active phase pill
    phasePills.forEach(pill => {
      const pTime = parseInt(pill.getAttribute('data-time'), 10);
      pill.classList.remove('active');
      if (Math.abs(replaySeconds - pTime) < 450) {
        pill.classList.add('active');
      }
    });

    // Smoothly calculate interpolated mission telemetry
    const simData = getInterpolatedMissionData(replaySeconds);

    // Update RUL & Sustain Window for current replay phase
    let timelineStatus = 'OPTIMAL';
    let timelineStr = null;
    let timelineValid = true;
    if (simData.rul_cycles <= 35) {
      timelineStatus = 'CRITICAL_RTB';
      timelineValid = false;
      timelineStr = '⚠️ 1h 24m EMERGENCY RTB WINDOW';
    } else if (simData.rul_cycles <= 180) {
      timelineStatus = 'ELEVATED_WEAR';
      timelineStr = '17h 30m Protected Loiter';
    } else if (simData.rul_cycles <= 420) {
      timelineStr = '42h 00m Mission Endurance';
    } else {
      timelineStr = '48h 30m Mission Endurance';
    }

    updateRulDisplay(
      simData.adjusted_rul || simData.rul_cycles,
      simData.extension_cycles || 0,
      timelineValid,
      timelineStr,
      timelineStatus
    );

    engine3D.updateTelemetryState(simData);
    for (const [s, val] of Object.entries(simData.telemetry)) {
      updateGauge(s, val);
    }

    const cylHealth = Math.max(0.05, Math.min(1.0, 1.0 - Math.max(0.0, simData.telemetry.cht - 150.0) / 65.0));
    const crankHealth = Math.max(0.10, Math.min(1.0, 1.0 - Math.max(0.0, simData.telemetry.vibration_rms - 1.2) / 2.3));
    const lubeHealth = Math.max(0.10, Math.min(1.0, (simData.telemetry.oil_pressure - 130.0) / 180.0));
    const exhaustHealth = Math.max(0.10, Math.min(1.0, 1.0 - Math.max(0.0, simData.telemetry.egt - 650.0) / 200.0));

    updateSubassemblyHealth({
      cylinder_head: cylHealth,
      crankshaft: crankHealth,
      lubrication_system: lubeHealth,
      exhaust_manifold: exhaustHealth,
    }, simData.telemetry);
  }

  if (replayTimeline) {
    replayTimeline.addEventListener('input', (e) => {
      applyMissionTimelineStep(parseInt(e.target.value, 10));
    });
  }

  if (btnReplayPlay) {
    btnReplayPlay.addEventListener('click', () => {
      isReplayPlaying = !isReplayPlaying;
      if (isReplayPlaying) {
        btnReplayPlay.textContent = '⏸ PAUSE';
        btnReplayPlay.classList.add('active');
        // Calm, readable update rate: ticks every 200ms with smooth progression
        replayInterval = setInterval(() => {
          const stepSize = Math.max(1, Math.round(2 * replaySpeed));
          applyMissionTimelineStep(replaySeconds + stepSize);
          if (replaySeconds >= 6300) {
            isReplayPlaying = false;
            btnReplayPlay.textContent = '▶ PLAY';
            btnReplayPlay.classList.remove('active');
            clearInterval(replayInterval);
          }
        }, 200);
      } else {
        btnReplayPlay.textContent = '▶ PLAY';
        btnReplayPlay.classList.remove('active');
        if (replayInterval) clearInterval(replayInterval);
      }
    });
  }

  if (btnReplayStepBack) {
    btnReplayStepBack.addEventListener('click', () => {
      applyMissionTimelineStep(replaySeconds - 30);
    });
  }

  if (btnReplayStepFwd) {
    btnReplayStepFwd.addEventListener('click', () => {
      applyMissionTimelineStep(replaySeconds + 30);
    });
  }

  if (selectReplaySpeed) {
    selectReplaySpeed.addEventListener('change', (e) => {
      replaySpeed = parseFloat(e.target.value);
    });
  }

  phasePills.forEach(pill => {
    pill.addEventListener('click', () => {
      const targetSec = parseInt(pill.getAttribute('data-time'), 10);
      applyMissionTimelineStep(targetSec);
    });
  });

  // Setup Fault Injection Buttons
  const injectBtns = document.querySelectorAll('.inject-btn');
  injectBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const faultType = btn.getAttribute('data-fault');
      injectBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      try {
        await fetch('/api/simulate/inject', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fault_type: faultType || null })
        });
      } catch (err) {
        console.error('Fault injection error:', err);
      }
    });
  });

  // Benchmarks Modal
  btnBenchmarks.addEventListener('click', async () => {
    modalBackdrop.classList.add('open');
    try {
      const res = await fetch('/api/benchmarks');
      const data = await res.json();
      benchmarkTbody.innerHTML = data.metrics.map(m => `
        <tr>
          <td><strong>${m.metric}</strong></td>
          <td>${m.traditional_dl}</td>
          <td>${m.modular_dt}</td>
          <td class="highlight-col">${m.agentic_pinn_drl}</td>
          <td><span class="gain-badge">${m.improvement}</span></td>
        </tr>
      `).join('');
    } catch (e) {
      console.error(e);
    }
  });

  modalClose.addEventListener('click', () => {
    modalBackdrop.classList.remove('open');
  });

  modalBackdrop.addEventListener('click', (e) => {
    if (e.target === modalBackdrop) {
      modalBackdrop.classList.remove('open');
    }
  });

  // Architecture Modal Open & Close
  if (btnArchitecture && modalArchitecture) {
    btnArchitecture.addEventListener('click', () => {
      modalArchitecture.classList.add('open');
    });

    if (modalArchClose) {
      modalArchClose.addEventListener('click', () => {
        modalArchitecture.classList.remove('open');
      });
    }

    modalArchitecture.addEventListener('click', (e) => {
      if (e.target === modalArchitecture) {
        modalArchitecture.classList.remove('open');
      }
    });

    // Tab Switching Logic
    archTabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');
        archTabBtns.forEach(b => b.classList.remove('active'));
        archTabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetPane = document.getElementById(tabId);
        if (targetPane) targetPane.classList.add('active');
      });
    });

    // Slide 3 Subtab Diagram Switching (DAG vs Parameters vs Full Slide)
    const btnViewDag = document.getElementById('btn-view-dag');
    const btnViewPipeline = document.getElementById('btn-view-pipeline');
    const btnViewSlide = document.getElementById('btn-view-slide');
    const slide3Img = document.getElementById('slide3-img');

    if (btnViewDag && btnViewPipeline && btnViewSlide && slide3Img) {
      const subtabBtns = [btnViewDag, btnViewPipeline, btnViewSlide];

      btnViewDag.addEventListener('click', () => {
        subtabBtns.forEach(b => b.classList.remove('active'));
        btnViewDag.classList.add('active');
        slide3Img.src = '/static/assets/orchestrator_routing_dag.svg';
        slide3Img.alt = 'IdeaForge AI Agent Orchestrator Routing DAG';
      });

      btnViewPipeline.addEventListener('click', () => {
        subtabBtns.forEach(b => b.classList.remove('active'));
        btnViewPipeline.classList.add('active');
        slide3Img.src = '/static/assets/layer_parameter_pipeline.svg';
        slide3Img.alt = 'IdeaForge Layer-by-Layer Parameters and Baseline Paper Comparison';
      });

      btnViewSlide.addEventListener('click', () => {
        subtabBtns.forEach(b => b.classList.remove('active'));
        btnViewSlide.classList.add('active');
        slide3Img.src = '/static/assets/ai_agent_workflow.jpg';
        slide3Img.alt = 'IdeaForge Slide 3 Full PPT Slide';
      });
    }
  }

  // Telemetry Gauge Thresholds
  const GAUGE_CONFIG = {
    rpm: { min: 3000, max: 6000, warn: 5200, crit: 5600, unit: 'RPM' },
    cht: { min: 80, max: 240, warn: 185, crit: 210, unit: '°C' },
    egt: { min: 450, max: 880, warn: 740, crit: 800, unit: '°C' },
    oil_pressure: { min: 100, max: 450, warn: 220, crit: 180, inverse: true, unit: 'kPa' },
    oil_temp: { min: 50, max: 150, warn: 115, crit: 130, unit: '°C' },
    vibration_rms: { min: 0.5, max: 4.5, warn: 2.6, crit: 3.5, unit: 'g' },
    fuel_flow: { min: 4.0, max: 18.0, warn: 14.0, crit: 16.5, unit: 'L/h' },
    afr: { min: 10.5, max: 16.5, warn: 15.0, crit: 15.8, unit: ':1' }
  };

  function updateGauge(sensor, value) {
    const cfg = GAUGE_CONFIG[sensor];
    if (!cfg) return;

    const valEl = document.getElementById(`val-${sensor}`);
    const barEl = document.getElementById(`bar-${sensor}`);

    if (valEl) valEl.textContent = `${value.toFixed(sensor === 'vibration_rms' || sensor === 'fuel_flow' || sensor === 'afr' ? 2 : 1)} ${cfg.unit}`;

    if (barEl) {
      const pct = Math.max(0, Math.min(100, ((value - cfg.min) / (cfg.max - cfg.min)) * 100));
      barEl.style.width = `${pct}%`;

      barEl.classList.remove('warning', 'critical');
      if (cfg.inverse) {
        if (value <= cfg.crit) barEl.classList.add('critical');
        else if (value <= cfg.warn) barEl.classList.add('warning');
      } else {
        if (value >= cfg.crit) barEl.classList.add('critical');
        else if (value >= cfg.warn) barEl.classList.add('warning');
      }
    }
  }

  // WebSocket Live Streaming
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('[Telemetry WS] Connected');
      wsStatus.textContent = '10 Hz SYNCED';
      wsStatus.style.color = 'var(--accent-emerald)';
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // 1. Update 3D viewport
        engine3D.updateTelemetryState(data);
        if (window._engine3DExtension) {
          window._engine3DExtension.onPayload(data);
        }

        // 2. Update Gauges
        if (data.telemetry) {
          for (const [s, val] of Object.entries(data.telemetry)) {
            updateGauge(s, val);
          }
        }

        // 3. Update PINN Health Card
        updateRulDisplay(
          data.adjusted_rul !== undefined ? data.adjusted_rul : data.rul_cycles,
          data.extension_cycles,
          data.is_physically_valid,
          data.sustain_flight_str,
          data.mission_status
        );
        if (data.fourier_residual !== undefined && fourierResidual) {
          fourierResidual.textContent = `${data.fourier_residual.toFixed(3)} °C`;
        }
        if (data.physical_gradient !== undefined && physicalGradient) {
          physicalGradient.textContent = `${data.physical_gradient.toFixed(3)} °C/cyc`;
        }
        if (fourierAdherence) {
          fourierAdherence.textContent = data.is_physically_valid ? '100% COMPLIANT' : 'BOUNDARY DRIFT';
          fourierAdherence.style.color = data.is_physically_valid ? 'var(--accent-emerald)' : 'var(--accent-rose)';
        }

        // 4. Update Fault Archetype Classifier
        if (data.fault_archetype && faultName) {
          faultName.textContent = data.fault_archetype.toUpperCase().replace('_', ' ');
          faultConf.textContent = `${Math.round((data.fault_confidence || 0.85) * 100)}% CONF`;

          // Update probability bars
          if (data.fault_probabilities) {
            for (const [arch, prob] of Object.entries(data.fault_probabilities)) {
              const bar = document.getElementById(`fault-bar-${arch}`);
              const pct = document.getElementById(`fault-pct-${arch}`);
              if (bar) bar.style.width = `${Math.round(prob * 100)}%`;
              if (pct) pct.textContent = `${Math.round(prob * 100)}%`;
            }
          }
        }

        // 5. Update DRL Prognostic Actions
        if (data.drl_action && drlText) {
          drlText.textContent = data.drl_action.recommendation || 'Nominal cruise envelope';
          if (drlShieldBadge) {
            drlShieldBadge.style.display = data.drl_action.shield_applied ? 'inline-flex' : 'none';
          }
        }

        // 6. Sensor Auditor Auto-Recovery Counter
        if (data.sensor_audit && auditRecoveries) {
          auditRecoveries.textContent = data.sensor_audit.total_corrections || 0;
        }

        // 7. Component Health Matrix
        if (data.component_health) {
          updateSubassemblyHealth(data.component_health, data.telemetry);
        }

      } catch (err) {
        console.error('Telemetry processing error:', err);
      }
    };

    socket.onclose = () => {
      console.log('[Telemetry WS] Closed, reconnecting in 2s...');
      wsStatus.textContent = 'DISCONNECTED';
      wsStatus.style.color = 'var(--accent-rose)';
      setTimeout(connectWebSocket, 2000);
    };

    socket.onerror = (err) => {
      console.error('[Telemetry WS] Error:', err);
      socket.close();
    };
  }

  connectWebSocket();
});

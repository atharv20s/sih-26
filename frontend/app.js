/**
 * SIH26054 Digital Twin Front-End Dashboard Controller
 * NASA Open MCT–Grade Operational Mission Control Ground Station
 *
 * Coordinates:
 *   - Real-time 10 Hz telemetry ingestion into TelemetryStore typed ring buffers
 *   - Monitored Parameter Rail with tabular monospace numbers, trends, & canvas sparklines
 *   - Multi-Channel Strip Charts with synchronized time axes & certified limit bands
 *   - NASA Open MCT Time Conductor (LIVE vs FIXED mode, scrubbing 3D twin & health state)
 *   - Master Caution & Warning Annunciator banner with ACK latching
 *   - Dedicated Event Log recording state transitions & threshold cross-overs
 *   - Three.js 3D viewport synchronization with native X/Y/Z cross-section clipping
 *   - PINN thermodynamic compliance, RUL prognostics, & DRL safety policy displays
 *   - Fault injection testbed & Presentation modals
 */

import {
  LIMITS,
  STATE,
  STATE_NAMES,
  STATE_COLORS,
  telemetryStore,
  eventLog,
  evaluateAlarm,
} from './telemetry_store.js';

document.addEventListener('DOMContentLoaded', () => {
  // -------------------------------------------------------------------------
  // 1. Initialize 3D Digital Twin Engine
  // -------------------------------------------------------------------------
  const engine3D = new AeroEngine3D('webgl-canvas');
  window._engine3D = engine3D;
  window.dispatchEvent(new CustomEvent('engine3dReady', { detail: engine3D }));

  // -------------------------------------------------------------------------
  // 2. Mission Elapsed Time & Clock
  // -------------------------------------------------------------------------
  let missionSeconds = 2535; // Default: T+00:42:15
  let lastTimestampSec = 0;
  const missionClockEl = document.getElementById('mission-t-clock');

  function formatTime(totalSec) {
    const s = Math.max(0, Math.floor(totalSec));
    const h = Math.floor(s / 3600).toString().padStart(2, '0');
    const m = Math.floor((s % 3600) / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `${h}:${m}:${sec}`;
  }

  // -------------------------------------------------------------------------
  // 3. Master Caution & Warning Annunciator System
  // -------------------------------------------------------------------------
  const annunciatorCaution = document.getElementById('annunciator-caution');
  const annunciatorWarning = document.getElementById('annunciator-warning');
  const btnAckAlarm = document.getElementById('btn-ack-alarm');
  let alarmsAcknowledged = false;

  function updateAnnunciators() {
    const highestSev = eventLog.getHighestActiveSeverity();
    const hasUnack = eventLog.hasUnacknowledgedAlarm();

    if (annunciatorCaution) {
      annunciatorCaution.classList.toggle('caution-active', highestSev === STATE.CAUTION || highestSev >= STATE.WARNING);
    }
    if (annunciatorWarning) {
      annunciatorWarning.classList.toggle('warning-active', highestSev >= STATE.WARNING);
    }
    if (btnAckAlarm) {
      btnAckAlarm.classList.toggle('needs-ack', hasUnack && highestSev >= STATE.CAUTION);
    }
  }

  if (btnAckAlarm) {
    btnAckAlarm.addEventListener('click', () => {
      eventLog.acknowledgeAll();
      alarmsAcknowledged = true;
      btnAckAlarm.classList.remove('needs-ack');
    });
  }

  // -------------------------------------------------------------------------
  // 4. Dedicated Event Log (NASA Open MCT Standard)
  // -------------------------------------------------------------------------
  const eventLogContainer = document.getElementById('event-log-container');
  const eventCountBadge = document.getElementById('event-count-badge');
  let hasLoggedFirstEvent = false;

  eventLog.onEvent((entry) => {
    if (!eventLogContainer) return;

    if (!hasLoggedFirstEvent) {
      eventLogContainer.innerHTML = '';
      hasLoggedFirstEvent = true;
    }

    const row = document.createElement('div');
    const sevClass = entry.severity >= 3 ? 'ev-critical' : (entry.severity >= 1 ? 'ev-caution' : 'ev-nominal');
    const badgeClass = entry.severity >= 3 ? 'critical' : (entry.severity >= 1 ? 'caution' : 'nominal');

    row.className = `event-log-entry ${sevClass}`;
    row.innerHTML = `
      <span class="event-time-tag">${formatTime(entry.time)}</span>
      <span class="event-badge ${badgeClass}">${entry.stateName}</span>
      <span class="event-text" title="${entry.text}">${entry.text}</span>
    `;

    eventLogContainer.insertBefore(row, eventLogContainer.firstChild);

    // Keep max 80 rows in DOM
    while (eventLogContainer.children.length > 80) {
      eventLogContainer.removeChild(eventLogContainer.lastChild);
    }

    if (eventCountBadge) {
      eventCountBadge.textContent = `${eventLog.events.length} EVENTS`;
    }

    updateAnnunciators();
  });

  // -------------------------------------------------------------------------
  // 5. Monitored Parameter Rail (Left Column)
  // -------------------------------------------------------------------------
  const paramRailContainer = document.getElementById('param-rail-container');
  const paramRailCache = new Map();

  const PARAM_CHANNELS = [
    'cht',
    'egt',
    'vibration_rms',
    'oil_pressure',
    'oil_temp',
    'rpm',
    'fuel_flow',
    'map',
    'afr',
    'coolant_temp',
  ];

  function buildParameterRail() {
    if (!paramRailContainer) return;
    paramRailContainer.innerHTML = '';

    PARAM_CHANNELS.forEach((key) => {
      const lim = LIMITS[key];
      if (!lim) return;

      const row = document.createElement('div');
      row.className = 'param-row';
      row.id = `param-row-${key}`;

      row.innerHTML = `
        <div class="param-label-wrap">
          <span class="param-label" title="${lim.description || lim.label}">
            ${lim.label}
            <span class="param-imp-tag" id="imp-tag-${key}" style="display:none;">IMP</span>
          </span>
          <span class="param-subtag">${key.toUpperCase()}</span>
        </div>
        <div style="display:flex; align-items:baseline; justify-content:flex-end; gap:3px;">
          <span class="param-val tnum" id="param-val-${key}">--</span>
          <span class="param-unit">${lim.unit}</span>
        </div>
        <div class="param-trend flat" id="param-trend-${key}">― 0.0</div>
        <div class="param-sparkline-wrap">
          <canvas class="param-sparkline-canvas" id="param-spark-${key}" width="48" height="18"></canvas>
        </div>
        <div class="param-bar-wrap">
          <div class="param-bar-bg">
            <div class="param-bar-fill" id="param-fill-${key}" style="width:50%;"></div>
            <div class="param-bar-marker" id="param-mark-${key}"></div>
          </div>
        </div>
        <div class="param-state-chip nominal" id="param-chip-${key}">NOM</div>
      `;

      paramRailContainer.appendChild(row);

      const sparkCanvas = document.getElementById(`param-spark-${key}`);
      const sparkCtx = sparkCanvas ? sparkCanvas.getContext('2d') : null;

      paramRailCache.set(key, {
        rowEl: row,
        valEl: document.getElementById(`param-val-${key}`),
        impTagEl: document.getElementById(`imp-tag-${key}`),
        trendEl: document.getElementById(`param-trend-${key}`),
        sparkCanvas,
        sparkCtx,
        fillEl: document.getElementById(`param-fill-${key}`),
        markEl: document.getElementById(`param-mark-${key}`),
        chipEl: document.getElementById(`param-chip-${key}`),
      });

      // Position redline marker on mini bar
      if (lim.redHigh !== undefined) {
        const markPct = Math.max(0, Math.min(100, ((lim.redHigh - lim.displayMin) / (lim.displayMax - lim.displayMin)) * 100));
        const markEl = document.getElementById(`param-mark-${key}`);
        if (markEl) markEl.style.left = `${markPct}%`;
      } else if (lim.redLow !== undefined) {
        const markPct = Math.max(0, Math.min(100, ((lim.redLow - lim.displayMin) / (lim.displayMax - lim.displayMin)) * 100));
        const markEl = document.getElementById(`param-mark-${key}`);
        if (markEl) markEl.style.left = `${markPct}%`;
      }
    });
  }

  buildParameterRail();

  function drawSparkline(ctx, width, height, pts, state) {
    if (!ctx || pts.length < 2) return;
    ctx.clearRect(0, 0, width, height);

    let min = pts[0][1];
    let max = pts[0][1];
    for (let i = 1; i < pts.length; i++) {
      const v = pts[i][1];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min > 0.001 ? max - min : 1.0;

    const strokeColor = state >= 3 ? '#f85149' : (state >= 1 ? '#d29922' : '#38bdf8');

    ctx.beginPath();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = strokeColor;

    for (let i = 0; i < pts.length; i++) {
      const x = (i / (pts.length - 1)) * width;
      const y = height - ((pts[i][1] - min) / range) * (height - 3) - 1.5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  let lastSparklineRender = 0;

  function updateParameterRail(telemetryObj, imputedList = []) {
    if (!telemetryObj) return;

    const nowSec = Date.now() / 1000;
    const shouldRenderSparks = (nowSec - lastSparklineRender) >= 0.2; // 5 Hz sparkline updates

    PARAM_CHANNELS.forEach((key) => {
      const lim = LIMITS[key];
      const refs = paramRailCache.get(key);
      if (!lim || !refs) return;

      const rawVal = telemetryObj[key];
      if (rawVal === undefined || Number.isNaN(rawVal)) return;
      const val = typeof rawVal === 'number' ? rawVal : parseFloat(rawVal);

      const state = evaluateAlarm(key, val);

      // 1. Monospace Tabular Value
      refs.valEl.textContent = val.toFixed(lim.decimals);
      const isImputed = imputedList.includes(key);
      refs.valEl.classList.toggle('imputed', isImputed);
      if (refs.impTagEl) refs.impTagEl.style.display = isImputed ? 'inline-block' : 'none';

      // 2. Trend rate (10-second slope)
      const chBuf = telemetryStore.getChannel(key);
      const rate = chBuf.trend(10);
      const absRate = Math.abs(rate);
      const sign = rate > 0.05 ? '▲ +' : (rate < -0.05 ? '▼ -' : '― ');
      refs.trendEl.textContent = `${sign}${absRate.toFixed(1)}`;
      refs.trendEl.className = 'param-trend ' + (rate > 0.05 ? 'up' : (rate < -0.05 ? 'down' : 'flat'));

      // 3. Mini Limit Bar
      const pct = Math.max(0, Math.min(100, ((val - lim.displayMin) / (lim.displayMax - lim.displayMin)) * 100));
      refs.fillEl.style.width = `${pct}%`;
      if (state >= 3) {
        refs.fillEl.style.backgroundColor = '#f85149';
      } else if (state >= 1) {
        refs.fillEl.style.backgroundColor = '#d29922';
      } else {
        refs.fillEl.style.backgroundColor = '#3fb950';
      }

      // 4. State Chip & Row Highlighting
      refs.rowEl.classList.remove('state-caution', 'state-critical');
      refs.chipEl.className = 'param-state-chip';

      if (state >= 3) {
        refs.rowEl.classList.add('state-critical');
        refs.chipEl.classList.add('critical');
        refs.chipEl.textContent = 'CRIT';
      } else if (state >= 1) {
        refs.rowEl.classList.add('state-caution');
        refs.chipEl.classList.add('caution');
        refs.chipEl.textContent = 'CAUT';
      } else {
        refs.chipEl.classList.add('nominal');
        refs.chipEl.textContent = 'NOM';
      }

      // 5. Sparkline
      if (shouldRenderSparks && refs.sparkCtx) {
        const historyPts = chBuf.range(chBuf.latestTime() - 30, chBuf.latestTime());
        drawSparkline(refs.sparkCtx, 48, 18, historyPts, state);
      }
    });

    if (shouldRenderSparks) {
      lastSparklineRender = nowSec;
    }
  }

  // -------------------------------------------------------------------------
  // 6. Multi-Channel Strip Charts (NASA Open MCT Standard)
  // -------------------------------------------------------------------------
  const stripCanvas = document.getElementById('strip-chart-canvas');
  const stripCtx = stripCanvas ? stripCanvas.getContext('2d') : null;
  const stripValCht = document.getElementById('strip-val-cht');
  const stripValEgt = document.getElementById('strip-val-egt');
  const stripValVib = document.getElementById('strip-val-vib');
  const stripValOil = document.getElementById('strip-val-oil');

  const STRIP_TRACKS = [
    { key: 'cht', label: 'CHT', color: '#f59e0b', min: 100, max: 230, warn: 185, crit: 210, unit: '°C' },
    { key: 'egt', label: 'EGT', color: '#f97316', min: 500, max: 850, warn: 740, crit: 800, unit: '°C' },
    { key: 'vibration_rms', label: 'VIB', color: '#c084fc', min: 0.0, max: 4.5, warn: 2.5, crit: 3.5, unit: 'g' },
    { key: 'oil_pressure', label: 'OIL P', color: '#38bdf8', min: 120, max: 420, warn: 220, crit: 180, inverse: true, unit: 'kPa' },
  ];

  function resizeStripCanvas() {
    if (!stripCanvas) return;
    const rect = stripCanvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (stripCanvas.width !== Math.floor(rect.width * dpr) || stripCanvas.height !== Math.floor(rect.height * dpr)) {
      stripCanvas.width = Math.floor(rect.width * dpr);
      stripCanvas.height = Math.floor(rect.height * dpr);
    }
  }

  function renderStripCharts(targetTime = null) {
    if (!stripCtx || !stripCanvas) return;
    resizeStripCanvas();

    const w = stripCanvas.width;
    const h = stripCanvas.height;
    if (w < 10 || h < 10) return;

    stripCtx.clearRect(0, 0, w, h);

    // Dark slate background
    stripCtx.fillStyle = '#07090e';
    stripCtx.fillRect(0, 0, w, h);

    const now = targetTime !== null ? targetTime : telemetryStore.currentMissionTime;
    const windowSec = 120.0;
    const tStart = Math.max(0, now - windowSec);
    const tEnd = now;

    // Draw vertical time grid lines (every 30 seconds)
    stripCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    stripCtx.lineWidth = 1;
    stripCtx.fillStyle = '#485464';
    stripCtx.font = '16px JetBrains Mono, monospace';

    for (let secOffset = 0; secOffset <= windowSec; secOffset += 30) {
      const x = (secOffset / windowSec) * w;
      stripCtx.beginPath();
      stripCtx.moveTo(x, 0);
      stripCtx.lineTo(x, h);
      stripCtx.stroke();

      const tag = secOffset === windowSec ? 'NOW' : `-${Math.round(windowSec - secOffset)}s`;
      stripCtx.fillText(tag, x + 4, h - 6);
    }

    const trackH = h / STRIP_TRACKS.length;

    STRIP_TRACKS.forEach((trk, idx) => {
      const topY = idx * trackH;
      const botY = topY + trackH;

      // Track separator line
      if (idx > 0) {
        stripCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        stripCtx.beginPath();
        stripCtx.moveTo(0, topY);
        stripCtx.lineTo(w, topY);
        stripCtx.stroke();
      }

      // Draw dashed caution/critical threshold line
      const trkRange = trk.max - trk.min;
      const warnY = botY - ((trk.warn - trk.min) / trkRange) * trackH;
      const critY = botY - ((trk.crit - trk.min) / trkRange) * trackH;

      stripCtx.setLineDash([4, 4]);
      stripCtx.lineWidth = 1;
      stripCtx.strokeStyle = 'rgba(210, 153, 34, 0.45)';
      stripCtx.beginPath();
      stripCtx.moveTo(0, warnY);
      stripCtx.lineTo(w, warnY);
      stripCtx.stroke();

      stripCtx.strokeStyle = 'rgba(248, 81, 73, 0.55)';
      stripCtx.beginPath();
      stripCtx.moveTo(0, critY);
      stripCtx.lineTo(w, critY);
      stripCtx.stroke();
      stripCtx.setLineDash([]); // Reset dash

      // Fetch points from ChannelBuffer
      const buf = telemetryStore.getChannel(trk.key);
      const pts = buf.range(tStart, tEnd);

      if (pts.length >= 2) {
        stripCtx.beginPath();
        stripCtx.lineWidth = 1.6;
        stripCtx.strokeStyle = trk.color;

        for (let i = 0; i < pts.length; i++) {
          const [t, val] = pts[i];
          const x = ((t - tStart) / (tEnd - tStart || 1)) * w;
          const clampedVal = Math.max(trk.min, Math.min(trk.max, val));
          const y = botY - ((clampedVal - trk.min) / trkRange) * trackH;

          if (i === 0) stripCtx.moveTo(x, y);
          else stripCtx.lineTo(x, y);
        }
        stripCtx.stroke();
      }
    });

    // Update legend readout values
    const latestCht = telemetryStore.getChannel('cht').latest();
    const latestEgt = telemetryStore.getChannel('egt').latest();
    const latestVib = telemetryStore.getChannel('vibration_rms').latest();
    const latestOil = telemetryStore.getChannel('oil_pressure').latest();

    if (stripValCht && Number.isFinite(latestCht)) stripValCht.textContent = `${latestCht.toFixed(1)} °C`;
    if (stripValEgt && Number.isFinite(latestEgt)) stripValEgt.textContent = `${latestEgt.toFixed(1)} °C`;
    if (stripValVib && Number.isFinite(latestVib)) stripValVib.textContent = `${latestVib.toFixed(2)} g`;
    if (stripValOil && Number.isFinite(latestOil)) stripValOil.textContent = `${latestOil.toFixed(1)} kPa`;
  }

  // Animation loop for strip chart
  function animateStripChart() {
    if (conductorMode === 'LIVE') {
      renderStripCharts();
    }
    requestAnimationFrame(animateStripChart);
  }
  requestAnimationFrame(animateStripChart);

  // -------------------------------------------------------------------------
  // 7. NASA Open MCT Time Conductor (Bidirectional LIVE / FIXED Replay)
  // -------------------------------------------------------------------------
  const btnConductorMode = document.getElementById('btn-conductor-mode');
  const btnConductorStepBack = document.getElementById('btn-conductor-step-back');
  const btnConductorPlayPause = document.getElementById('btn-conductor-play-pause');
  const btnConductorStepFwd = document.getElementById('btn-conductor-step-fwd');
  const conductorTimeline = document.getElementById('conductor-timeline');
  const conductorTimeCurrent = document.getElementById('conductor-time-current');
  const btnConductorResume = document.getElementById('btn-conductor-resume');
  const conductorStatusBadge = document.getElementById('conductor-status-badge');

  let conductorMode = 'LIVE'; // 'LIVE' | 'FIXED'
  let isConductorPlaying = false;
  let conductorPlayInterval = null;

  function setConductorMode(mode) {
    conductorMode = mode;

    if (mode === 'LIVE') {
      if (btnConductorMode) {
        btnConductorMode.textContent = 'LIVE (NOW)';
        btnConductorMode.className = 'conductor-mode-pill mode-live';
      }
      if (conductorStatusBadge) {
        conductorStatusBadge.textContent = 'LIVE (10 Hz)';
        conductorStatusBadge.className = 'conductor-mode-pill mode-live';
      }
      if (conductorTimeline) {
        conductorTimeline.value = 1200;
        conductorTimeline.classList.remove('fixed-active');
      }
      if (conductorTimeCurrent) {
        conductorTimeCurrent.textContent = `LIVE (T+${formatTime(missionSeconds)})`;
        conductorTimeCurrent.style.color = '#3fb950';
      }
      if (btnConductorPlayPause) {
        btnConductorPlayPause.textContent = '⏸ FREEZE';
      }
      clearInterval(conductorPlayInterval);
      isConductorPlaying = false;
    } else {
      if (btnConductorMode) {
        btnConductorMode.textContent = 'FIXED (REWOUND)';
        btnConductorMode.className = 'conductor-mode-pill mode-fixed';
      }
      if (conductorStatusBadge) {
        conductorStatusBadge.textContent = 'FIXED (PAUSED)';
        conductorStatusBadge.className = 'conductor-mode-pill mode-fixed';
      }
      if (conductorTimeline) {
        conductorTimeline.classList.add('fixed-active');
      }
      if (conductorTimeCurrent) {
        conductorTimeCurrent.style.color = '#d29922';
      }
    }
  }

  function applyConductorScrub(sliderVal) {
    setConductorMode('FIXED');

    // sliderVal is 0 to 1200 (1200 = now, 0 = now - 120s)
    const secOffset = (1200 - sliderVal) * 0.1;
    const targetTime = Math.max(0, missionSeconds - secOffset);

    if (conductorTimeCurrent) {
      conductorTimeCurrent.textContent = `FIXED (T+${formatTime(targetTime)} / -${secOffset.toFixed(1)}s)`;
    }

    // Retrieve historical frame from TelemetryStore
    const frame = telemetryStore.getHistoricalFrame(targetTime);
    if (frame) {
      // Rewind 3D twin mesh, thermal heatmap & physical sensor nodes
      engine3D.applyHistoricalFrame(frame);

      // Rewind Parameter Rail
      updateParameterRail(frame.telemetry);

      // Rewind Subassembly Health Matrix
      if (frame.component_health) {
        updateSubassemblyHealth(frame.component_health, frame.telemetry);
      }

      // Rewind RUL Card
      updateRulDisplay(frame.adjusted_rul, 0, frame.is_physically_valid);
    }

    // Render strip chart centered on historical scrub time
    renderStripCharts(targetTime);
  }

  if (conductorTimeline) {
    conductorTimeline.addEventListener('input', (e) => {
      applyConductorScrub(parseInt(e.target.value, 10));
    });
  }

  if (btnConductorMode) {
    btnConductorMode.addEventListener('click', () => {
      if (conductorMode === 'LIVE') {
        applyConductorScrub(1100); // Step back 10s into history
      } else {
        setConductorMode('LIVE');
      }
    });
  }

  if (btnConductorResume) {
    btnConductorResume.addEventListener('click', () => {
      setConductorMode('LIVE');
    });
  }

  if (btnConductorStepBack) {
    btnConductorStepBack.addEventListener('click', () => {
      const cur = conductorTimeline ? parseInt(conductorTimeline.value, 10) : 1200;
      const nextVal = Math.max(0, cur - 100);
      if (conductorTimeline) conductorTimeline.value = nextVal;
      applyConductorScrub(nextVal);
    });
  }

  if (btnConductorStepFwd) {
    btnConductorStepFwd.addEventListener('click', () => {
      const cur = conductorTimeline ? parseInt(conductorTimeline.value, 10) : 1200;
      const nextVal = Math.min(1200, cur + 100);
      if (conductorTimeline) conductorTimeline.value = nextVal;
      if (nextVal >= 1200) {
        setConductorMode('LIVE');
      } else {
        applyConductorScrub(nextVal);
      }
    });
  }

  if (btnConductorPlayPause) {
    btnConductorPlayPause.addEventListener('click', () => {
      if (conductorMode === 'LIVE') {
        applyConductorScrub(1200);
        btnConductorPlayPause.textContent = '▶ PLAY';
      } else {
        isConductorPlaying = !isConductorPlaying;
        if (isConductorPlaying) {
          btnConductorPlayPause.textContent = '⏸ PAUSE';
          conductorPlayInterval = setInterval(() => {
            const cur = parseInt(conductorTimeline.value, 10);
            if (cur >= 1200) {
              setConductorMode('LIVE');
            } else {
              conductorTimeline.value = cur + 10;
              applyConductorScrub(cur + 10);
            }
          }, 100);
        } else {
          btnConductorPlayPause.textContent = '▶ PLAY';
          clearInterval(conductorPlayInterval);
        }
      }
    });
  }

  // -------------------------------------------------------------------------
  // 8. Viewport Cross-Section Clipping Controls
  // -------------------------------------------------------------------------
  const clipChkX = document.getElementById('clip-chk-x');
  const clipSlideX = document.getElementById('clip-slide-x');
  const clipChkY = document.getElementById('clip-chk-y');
  const clipSlideY = document.getElementById('clip-slide-y');
  const clipChkZ = document.getElementById('clip-chk-z');
  const clipSlideZ = document.getElementById('clip-slide-z');

  function updateClippingPlane(axis, chkEl, slideEl) {
    if (!chkEl || !slideEl) return;
    const enabled = chkEl.checked;
    const val = parseFloat(slideEl.value);
    engine3D.setClippingPlane(axis, enabled, val);
  }

  if (clipChkX && clipSlideX) {
    clipChkX.addEventListener('change', () => updateClippingPlane('x', clipChkX, clipSlideX));
    clipSlideX.addEventListener('input', () => updateClippingPlane('x', clipChkX, clipSlideX));
  }
  if (clipChkY && clipSlideY) {
    clipChkY.addEventListener('change', () => updateClippingPlane('y', clipChkY, clipSlideY));
    clipSlideY.addEventListener('input', () => updateClippingPlane('y', clipChkY, clipSlideY));
  }
  if (clipChkZ && clipSlideZ) {
    clipChkZ.addEventListener('change', () => updateClippingPlane('z', clipChkZ, clipSlideZ));
    clipSlideZ.addEventListener('input', () => updateClippingPlane('z', clipChkZ, clipSlideZ));
  }

  // -------------------------------------------------------------------------
  // 9. 3D Viewport Controls & Exploded View Slider
  // -------------------------------------------------------------------------
  const explodedSlider = document.getElementById('exploded-slider');
  const explodedVal = document.getElementById('exploded-val');
  const btnHeatmap = document.getElementById('btn-heatmap');
  const btnZoomIn = document.getElementById('btn-zoom-in');
  const btnZoomOut = document.getElementById('btn-zoom-out');
  const btnResetView = document.getElementById('btn-reset-view');

  if (explodedSlider && explodedVal) {
    explodedSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      explodedVal.textContent = `${Math.round(val)}%`;
      engine3D.setExplodedView(val);
    });
  }

  if (btnHeatmap) {
    btnHeatmap.addEventListener('click', () => {
      const active = engine3D.toggleHeatmap();
      btnHeatmap.classList.toggle('active', active);
    });
  }

  if (btnZoomIn) btnZoomIn.addEventListener('click', () => engine3D.zoomIn(0.25));
  if (btnZoomOut) btnZoomOut.addEventListener('click', () => engine3D.zoomOut(0.25));
  if (btnResetView) btnResetView.addEventListener('click', () => engine3D.resetView());

  const viewBtns = document.querySelectorAll('.vm-btn, .mvm-btn');
  viewBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-view');
      viewBtns.forEach((b) => {
        b.classList.toggle('active', b.getAttribute('data-view') === mode);
      });
      engine3D.setViewMode(mode);
    });
  });

  // -------------------------------------------------------------------------
  // 10. Subassembly Health Matrix Controller
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
      },
    ];

    components.forEach((comp) => {
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
        if (pct < 55) barEl.classList.add('critical');
        else if (pct < 80) barEl.classList.add('warning');
      }

      if (metricEl) metricEl.textContent = comp.metric;

      if (condEl) {
        condEl.classList.remove('warning', 'critical');
        if (pct >= 85) condEl.textContent = 'NOMINAL';
        else if (pct >= 70) {
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
  // 11. Remaining Useful Life (RUL) & PINN Compliance Display
  // -------------------------------------------------------------------------
  const rulNumber = document.getElementById('rul-number');
  const rulExtension = document.getElementById('rul-extension');
  const rulTimeVal = document.getElementById('rul-time-val');
  const rulTimeExt = document.getElementById('rul-time-ext');
  const rulStatusBadge = document.getElementById('rul-status-badge');
  const rulHeroCard = document.getElementById('rul-hero-card');
  const fourierResidual = document.getElementById('fourier-residual');
  const physicalGradient = document.getElementById('physical-gradient');
  const fourierAdherence = document.getElementById('fourier-adherence');

  function updateRulDisplay(rulCycles, extensionCycles = 0, isPhysicallyValid = true, sustainStr = null, missionStatus = null) {
    const rawCycles = rulCycles !== undefined && !isNaN(rulCycles) ? rulCycles : 485.0;
    const cycles = Math.round(rawCycles);
    const ext = extensionCycles !== undefined && !isNaN(extensionCycles) ? extensionCycles : 0.0;

    if (rulNumber) rulNumber.textContent = cycles;

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

  // -------------------------------------------------------------------------
  // 12. Fault Classifier & DRL Displays
  // -------------------------------------------------------------------------
  const faultName = document.getElementById('fault-name');
  const faultConf = document.getElementById('fault-conf');
  const drlText = document.getElementById('drl-text');
  const drlShieldBadge = document.getElementById('drl-shield-badge');
  const dataIntegrityStatus = document.getElementById('data-integrity-status');

  // Setup Fault Injection Buttons
  const injectBtns = document.querySelectorAll('.inject-btn');
  injectBtns.forEach((btn) => {
    btn.addEventListener('click', async () => {
      const faultType = btn.getAttribute('data-fault');
      injectBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      try {
        await fetch('/api/simulate/inject', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fault_type: faultType || null }),
        });
      } catch (err) {
        console.error('Fault injection error:', err);
      }
    });
  });

  // -------------------------------------------------------------------------
  // 13. Benchmarks Modal
  // -------------------------------------------------------------------------
  const btnBenchmarks = document.getElementById('btn-benchmarks');
  const modalBackdrop = document.getElementById('modal-backdrop');
  const modalClose = document.getElementById('modal-close');
  const benchmarkTbody = document.getElementById('benchmark-tbody');

  if (btnBenchmarks && modalBackdrop) {
    btnBenchmarks.addEventListener('click', async () => {
      modalBackdrop.classList.add('open');
      try {
        const res = await fetch('/api/benchmarks');
        const data = await res.json();
        if (benchmarkTbody && data.metrics) {
          benchmarkTbody.innerHTML = data.metrics
            .map(
              (m) => `
            <tr>
              <td><strong>${m.metric}</strong></td>
              <td>${m.traditional_dl}</td>
              <td>${m.modular_dt}</td>
              <td class="highlight-col">${m.agentic_pinn_drl}</td>
              <td><span class="gain-badge">${m.improvement}</span></td>
            </tr>
          `
            )
            .join('');
        }
      } catch (e) {
        console.error(e);
      }
    });

    if (modalClose) modalClose.addEventListener('click', () => modalBackdrop.classList.remove('open'));
    modalBackdrop.addEventListener('click', (e) => {
      if (e.target === modalBackdrop) modalBackdrop.classList.remove('open');
    });
  }

  // -------------------------------------------------------------------------
  // 14. Architecture & Presentation Deck Modal
  // -------------------------------------------------------------------------
  const btnArchitecture = document.getElementById('btn-architecture');
  const modalArchitecture = document.getElementById('modal-architecture');
  const modalArchClose = document.getElementById('modal-arch-close');
  const archTabBtns = document.querySelectorAll('.arch-tab-btn');
  const archTabPanes = document.querySelectorAll('.arch-tab-pane');

  if (btnArchitecture && modalArchitecture) {
    btnArchitecture.addEventListener('click', () => modalArchitecture.classList.add('open'));
    if (modalArchClose) modalArchClose.addEventListener('click', () => modalArchitecture.classList.remove('open'));
    modalArchitecture.addEventListener('click', (e) => {
      if (e.target === modalArchitecture) modalArchitecture.classList.remove('open');
    });

    archTabBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');
        archTabBtns.forEach((b) => b.classList.remove('active'));
        archTabPanes.forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        const targetPane = document.getElementById(tabId);
        if (targetPane) targetPane.classList.add('active');
      });
    });

    const btnViewDag = document.getElementById('btn-view-dag');
    const btnViewPipeline = document.getElementById('btn-view-pipeline');
    const btnViewSlide = document.getElementById('btn-view-slide');
    const slide3Img = document.getElementById('slide3-img');

    if (btnViewDag && btnViewPipeline && btnViewSlide && slide3Img) {
      const subtabBtns = [btnViewDag, btnViewPipeline, btnViewSlide];

      btnViewDag.addEventListener('click', () => {
        subtabBtns.forEach((b) => b.classList.remove('active'));
        btnViewDag.classList.add('active');
        slide3Img.src = '/static/assets/orchestrator_routing_dag.svg';
        slide3Img.alt = 'IdeaForge AI Agent Orchestrator Routing DAG';
      });

      btnViewPipeline.addEventListener('click', () => {
        subtabBtns.forEach((b) => b.classList.remove('active'));
        btnViewPipeline.classList.add('active');
        slide3Img.src = '/static/assets/layer_parameter_pipeline.svg';
        slide3Img.alt = 'IdeaForge Layer-by-Layer Parameters and Baseline Paper Comparison';
      });

      btnViewSlide.addEventListener('click', () => {
        subtabBtns.forEach((b) => b.classList.remove('active'));
        btnViewSlide.classList.add('active');
        slide3Img.src = '/static/assets/ai_agent_workflow.jpg';
        slide3Img.alt = 'IdeaForge Slide 3 Full PPT Slide';
      });
    }
  }

  // -------------------------------------------------------------------------
  // 15. Day / Night Tactical Mode Toggle
  // -------------------------------------------------------------------------
  const btnThemeToggle = document.getElementById('btn-theme-toggle');
  const themeIcon = document.getElementById('theme-icon');
  const themeText = document.getElementById('theme-text');

  function updateTheme(isLight) {
    if (isLight) {
      document.body.classList.add('light-theme');
      if (themeIcon) themeIcon.textContent = '🌙';
      if (themeText) themeText.textContent = 'NIGHT MODE';
      if (window._engine3D) window._engine3D.setTheme('day');
      localStorage.setItem('tapas_theme', 'light');
    } else {
      document.body.classList.remove('light-theme');
      if (themeIcon) themeIcon.textContent = '☀️';
      if (themeText) themeText.textContent = 'DAY MODE';
      if (window._engine3D) window._engine3D.setTheme('night');
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

  // -------------------------------------------------------------------------
  // 16. WebSocket Live 10 Hz Telemetry Streaming
  // -------------------------------------------------------------------------
  const wsStatus = document.getElementById('ws-status');

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('[Telemetry WS] Connected to 10 Hz telemetry stream');
      if (wsStatus) {
        wsStatus.textContent = '10 Hz SYNCED';
        wsStatus.style.color = 'var(--accent-emerald)';
      }
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Compute simulated mission elapsed time
        missionSeconds += 0.1;
        if (missionClockEl) missionClockEl.textContent = formatTime(missionSeconds);

        const tel = data.telemetry || {};

        // Ingest into TelemetryStore ring buffer
        telemetryStore.ingest(tel, missionSeconds, data);

        // Check for state transitions & emit into dedicated EventLog
        eventLog.checkTransitions(tel, missionSeconds);
        updateAnnunciators();

        const imputedList = (data.sensor_audit && data.sensor_audit.imputed_fields) || [];
        if (dataIntegrityStatus) {
          if (imputedList.length > 0) {
            dataIntegrityStatus.textContent = `${imputedList.length} IMPUTED`;
            dataIntegrityStatus.style.color = '#38bdf8';
          } else {
            dataIntegrityStatus.textContent = '100% (AUDITED)';
            dataIntegrityStatus.style.color = '#3fb950';
          }
        }

        // If in LIVE mode, immediately drive all operational widgets
        if (conductorMode === 'LIVE') {
          // 1. Update 3D twin mesh, thermal heatmap & sensor nodes
          engine3D.updateTelemetryState(data);
          if (window._engine3DExtension) {
            window._engine3DExtension.onPayload(data);
          }

          // 2. Update Monitored Parameter Rail with tabular values, trends, & sparklines
          updateParameterRail(tel, imputedList);

          // 3. Update PINN Residuals & Adherence
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

          // 4. Update RUL Prognostic Card
          updateRulDisplay(
            data.adjusted_rul !== undefined ? data.adjusted_rul : data.rul_cycles,
            data.extension_cycles,
            data.is_physically_valid,
            data.sustain_flight_str,
            data.mission_status
          );

          // 5. Update Fault Archetype Classifier
          if (data.fault_archetype && faultName) {
            faultName.textContent = data.fault_archetype.toUpperCase().replace('_', ' ');
            if (faultConf) faultConf.textContent = `${Math.round((data.fault_confidence || 0.85) * 100)}% CONF`;

            if (data.fault_probabilities) {
              for (const [arch, prob] of Object.entries(data.fault_probabilities)) {
                const bar = document.getElementById(`fault-bar-${arch}`);
                const pct = document.getElementById(`fault-pct-${arch}`);
                if (bar) bar.style.width = `${Math.round(prob * 100)}%`;
                if (pct) pct.textContent = `${Math.round(prob * 100)}%`;
              }
            }
          }

          // 6. Update DRL Prognostic Actions
          if (data.drl_action && drlText) {
            drlText.textContent = data.drl_action.recommendation || 'Nominal cruise envelope';
            if (drlShieldBadge) {
              drlShieldBadge.style.display = data.drl_action.shield_applied ? 'inline-flex' : 'none';
            }
          }

          // 7. Update Subassembly Health Matrix
          if (data.component_health) {
            updateSubassemblyHealth(data.component_health, tel);
          }

          // 8. Update conductor live text
          if (conductorTimeCurrent) {
            conductorTimeCurrent.textContent = `LIVE (T+${formatTime(missionSeconds)})`;
          }
        }
      } catch (err) {
        console.error('[Telemetry] Processing error:', err);
      }
    };

    socket.onclose = () => {
      console.log('[Telemetry WS] Connection lost, reconnecting in 2s...');
      if (wsStatus) {
        wsStatus.textContent = 'DISCONNECTED';
        wsStatus.style.color = 'var(--accent-rose)';
      }
      setTimeout(connectWebSocket, 2000);
    };

    socket.onerror = (err) => {
      console.error('[Telemetry WS] Error:', err);
      socket.close();
    };
  }

  connectWebSocket();
});

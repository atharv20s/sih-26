// PRAHARI Mission Map — 2D SVG rendering, trajectory animation, click
// interaction, and the "Simulate Attack" tie-in to the real fault-injection
// pipeline already built for the Live Ops Console.

(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const { DRDO_STATIONS, COMM_STATIONS, TRAJECTORY, ATTACK_TYPES } = window.MISSION_MAP_DATA;

  let PROJECTION = null; // loaded from india_outline.json — must match scripts/build_india_map.py

  const svg = document.getElementById('map-svg');
  const detailCard = document.getElementById('detail-card');
  const threatBanner = document.getElementById('threat-banner');
  const threatBannerText = document.getElementById('threat-banner-text');
  const missionClockEl = document.getElementById('mission-clock');
  const attackListEl = document.getElementById('attack-type-list');
  const clearBtn = document.getElementById('btn-clear-attack');

  function project(lng, lat) {
    const b = PROJECTION.projection_bounds;
    const x = (lng - b.lng_min) / (b.lng_max - b.lng_min) * b.view_w;
    const y = (1 - (lat - b.lat_min) / (b.lat_max - b.lat_min)) * b.view_h;
    return [x, y];
  }

  function el(tag, attrs) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function showDetail(title, lines, color) {
    detailCard.innerHTML = `<div class="dc-title" style="color:${color || 'var(--text-primary)'}">${title}</div>` +
      lines.map(l => `<div>${l}</div>`).join('');
  }

  function renderOutline() {
    const path = el('path', {
      d: PROJECTION.path,
      fill: 'rgba(19,32,51,0.85)',
      stroke: '#38bdf8',
      'stroke-width': '1.4',
      'stroke-opacity': '0.7',
    });
    svg.appendChild(path);

    // Subtle glow filter for the outline stroke
    const defs = el('defs', {});
    defs.innerHTML = `<filter id="glow"><feGaussianBlur stdDeviation="1.5" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`;
    svg.insertBefore(defs, svg.firstChild);
    path.setAttribute('filter', 'url(#glow)');
  }

  function renderTrajectoryPath() {
    const pts = TRAJECTORY.map(wp => project(wp.lng, wp.lat));
    const d = 'M' + pts.map(p => p.join(',')).join(' L');

    // Wide soft glow pass underneath, then the crisp dashed line on top —
    // makes the route read clearly as a flight path against the dark map
    // instead of a thin scribble.
    const glowLine = el('path', {
      d, fill: 'none', stroke: '#58a6ff', 'stroke-width': '7',
      'stroke-opacity': '0.18', 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
    });
    svg.appendChild(glowLine);

    const line = el('path', {
      d, fill: 'none', stroke: '#7dd3ff', 'stroke-width': '2.6',
      'stroke-dasharray': '10 6', 'stroke-opacity': '0.95',
      'stroke-linecap': 'round', 'stroke-linejoin': 'round',
    });
    svg.appendChild(line);

    // Small directional chevrons at the midpoint of each leg so the route
    // reads as a flown path, not just a connect-the-dots outline.
    for (let i = 0; i < pts.length - 1; i++) {
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[i + 1];
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      const ang = Math.atan2(y2 - y1, x2 - x1);
      const size = 7;
      const p1 = [mx - size * Math.cos(ang - 0.4), my - size * Math.sin(ang - 0.4)];
      const p2 = [mx - size * Math.cos(ang + 0.4), my - size * Math.sin(ang + 0.4)];
      const chevron = el('path', {
        d: `M${p1[0]},${p1[1]} L${mx},${my} L${p2[0]},${p2[1]}`,
        fill: 'none', stroke: '#7dd3ff', 'stroke-width': '2',
        'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-opacity': '0.9',
      });
      svg.appendChild(chevron);
    }
  }

  function renderWaypoints() {
    TRAJECTORY.forEach((wp, i) => {
      const [x, y] = project(wp.lng, wp.lat);
      const isThreat = !!wp.isThreatZone;
      const dot = el('circle', {
        cx: x, cy: y, r: isThreat ? 7 : 4.5,
        fill: isThreat ? '#f85149' : '#58a6ff',
        stroke: '#0a0e14', 'stroke-width': '1.2',
        style: 'cursor:pointer;',
      });
      dot.addEventListener('click', () => {
        showDetail(wp.label, [wp.narrative, `Altitude: ${wp.alt_ft.toLocaleString()} ft`], isThreat ? '#f85149' : '#58a6ff');
      });
      svg.appendChild(dot);
      if (isThreat) {
        const ring = el('circle', { cx: x, cy: y, r: 7, fill: 'none', stroke: '#f85149', 'stroke-width': '1.5', opacity: '0.7' });
        ring.innerHTML = `<animate attributeName="r" values="7;16;7" dur="1.8s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0.7;0;0.7" dur="1.8s" repeatCount="indefinite"/>`;
        svg.appendChild(ring);
      }
    });
  }

  function renderStations(list, color, radius, labelKey) {
    list.forEach((s) => {
      const [x, y] = project(s.lng, s.lat);
      const g = el('g', { style: 'cursor:pointer;' });
      const dot = el('circle', { cx: x, cy: y, r: radius, fill: color, stroke: '#0a0e14', 'stroke-width': '1.2' });
      const label = el('text', {
        x: x + radius + 4, y: y + 3, fill: 'var(--text-secondary)', 'font-size': '9',
        'font-family': 'JetBrains Mono, monospace',
      });
      label.textContent = s.city || s.name;
      g.appendChild(dot);
      g.appendChild(label);
      g.addEventListener('click', () => {
        if (s.type) {
          showDetail(s.name, [`City: ${s.city}`, `Type: ${s.type}`, 'Public, city-level location.'], color);
        } else {
          showDetail(s.name, [s.note], color);
        }
      });
      svg.appendChild(g);
    });
  }

  let aircraftMarker = null;
  function renderAircraftMarker() {
    aircraftMarker = el('circle', { r: 6, fill: '#3fb950', stroke: '#0a0e14', 'stroke-width': '1.5' });
    svg.appendChild(aircraftMarker);
    const halo = el('circle', { r: 10, fill: 'none', stroke: '#3fb950', 'stroke-width': '1', opacity: '0.5' });
    halo.setAttribute('id', 'aircraft-halo');
    svg.appendChild(halo);
  }

  // --- Animation clock: t goes 0 -> 1 across the whole trajectory, looping.
  // Shared globally (window.missionMapClock) so mission-map-3d.js stays in sync.
  const CYCLE_SECONDS = 40;
  let startTime = performance.now();
  window.missionMapClock = { t: 0, segIndex: 0, segFrac: 0 };

  function interpolateTrajectory(t) {
    const n = TRAJECTORY.length - 1;
    const scaled = t * n;
    const segIndex = Math.min(n - 1, Math.floor(scaled));
    const segFrac = scaled - segIndex;
    const a = TRAJECTORY[segIndex];
    const b = TRAJECTORY[segIndex + 1];
    return {
      lat: a.lat + (b.lat - a.lat) * segFrac,
      lng: a.lng + (b.lng - a.lng) * segFrac,
      alt_ft: a.alt_ft + (b.alt_ft - a.alt_ft) * segFrac,
      segIndex, segFrac,
    };
  }

  function tick() {
    const elapsed = (performance.now() - startTime) / 1000;
    const t = (elapsed % CYCLE_SECONDS) / CYCLE_SECONDS;
    const pos = interpolateTrajectory(t);
    window.missionMapClock = { t, ...pos };

    const [x, y] = project(pos.lng, pos.lat);
    if (aircraftMarker) {
      aircraftMarker.setAttribute('cx', x);
      aircraftMarker.setAttribute('cy', y);
      const halo = document.getElementById('aircraft-halo');
      if (halo) { halo.setAttribute('cx', x); halo.setAttribute('cy', y); }
    }

    const missionSeconds = Math.floor(elapsed);
    if (missionClockEl) {
      const m = Math.floor(missionSeconds / 60).toString().padStart(2, '0');
      const s = (missionSeconds % 60).toString().padStart(2, '0');
      missionClockEl.textContent = `T+${m}:${s}`;
    }

    requestAnimationFrame(tick);
  }

  // --- Threat scenario controls: multiple attack TYPES against the
  // aircraft's own systems, each a real fault-injection already used by the
  // Live Ops Console (src/server/server.py). Selecting one has a genuine
  // effect on the shared live simulation.
  let activeAttackId = null;

  function renderAttackButtons() {
    attackListEl.innerHTML = '';
    ATTACK_TYPES.forEach((atk) => {
      const btn = document.createElement('button');
      btn.className = 'btn btn-danger';
      btn.style.width = '100%';
      btn.style.textAlign = 'left';
      btn.dataset.attackId = atk.id;
      btn.textContent = atk.label;
      btn.addEventListener('click', () => triggerAttack(atk));
      attackListEl.appendChild(btn);
    });
  }

  async function injectFault(faultType) {
    try {
      const res = await fetch('/api/simulate/inject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ fault_type: faultType }),
      });
      if (res.status === 401) {
        window.location.href = '/login';
        return false;
      }
      return res.ok;
    } catch (err) {
      console.error('[MissionMap] fault injection request failed', err);
      return false;
    }
  }

  async function triggerAttack(atk) {
    activeAttackId = atk.id;
    Array.from(attackListEl.children).forEach((el) => {
      el.blur(); // clear any lingering focus ring so only .active reads as "selected"
      el.classList.toggle('active', el.dataset.attackId === atk.id);
    });
    threatBanner.classList.add('active');
    threatBannerText.textContent = `SIMULATED THREAT ACTIVE — ${atk.label.toUpperCase()}`;
    showDetail(atk.label, [atk.description], '#f85149');
    document.getElementById('resp-narrative').textContent =
      `Threat injected — watching PRAHARI's defense layers detect and mitigate it below.`;

    // Flash the always-visible top strip so the live reaction can't be missed.
    const strip = document.getElementById('live-monitor-strip');
    if (strip) {
      strip.style.borderColor = 'var(--accent-rose)';
      strip.style.boxShadow = '0 0 0 1px var(--accent-rose)';
      strip.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      setTimeout(() => { strip.style.borderColor = ''; strip.style.boxShadow = ''; }, 2000);
    }

    await injectFault(atk.faultType);
  }

  clearBtn.addEventListener('click', async () => {
    activeAttackId = null;
    Array.from(attackListEl.children).forEach((el) => el.classList.remove('active'));
    threatBanner.classList.remove('active');
    document.getElementById('resp-narrative').textContent = 'Select a threat to see PRAHARI detect and mitigate it live.';
    await injectFault(null);
  });

  // --- Live "System Response" panel: a lightweight read-only WS connection
  // to the same /ws/telemetry stream the Live Ops Console uses, so the
  // mitigation story (safe-mode gate, ensemble agreement, trend risk) is
  // real live data, not narrative text.
  function connectResponseFeed() {
    const statusEl = document.getElementById('resp-ws-status');
    const lmDot = document.getElementById('lm-ws-dot');
    const lmLabel = document.getElementById('lm-ws-label');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`);

    ws.onopen = () => {
      if (statusEl) { statusEl.textContent = '● LIVE'; statusEl.style.color = 'var(--accent-emerald)'; }
      if (lmDot) lmDot.classList.add('live');
      if (lmLabel) lmLabel.textContent = 'LIVE TELEMETRY MONITORING';
    };
    ws.onclose = (event) => {
      if (statusEl) { statusEl.textContent = '○ disconnected'; statusEl.style.color = 'var(--text-muted)'; }
      if (lmDot) lmDot.classList.remove('live');
      if (lmLabel) lmLabel.textContent = 'RECONNECTING…';
      if (event.code === 4401) { window.location.href = '/login'; return; }
      setTimeout(connectResponseFeed, 2000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch (e) { return; }

      const fault = data.fault_archetype || 'nominal';
      const actionMode = (data.drl_action && data.drl_action.action_mode) || 'AUTONOMOUS_ACTION';
      const agreement = data.fault_agreement_score !== undefined ? data.fault_agreement_score : 1.0;
      const trend = data.trend_risk_score !== undefined ? data.trend_risk_score : 0;
      const safeMode = actionMode !== 'AUTONOMOUS_ACTION';
      const faultColor = fault === 'nominal' ? 'var(--text-primary)' : 'var(--accent-rose)';
      const modeColor = safeMode ? 'var(--accent-rose)' : 'var(--accent-emerald)';

      const faultEl = document.getElementById('resp-fault');
      const modeEl = document.getElementById('resp-action-mode');
      const agreeEl = document.getElementById('resp-agreement');
      const trendEl = document.getElementById('resp-trend');
      if (faultEl) { faultEl.textContent = fault.toUpperCase(); faultEl.style.color = faultColor; }
      if (modeEl) { modeEl.textContent = safeMode ? 'SAFE MODE' : 'AUTONOMOUS'; modeEl.style.color = modeColor; }
      if (agreeEl) agreeEl.textContent = `${Math.round(agreement * 100)}%`;
      if (trendEl) { trendEl.textContent = `${Math.round(trend * 100)}%`; trendEl.style.color = trend >= 0.5 ? 'var(--accent-amber)' : 'var(--text-primary)'; }

      // Always-visible strip at the top of the page — mirrors the sidebar
      // panel so the live reaction is impossible to miss without scrolling.
      const lmFault = document.getElementById('lm-fault');
      const lmMode = document.getElementById('lm-mode');
      const lmAgreement = document.getElementById('lm-agreement');
      const lmTrend = document.getElementById('lm-trend');
      if (lmFault) { lmFault.textContent = fault.toUpperCase(); lmFault.style.color = faultColor; }
      if (lmMode) { lmMode.textContent = safeMode ? 'SAFE MODE' : 'AUTONOMOUS'; lmMode.style.color = modeColor; }
      if (lmAgreement) lmAgreement.textContent = `${Math.round(agreement * 100)}%`;
      if (lmTrend) { lmTrend.textContent = `${Math.round(trend * 100)}%`; lmTrend.style.color = trend >= 0.5 ? 'var(--accent-amber)' : 'var(--text-primary)'; }

      if (activeAttackId) {
        const narrativeEl = document.getElementById('resp-narrative');
        if (narrativeEl) {
          narrativeEl.textContent = safeMode
            ? '⚠ Layer 4 safe-mode gate ENGAGED — holding last known-good setting, escalated to operator. This is PRAHARI avoiding an unsafe autonomous action under an active threat.'
            : '✓ Telemetry integrity/ensemble agreement restored — PRAHARI has mitigated the threat and resumed autonomous operation.';
        }
      }
    };
  }

  async function init() {
    const resp = await fetch('/static/assets/india_outline.json');
    PROJECTION = await resp.json();

    renderOutline();
    renderTrajectoryPath();
    renderStations(COMM_STATIONS, '#10b981', 4, 'name');
    renderStations(DRDO_STATIONS, '#f59e0b', 5, 'name');
    renderWaypoints();
    renderAircraftMarker();
    requestAnimationFrame(tick);

    renderAttackButtons();
    connectResponseFeed();
  }

  init();
})();

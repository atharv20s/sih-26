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

  // DOM Elements
  const explodedSlider = document.getElementById('exploded-slider');
  const explodedVal = document.getElementById('exploded-val');
  const btnHeatmap = document.getElementById('btn-heatmap');
  const btnResetView = document.getElementById('btn-reset-view');

  const rulNumber = document.getElementById('rul-number');
  const rulExtension = document.getElementById('rul-extension');
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

  // Exploded View Slider Listener
  explodedSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    explodedVal.textContent = `${Math.round(val * 100)}%`;
    engine3D.setExplodedView(val);
  });

  // Toggle Heatmap Button
  btnHeatmap.addEventListener('click', () => {
    engine3D.heatmapEnabled = !engine3D.heatmapEnabled;
    btnHeatmap.classList.toggle('active', engine3D.heatmapEnabled);
    if (!engine3D.heatmapEnabled) {
      engine3D.headMat.emissiveIntensity = 0.0;
      engine3D.exhaustMat.emissiveIntensity = 0.0;
    }
  });

  // Reset Camera View
  btnResetView.addEventListener('click', () => {
    engine3D.camera.position.set(3.8, 2.6, 4.8);
    engine3D.controls.target.set(0, 0.4, 0);
  });

  // Setup Fault Injection Buttons
  const injectBtns = document.querySelectorAll('.inject-btn');
  injectBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const faultType = btn.getAttribute('data-fault');
      injectBtns.forEach(b => b.classList.remove('active'));

      if (faultType) {
        btn.classList.add('active');
      }

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

        // 2. Update Gauges
        if (data.telemetry) {
          for (const [s, val] of Object.entries(data.telemetry)) {
            updateGauge(s, val);
          }
        }

        // 3. Update PINN Health Card
        if (data.rul_cycles !== undefined) {
          rulNumber.textContent = Math.round(data.adjusted_rul || data.rul_cycles);
        }
        if (data.extension_cycles !== undefined && rulExtension) {
          rulExtension.textContent = data.extension_cycles > 0 ? `+${data.extension_cycles.toFixed(1)} cyc (DRL Protected)` : `Nominal Cruise`;
        }
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
          for (const [comp, hVal] of Object.entries(data.component_health)) {
            const el = document.getElementById(`comp-status-${comp}`);
            if (el) {
              const pct = Math.round(hVal * 100);
              el.textContent = `${pct}%`;
              el.style.color = pct > 70 ? 'var(--accent-emerald)' : (pct > 40 ? 'var(--accent-amber)' : 'var(--accent-rose)');
            }
          }
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

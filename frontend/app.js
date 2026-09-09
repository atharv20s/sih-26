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
    const coolingFactor = Math.max(0.5, (1.0 + (15.0 - ambC) * 0.008) * Math.sqrt(densityRatio)).toFixed(2);
    envConvection.textContent = `Cooling Factor: ${coolingFactor}x`;

    let condition = 'ISA Standard Day';
    if (ambC > 35) condition = 'High Thermal Heat Soak';
    else if (ambC < -10) condition = 'Sub-Zero Freezing Airframe';
    if (altFt > 15000) condition += ' (Hypobaric)';
    envCondition.textContent = `Condition: ${condition}`;

    // Feed environment parameters into live engine simulation
    const envChtOffset = (ambC - 15.0) * 0.4 - (altFt / 20000.0) * 8.0;
    engine3D.updateTelemetryState({
      telemetry: {
        cht: Math.max(110.0, 148.9 + envChtOffset),
        egt: Math.max(500.0, 666.1 + (ambC - 15.0) * 0.6)
      }
    });
  }

  if (sliderAltitude) sliderAltitude.addEventListener('input', updateEnvironmentalState);
  if (sliderAmbientTemp) sliderAmbientTemp.addEventListener('input', updateEnvironmentalState);

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
      if (Math.abs(replaySeconds - pTime) < 500) {
        pill.classList.add('active');
      }
    });

    // Replay telemetry based on mission profile phase
    let simData = {
      telemetry: { rpm: 4800, cht: 148.9, egt: 666.1, oil_pressure: 320, oil_temp: 85, vibration_rms: 1.2, fuel_flow: 9.5, afr: 13.8 },
      rul_cycles: 380,
      adjusted_rul: 380,
      fault_archetype: 'nominal',
      fault_confidence: 0.94
    };

    if (replaySeconds < 300) {
      // Taxi
      simData.telemetry.rpm = 2200;
      simData.telemetry.cht = 115.0;
      simData.telemetry.egt = 540.0;
      simData.rul_cycles = 490;
    } else if (replaySeconds < 1500) {
      // Climb
      simData.telemetry.rpm = 5400;
      simData.telemetry.cht = 168.0;
      simData.telemetry.egt = 720.0;
      simData.rul_cycles = 420;
    } else if (replaySeconds >= 4200 && replaySeconds < 5100) {
      // Thermal Spike Micro-Fault
      simData.telemetry.rpm = 5300;
      simData.telemetry.cht = 214.5;
      simData.telemetry.egt = 792.0;
      simData.rul_cycles = 92;
      simData.adjusted_rul = 92;
      simData.fault_archetype = 'thermal_runaway';
      simData.fault_confidence = 0.91;
    } else if (replaySeconds >= 5100 && replaySeconds < 5800) {
      // DRL De-Rate Active Recovery
      simData.telemetry.rpm = 4450;
      simData.telemetry.cht = 172.0;
      simData.telemetry.egt = 680.0;
      simData.rul_cycles = 135;
      simData.adjusted_rul = 175;
      simData.extension_cycles = 40.0;
      simData.fault_archetype = 'thermal_runaway';
      simData.fault_confidence = 0.88;
    }

    engine3D.updateTelemetryState(simData);
    for (const [s, val] of Object.entries(simData.telemetry)) {
      updateGauge(s, val);
    }
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
        replayInterval = setInterval(() => {
          applyMissionTimelineStep(replaySeconds + Math.round(1 * replaySpeed));
          if (replaySeconds >= 6300) {
            isReplayPlaying = false;
            btnReplayPlay.textContent = '▶ PLAY';
            btnReplayPlay.classList.remove('active');
            clearInterval(replayInterval);
          }
        }, 100);
      } else {
        btnReplayPlay.textContent = '▶ PLAY';
        btnReplayPlay.classList.remove('active');
        if (replayInterval) clearInterval(replayInterval);
      }
    });
  }

  if (btnReplayStepBack) {
    btnReplayStepBack.addEventListener('click', () => {
      applyMissionTimelineStep(replaySeconds - 10);
    });
  }

  if (btnReplayStepFwd) {
    btnReplayStepFwd.addEventListener('click', () => {
      applyMissionTimelineStep(replaySeconds + 10);
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

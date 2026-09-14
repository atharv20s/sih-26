/**
 * SIH26054 P1 — 3D Twin Feature Extensions
 *
 * Adds three missing problem-statement requirements to AeroEngine3D without
 * modifying engine_3d.js:
 *
 *   1. Cross-section clipping planes  (X / Y / Z sliders in the control panel)
 *   2. Raycasted clickable hotspots   (click component → floating metric card)
 *   3. Glowing sensor-node markers    (emissive spheres at physical sensor locations)
 *      + predicted-failure flashing indicator (oscillates on shield_applied / severity CRITICAL)
 *
 * Usage: include AFTER engine_3d.js and after the AeroEngine3D instance is created.
 *
 *   <script src="engine_3d.js"></script>
 *   <script src="engine_3d_p1.js"></script>
 *
 * The extension attaches itself to window._engine3D (the global AeroEngine3D instance).
 * If the instance isn't ready yet, it waits for a 'engine3dReady' custom event.
 */

(function () {
  'use strict';

  // ---- Sensor location catalogue (approximate 3D positions on the engine model) ----
  // These coordinates are in the same model space used by AeroEngine3D.
  // Adjust Z/Y values if the engine geometry shifts between model updates.
  const SENSOR_NODES = [
    {
      id: 'cht_sensor',
      label: 'CHT Sensor',
      field: 'cht',
      unit: '°C',
      position: new THREE.Vector3(-0.55, 0.42, 0.0),
      warnThreshold: 210,
      critThreshold: 260,
      color: 0xff4400,
    },
    {
      id: 'egt_sensor',
      label: 'EGT Probe',
      field: 'egt',
      unit: '°C',
      position: new THREE.Vector3(0.90, -0.10, 0.0),
      warnThreshold: 740,
      critThreshold: 840,
      color: 0xff6600,
    },
    {
      id: 'oil_pressure_sensor',
      label: 'Oil Pressure',
      field: 'oil_pressure',
      unit: ' kPa',
      position: new THREE.Vector3(0.0, -0.55, 0.25),
      warnThreshold: null,   // low-side threshold — handled separately
      critThreshold: null,
      lowWarn: 100,
      lowCrit: 40,
      color: 0x00aaff,
    },
    {
      id: 'vibration_sensor',
      label: 'Vibration RMS',
      field: 'vibration_rms',
      unit: ' g',
      position: new THREE.Vector3(-0.30, 0.10, 0.55),
      warnThreshold: 8,
      critThreshold: 15,
      color: 0xaa00ff,
    },
    {
      id: 'oil_temp_sensor',
      label: 'Oil Temp',
      field: 'oil_temp',
      unit: '°C',
      position: new THREE.Vector3(0.0, -0.55, -0.25),
      warnThreshold: 130,
      critThreshold: 160,
      color: 0x00ffaa,
    },
  ];

  // ---- Component raycasting proxy map (rough AABB positions for each named part) ----
  const COMPONENT_PROXIES = [
    {
      id: 'cylinder_head',
      label: 'Cylinder Head',
      size: new THREE.Vector3(0.6, 0.5, 1.2),
      center: new THREE.Vector3(-0.55, 0.38, 0.0),
      fields: ['cht', 'egt'],
      healthKey: 'cylinder_head',
    },
    {
      id: 'crankshaft',
      label: 'Crankshaft',
      size: new THREE.Vector3(0.25, 0.25, 1.6),
      center: new THREE.Vector3(0.0, 0.0, 0.0),
      fields: ['vibration_rms', 'rpm'],
      healthKey: 'crankshaft',
    },
    {
      id: 'oil_sump',
      label: 'Oil Sump / Lube',
      size: new THREE.Vector3(0.9, 0.35, 1.1),
      center: new THREE.Vector3(0.0, -0.55, 0.0),
      fields: ['oil_pressure', 'oil_temp'],
      healthKey: 'lubrication_system',
    },
    {
      id: 'exhaust_manifold',
      label: 'Exhaust Manifold',
      size: new THREE.Vector3(0.25, 0.3, 1.1),
      center: new THREE.Vector3(0.88, -0.08, 0.0),
      fields: ['egt', 'fuel_flow'],
      healthKey: 'exhaust_manifold',
    },
  ];

  // -------------------------------------------------------------------------
  // Main extension class
  // -------------------------------------------------------------------------
  class Engine3DExtension {
    constructor(engine) {
      this._e = engine;                  // AeroEngine3D instance
      this._latestPayload = null;        // last WS dispatch_payload received
      this._flashPhase = 0.0;
      this._failureFlashing = false;

      // Clipping plane state
      this._clipPlanes = {
        x: new THREE.Plane(new THREE.Vector3(-1, 0, 0), 99),
        y: new THREE.Plane(new THREE.Vector3(0, -1, 0), 99),
        z: new THREE.Plane(new THREE.Vector3(0, 0, -1), 99),
      };
      this._clipEnabled = { x: false, y: false, z: false };

      // Enable local clipping on renderer
      this._e.renderer.localClippingEnabled = true;

      // Raycaster
      this._raycaster = new THREE.Raycaster();
      this._mouse = new THREE.Vector2();
      this._proxyMeshes = [];
      this._hotspotCard = null;

      this._initSensorMarkers();
      this._initProxyMeshes();
      this._initHotspotCard();
      this._initClipControls();
      this._hookCanvasClick();
      this._hookAnimateLoop();
    }

    // ---- 1. Sensor Marker Spheres -------------------------------------------

    _initSensorMarkers() {
      this._sensorMarkers = {};

      SENSOR_NODES.forEach(node => {
        const geo  = new THREE.SphereGeometry(0.065, 12, 12);
        const mat  = new THREE.MeshStandardMaterial({
          color:     node.color,
          emissive:  new THREE.Color(node.color),
          emissiveIntensity: 0.6,
          transparent: true,
          opacity:   0.88,
          depthTest: false,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(node.position);
        mesh.renderOrder = 999;
        mesh.name = node.id;

        // Small ring halo
        const ringGeo = new THREE.RingGeometry(0.085, 0.11, 24);
        const ringMat = new THREE.MeshBasicMaterial({
          color:       node.color,
          transparent: true,
          opacity:     0.35,
          side:        THREE.DoubleSide,
          depthTest:   false,
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.renderOrder = 998;
        mesh.add(ring);

        this._e.scene.add(mesh);
        this._sensorMarkers[node.id] = { mesh, mat, ringMat, ring, node };
      });
    }

    _updateSensorMarkers(telemetry, shieldApplied, severity) {
      const t = this._flashPhase;

      SENSOR_NODES.forEach(({ id, field, warnThreshold, critThreshold, lowWarn, lowCrit }) => {
        const m = this._sensorMarkers[id];
        if (!m) return;
        const val = telemetry ? telemetry[field] : null;

        let status = 'ok';   // ok | warn | crit
        if (val !== null && val !== undefined) {
          if (warnThreshold !== null && critThreshold !== null) {
            if (val >= critThreshold) status = 'crit';
            else if (val >= warnThreshold) status = 'warn';
          } else if (lowWarn !== null) {
            if (val <= lowCrit) status = 'crit';
            else if (val <= lowWarn) status = 'warn';
          }
        }

        // Emissive intensity: pulsing on warn/crit
        let pulse = 0.5;
        if (status === 'warn')  pulse = 0.55 + 0.25 * Math.sin(t * 2.5);
        if (status === 'crit')  pulse = 0.7  + 0.5  * Math.sin(t * 6.0);

        m.mat.emissiveIntensity = pulse;
        m.ringMat.opacity       = status === 'ok' ? 0.18 : 0.45 + 0.25 * Math.sin(t * 4.0);

        // Colour shift on crit: flash toward white-hot
        if (status === 'crit') {
          const f = (Math.sin(t * 6.0) + 1.0) * 0.5;
          m.mat.emissive.setRGB(1.0, 0.3 + f * 0.7, f * 0.4);
        } else {
          m.mat.emissive.set(m.node.color);
        }

        // Overall failure-flash: scale pulse on DRL shield trigger
        if (shieldApplied && severity === 'CRITICAL') {
          m.mat.emissiveIntensity += 0.6 * Math.abs(Math.sin(t * 10.0));
        }
      });
    }

    // ---- 2. Raycasted Proxy Meshes + Hotspot Card --------------------------

    _initProxyMeshes() {
      COMPONENT_PROXIES.forEach(proxy => {
        const geo  = new THREE.BoxGeometry(proxy.size.x, proxy.size.y, proxy.size.z);
        const mat  = new THREE.MeshBasicMaterial({ visible: false });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(proxy.center);
        mesh.userData.proxyId = proxy.id;
        this._e.scene.add(mesh);
        this._proxyMeshes.push(mesh);
      });
    }

    _initHotspotCard() {
      const card = document.createElement('div');
      card.id = 'hotspot-card';
      Object.assign(card.style, {
        position:        'absolute',
        display:         'none',
        background:      'rgba(6, 12, 26, 0.92)',
        border:          '1px solid rgba(0, 200, 255, 0.35)',
        borderRadius:    '10px',
        padding:         '12px 16px',
        minWidth:        '180px',
        color:           '#e0eeff',
        fontFamily:      'Inter, sans-serif',
        fontSize:        '12px',
        lineHeight:      '1.7',
        backdropFilter:  'blur(10px)',
        boxShadow:       '0 4px 24px rgba(0,0,0,0.5)',
        zIndex:          '9999',
        pointerEvents:   'none',
        transition:      'opacity 0.18s ease',
      });
      document.body.appendChild(card);
      this._hotspotCard = card;

      // Close button overlay (pointer-events: all)
      const close = document.createElement('span');
      close.textContent = '✕';
      Object.assign(close.style, {
        position: 'absolute', top: '6px', right: '10px',
        cursor: 'pointer', pointerEvents: 'all', opacity: '0.6',
        fontSize: '13px',
      });
      close.addEventListener('click', () => this._hideCard());
      card.appendChild(close);
      this._cardClose = close;
    }

    _showCard(proxy, screenX, screenY) {
      const card = this._hotspotCard;
      const p    = this._latestPayload;
      const tel  = p?.telemetry || {};
      const ch   = p?.component_health || {};
      const pinn = p?.pinn_results || {};

      let html = `<div style="font-weight:700;font-size:13px;margin-bottom:6px;color:#62d4ff">${proxy.label}</div>`;

      proxy.fields.forEach(f => {
        const v = tel[f];
        const label = f.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const vStr  = v !== undefined && v !== null ? v.toFixed(1) : '—';
        html += `<div><span style="color:#8ab3cc">${label}:</span> <b>${vStr}</b></div>`;
      });

      const health = ch[proxy.healthKey];
      if (health !== undefined) {
        const pct  = Math.round(health * 100);
        const col  = health > 0.75 ? '#4ade80' : health > 0.45 ? '#facc15' : '#f87171';
        html += `<div style="margin-top:6px"><span style="color:#8ab3cc">Component health:</span> <b style="color:${col}">${pct}%</b></div>`;
      }

      if (pinn.predicted_rul !== undefined) {
        html += `<div><span style="color:#8ab3cc">PINN RUL:</span> <b>${pinn.predicted_rul} cycles</b></div>`;
      }
      if (pinn.model_active !== undefined) {
        const src = pinn.model_active ? '🟢 Model' : '🟡 Fallback';
        html += `<div style="font-size:10px;color:#60748a;margin-top:4px">Source: ${src}</div>`;
      }

      card.innerHTML = '';
      card.appendChild(this._cardClose);
      card.insertAdjacentHTML('beforeend', html);

      // Position card near click but keep within viewport
      const margin = 14;
      const cw = window.innerWidth, ch2 = window.innerHeight;
      let x = screenX + margin, y = screenY + margin;
      card.style.display = 'block';
      const rect = card.getBoundingClientRect();
      if (x + rect.width  > cw) x = screenX - rect.width  - margin;
      if (y + rect.height > ch2) y = screenY - rect.height - margin;
      card.style.left = x + 'px';
      card.style.top  = y + 'px';
    }

    _hideCard() {
      if (this._hotspotCard) this._hotspotCard.style.display = 'none';
    }

    _hookCanvasClick() {
      this._e.canvas.addEventListener('click', (evt) => {
        const rect = this._e.canvas.getBoundingClientRect();
        this._mouse.x =  ((evt.clientX - rect.left) / rect.width)  * 2 - 1;
        this._mouse.y = -((evt.clientY - rect.top)  / rect.height) * 2 + 1;

        this._raycaster.setFromCamera(this._mouse, this._e.camera);
        const hits = this._raycaster.intersectObjects(this._proxyMeshes);

        if (hits.length > 0) {
          const id    = hits[0].object.userData.proxyId;
          const proxy = COMPONENT_PROXIES.find(p => p.id === id);
          if (proxy) this._showCard(proxy, evt.clientX, evt.clientY);
        } else {
          this._hideCard();
        }
      });
    }

    // ---- 3. Clipping plane controls ----------------------------------------

    _initClipControls() {
      // Insert sliders into the existing control panel if present,
      // otherwise append to body as a floating panel.
      const container = this._findOrCreateClipPanel();

      ['x', 'y', 'z'].forEach(axis => {
        const label = document.createElement('label');
        label.style.cssText = 'display:flex;align-items:center;gap:8px;color:#8ab3cc;font-size:11px;cursor:pointer;';

        const cb = document.createElement('input');
        cb.type  = 'checkbox';
        cb.id    = `clip-enable-${axis}`;
        cb.addEventListener('change', () => {
          this._clipEnabled[axis] = cb.checked;
          this._applyClipping();
          if (!cb.checked) {
            // reset plane to far away (disable)
            this._clipPlanes[axis].constant = 99;
          }
        });

        const axisLabel = document.createTextNode(`Cut ${axis.toUpperCase()}`);

        const slider = document.createElement('input');
        slider.type  = 'range';
        slider.min   = '-3';
        slider.max   = '3';
        slider.step  = '0.05';
        slider.value = '0';
        slider.style.cssText = 'flex:1;accent-color:#00c8ff;';
        slider.id    = `clip-${axis}`;
        slider.addEventListener('input', () => {
          this._clipPlanes[axis].constant = parseFloat(slider.value);
          if (this._clipEnabled[axis]) this._applyClipping();
        });

        label.appendChild(cb);
        label.appendChild(axisLabel);
        label.appendChild(slider);
        container.appendChild(label);
      });
    }

    _findOrCreateClipPanel() {
      // Try to attach to an existing sidebar/controls element
      const existing = document.getElementById('control-panel')
                    || document.getElementById('controls-panel')
                    || document.querySelector('.controls-section')
                    || document.querySelector('.sidebar');
      if (existing) {
        const section = document.createElement('div');
        section.style.cssText = 'margin-top:12px;border-top:1px solid rgba(0,200,255,0.15);padding-top:10px;';
        const title = document.createElement('div');
        title.textContent = 'Cross-Section';
        title.style.cssText = 'color:#62d4ff;font-size:11px;font-weight:600;letter-spacing:0.06em;margin-bottom:8px;';
        section.appendChild(title);
        existing.appendChild(section);
        return section;
      }

      // Floating fallback panel
      const panel = document.createElement('div');
      panel.id = 'clip-controls-panel';
      Object.assign(panel.style, {
        position:       'fixed',
        bottom:         '16px',
        right:          '16px',
        background:     'rgba(6,12,26,0.88)',
        border:         '1px solid rgba(0,200,255,0.25)',
        borderRadius:   '10px',
        padding:        '12px 16px',
        minWidth:       '200px',
        zIndex:         '8000',
        backdropFilter: 'blur(10px)',
        display:        'flex',
        flexDirection:  'column',
        gap:            '8px',
        fontFamily:     'Inter, sans-serif',
      });
      const title = document.createElement('div');
      title.textContent = 'Cross-Section Planes';
      title.style.cssText = 'color:#62d4ff;font-size:11px;font-weight:600;letter-spacing:0.06em;';
      panel.appendChild(title);
      document.body.appendChild(panel);
      return panel;
    }

    _applyClipping() {
      const activePlanes = Object.entries(this._clipEnabled)
        .filter(([, on]) => on)
        .map(([ax]) => this._clipPlanes[ax]);

      // Apply to every material in the scene that supports clippingPlanes
      this._e.scene.traverse(obj => {
        if (!obj.isMesh) return;
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        mats.forEach(mat => {
          if (!mat) return;
          mat.clippingPlanes = activePlanes;
          mat.clipShadows    = true;
          mat.needsUpdate    = true;
        });
      });
    }

    // ---- Hook into the animate loop via prototype patch ---------------------

    _hookAnimateLoop() {
      const ext   = this;
      const origAnimate = this._e.constructor.prototype.animate;

      // Patch animate so we can inject our per-frame updates
      // without modifying engine_3d.js.
      const origRender = this._e.renderer.render.bind(this._e.renderer);
      this._e.renderer.render = function (scene, camera) {
        ext._onBeforeRender();
        origRender(scene, camera);
      };
    }

    _onBeforeRender() {
      this._flashPhase += 0.016;   // ~60 fps

      const p   = this._latestPayload;
      const tel = p?.telemetry            || {};
      const drl = p?.drl_action           || p?.drl_results || {};
      const fault = p?.fault_results      || {};

      const shieldApplied = !!(drl.shield_applied);
      const severity      = fault.severity || 'NOMINAL';

      this._updateSensorMarkers(tel, shieldApplied, severity);
    }

    // ---- Public API ---------------------------------------------------------

    /** Call this with each WebSocket dispatch_payload to keep markers updated. */
    onPayload(payload) {
      this._latestPayload = payload;
    }

    /** Enable/disable a single clipping axis programmatically. */
    setClip(axis, enabled, position = 0) {
      this._clipEnabled[axis]        = enabled;
      this._clipPlanes[axis].constant = position;
      const cb = document.getElementById(`clip-enable-${axis}`);
      if (cb) cb.checked = enabled;
      const sl = document.getElementById(`clip-${axis}`);
      if (sl) sl.value = position;
      this._applyClipping();
    }
  }

  // ---- Bootstrap: attach once AeroEngine3D is ready -----------------------

  function attach(engine) {
    window._engine3DExtension = new Engine3DExtension(engine);
    console.log('[Engine3DExtension] P1 features initialised (clipping, raycasting, sensor markers).');

    // Patch the app's WS handler to forward payloads to the extension.
    // Works whether the app uses connectWebSocket() or its own handler.
    const origOnMessage = window._wsOnMessage;
    window._wsOnMessage = function (evt) {
      try {
        const data = JSON.parse(evt.data);
        window._engine3DExtension.onPayload(data);
      } catch (_) {}
      if (origOnMessage) origOnMessage(evt);
    };

    // Expose a simple API on the global for app.js to call directly:
    //   window._engine3DExtension.onPayload(dispatchPayload);
  }

  if (window._engine3D) {
    attach(window._engine3D);
  } else {
    window.addEventListener('engine3dReady', (e) => attach(e.detail));
  }
})();

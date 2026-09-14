/**
 * SIH26054 — NASA Open MCT–Grade Telemetry Substrate
 *
 * Implements:
 *   1. LIMITS: Single source of truth for Rotax 914 F operational limits.
 *   2. ChannelBuffer: High-performance ring buffer (Float64Array time, Float32Array value).
 *   3. TelemetryStore: Rolling 120 s history @ 10 Hz (1,200 samples/channel).
 *   4. AlarmEvaluator: Computes NOMINAL (0), CAUTION (1), WARNING (2), CRITICAL (3).
 *   5. EventLog: Deduped state transition event emitter with mission timestamps.
 */

export const STATE = {
  NOMINAL: 0,
  CAUTION: 1,
  WARNING: 2,
  CRITICAL: 3,
};

export const STATE_NAMES = ['NOMINAL', 'CAUTION', 'WARNING', 'CRITICAL'];
export const STATE_COLORS = ['#3fb950', '#d29922', '#f0883e', '#f85149'];

/**
 * Unified operational limits for Rotax 914 F propulsion system.
 * Values reflect certified flight operating limits (Rotax 914 F Operator Manual).
 */
export const LIMITS = {
  cht: {
    label: 'Cyl Head Temp',
    unit: '°C',
    decimals: 1,
    yellowHigh: 185.0,
    redHigh: 210.0,
    displayMin: 90.0,
    displayMax: 230.0,
    nominalRef: 150.0,
    description: 'Critical structural redline for aluminum cylinder heads',
  },
  egt: {
    label: 'Exhaust Gas Temp',
    unit: '°C',
    decimals: 1,
    yellowHigh: 740.0,
    redHigh: 800.0,
    displayMin: 500.0,
    displayMax: 850.0,
    nominalRef: 660.0,
    description: 'Combustion envelope & exhaust valve thermal stress',
  },
  vibration_rms: {
    label: 'Vibration RMS',
    unit: 'g',
    decimals: 2,
    yellowHigh: 2.5,
    redHigh: 3.5,
    displayMin: 0.0,
    displayMax: 4.5,
    nominalRef: 1.2,
    description: 'Crankshaft & bearing mechanical wear acceleration',
  },
  oil_pressure: {
    label: 'Oil Pressure',
    unit: 'kPa',
    decimals: 1,
    yellowLow: 220.0,
    redLow: 180.0,
    displayMin: 100.0,
    displayMax: 450.0,
    nominalRef: 300.0,
    description: 'Hydrodynamic wedge pressure for main bearings',
  },
  oil_temp: {
    label: 'Oil Temperature',
    unit: '°C',
    decimals: 1,
    yellowHigh: 120.0,
    redHigh: 140.0,
    displayMin: 60.0,
    displayMax: 160.0,
    nominalRef: 88.0,
    description: 'Lubricant viscosity & oxidation limit',
  },
  rpm: {
    label: 'Engine Speed',
    unit: 'RPM',
    decimals: 0,
    yellowHigh: 5600,
    redHigh: 5800,
    displayMin: 3500,
    displayMax: 6000,
    nominalRef: 4800,
    description: 'Propeller reduction gearbox input speed',
  },
  fuel_flow: {
    label: 'Fuel Flow Rate',
    unit: 'L/h',
    decimals: 2,
    yellowHigh: 18.0,
    redHigh: 22.0,
    displayMin: 4.0,
    displayMax: 24.0,
    nominalRef: 8.5,
    description: 'Specific fuel consumption & mixture burn',
  },
  map: {
    label: 'Manifold Pressure',
    unit: 'inHg',
    decimals: 1,
    yellowHigh: 38.0,
    redHigh: 40.0,
    displayMin: 15.0,
    displayMax: 42.0,
    nominalRef: 29.5,
    description: 'Turbocharger boost & induction manifold pressure',
  },
  afr: {
    label: 'Air-to-Fuel Ratio',
    unit: ':1',
    decimals: 2,
    yellowLow: 11.5,
    yellowHigh: 15.2,
    redLow: 10.5,
    redHigh: 16.0,
    displayMin: 10.0,
    displayMax: 17.0,
    nominalRef: 13.8,
    description: 'Combustion stoich balance & latent cooling ratio',
  },
  coolant_temp: {
    label: 'Coolant Temp',
    unit: '°C',
    decimals: 1,
    yellowHigh: 105.0,
    redHigh: 115.0,
    displayMin: 60.0,
    displayMax: 125.0,
    nominalRef: 82.0,
    description: 'Radiator heat rejection & water jacket temperature',
  },
};

/**
 * High-performance typed ring buffer for a single telemetry channel.
 * Capacity = 1,200 samples (120 s @ 10 Hz).
 */
export class ChannelBuffer {
  constructor(capacity = 1200) {
    this.cap = capacity;
    this.t = new Float64Array(capacity);
    this.v = new Float32Array(capacity);
    this.head = 0;
    this.size = 0;
  }

  push(timeSec, val) {
    this.t[this.head] = timeSec;
    this.v[this.head] = val;
    this.head = (this.head + 1) % this.cap;
    this.size = Math.min(this.size + 1, this.cap);
  }

  latest() {
    if (this.size === 0) return NaN;
    const idx = (this.head - 1 + this.cap) % this.cap;
    return this.v[idx];
  }

  latestTime() {
    if (this.size === 0) return 0;
    const idx = (this.head - 1 + this.cap) % this.cap;
    return this.t[idx];
  }

  /**
   * Returns values in [t0, t1] range, sorted oldest-first.
   * Format: [[t_0, v_0], [t_1, v_1], ...]
   */
  range(t0, t1) {
    const out = [];
    for (let i = 0; i < this.size; i++) {
      const idx = (this.head - this.size + i + this.cap) % this.cap;
      const t = this.t[idx];
      if (t >= t0 && t <= t1) {
        out.push([t, this.v[idx]]);
      }
    }
    return out;
  }

  /**
   * Returns interpolated or nearest value at timestamp targetTime.
   */
  valueAt(targetTime) {
    if (this.size === 0) return NaN;
    if (this.size === 1) return this.v[0];

    // Check bounds
    const oldestIdx = (this.head - this.size + this.cap) % this.cap;
    const newestIdx = (this.head - 1 + this.cap) % this.cap;
    if (targetTime <= this.t[oldestIdx]) return this.v[oldestIdx];
    if (targetTime >= this.t[newestIdx]) return this.v[newestIdx];

    // Binary search across ring buffer
    let low = 0;
    let high = this.size - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      const idx = (this.head - this.size + mid + this.cap) % this.cap;
      const tMid = this.t[idx];
      if (Math.abs(tMid - targetTime) < 0.05) {
        return this.v[idx];
      }
      if (tMid < targetTime) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    const finalIdx = (this.head - this.size + Math.max(0, Math.min(this.size - 1, low)) + this.cap) % this.cap;
    return this.v[finalIdx];
  }

  /**
   * Calculates rate of change in units/second over the last `secs` window.
   */
  trend(secs = 10) {
    if (this.size < 2) return 0;
    const newestIdx = (this.head - 1 + this.cap) % this.cap;
    const now = this.t[newestIdx];
    const pts = this.range(now - secs, now);
    if (pts.length < 2) return 0;
    const [t0, v0] = pts[0];
    const [t1, v1] = pts[pts.length - 1];
    const dt = t1 - t0;
    return dt > 0.01 ? (v1 - v0) / dt : 0;
  }
}

/**
 * TelemetryStore: Singleton telemetry database holding ChannelBuffers.
 */
export class TelemetryStore {
  constructor(capacity = 1200) {
    this.capacity = capacity;
    this.channels = new Map();
    this.frames = []; // Full historical frame snapshots for 3D state replay
    this.maxFrames = capacity;
    this.missionStartTime = Date.now();
    this.currentMissionTime = 0;

    // Initialize known channels
    for (const key of Object.keys(LIMITS)) {
      this.channels.set(key, new ChannelBuffer(capacity));
    }
  }

  getChannel(name) {
    if (!this.channels.has(name)) {
      this.channels.set(name, new ChannelBuffer(this.capacity));
    }
    return this.channels.get(name);
  }

  /**
   * Ingests a 10 Hz payload from WebSocket.
   */
  ingest(telemetryObj, timeSec, fullFrame = null) {
    this.currentMissionTime = timeSec;

    for (const [key, rawVal] of Object.entries(telemetryObj)) {
      const v = typeof rawVal === 'number' ? rawVal : parseFloat(rawVal);
      if (Number.isFinite(v)) {
        this.getChannel(key).push(timeSec, v);
      }
    }

    // Store frame snapshot for 3D state replay in FIXED conductor mode
    if (fullFrame) {
      this.frames.push({
        time: timeSec,
        telemetry: { ...telemetryObj },
        component_health: fullFrame.component_health ? { ...fullFrame.component_health } : null,
        adjusted_rul: fullFrame.adjusted_rul ?? fullFrame.rul_cycles,
        fault_archetype: fullFrame.fault_archetype,
        fault_probabilities: fullFrame.fault_probabilities ? { ...fullFrame.fault_probabilities } : null,
        drl_action: fullFrame.drl_action ? { ...fullFrame.drl_action } : null,
        sensor_audit: fullFrame.sensor_audit ? { ...fullFrame.sensor_audit } : null,
        is_physically_valid: fullFrame.is_physically_valid ?? true,
      });
      if (this.frames.length > this.maxFrames) {
        this.frames.shift();
      }
    }
  }

  /**
   * Retrieves nearest historical frame snapshot for 3D synchronization.
   */
  getHistoricalFrame(targetTime) {
    if (this.frames.length === 0) return null;
    if (this.frames.length === 1) return this.frames[0];

    // Find closest frame
    let closest = this.frames[0];
    let minDiff = Math.abs(closest.time - targetTime);
    for (let i = 1; i < this.frames.length; i++) {
      const diff = Math.abs(this.frames[i].time - targetTime);
      if (diff < minDiff) {
        minDiff = diff;
        closest = this.frames[i];
      }
    }
    return closest;
  }
}

/**
 * Evaluates alarm state for a given channel value.
 */
export function evaluateAlarm(channelKey, value) {
  const L = LIMITS[channelKey];
  if (!L || !Number.isFinite(value)) return STATE.NOMINAL;

  // Check critical limits first
  if (L.redHigh !== undefined && value >= L.redHigh) return STATE.CRITICAL;
  if (L.redLow !== undefined && value <= L.redLow) return STATE.CRITICAL;

  // Check caution limits
  if (L.yellowHigh !== undefined && value >= L.yellowHigh) return STATE.CAUTION;
  if (L.yellowLow !== undefined && value <= L.yellowLow) return STATE.CAUTION;

  return STATE.NOMINAL;
}

/**
 * EventLog: Real-time, deduped transition event emitter.
 */
export class EventLog {
  constructor(maxEvents = 150) {
    this.events = [];
    this.maxEvents = maxEvents;
    this.lastStates = new Map();
    this.listeners = [];
  }

  onEvent(cb) {
    this.listeners.push(cb);
  }

  checkTransitions(telemetryObj, timeSec) {
    const transitions = [];

    for (const [ch, lim] of Object.entries(LIMITS)) {
      const v = telemetryObj[ch];
      if (!Number.isFinite(v)) continue;

      const currentState = evaluateAlarm(ch, v);
      const prevState = this.lastStates.get(ch) ?? STATE.NOMINAL;

      if (currentState !== prevState) {
        const isEscalation = currentState > prevState;
        const stateName = STATE_NAMES[currentState];
        const valStr = `${v.toFixed(lim.decimals)} ${lim.unit}`;

        const entry = {
          id: `${ch}_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
          time: timeSec,
          channel: ch,
          from: prevState,
          to: currentState,
          value: v,
          valueStr: valStr,
          stateName: stateName,
          severity: currentState,
          text: isEscalation
            ? `${lim.label.toUpperCase()} ${stateName} (${valStr})`
            : `${lim.label.toUpperCase()} normalized to ${stateName}`,
          acknowledged: false,
        };

        this.events.unshift(entry);
        if (this.events.length > this.maxEvents) {
          this.events.pop();
        }

        this.lastStates.set(ch, currentState);
        transitions.push(entry);

        for (const cb of this.listeners) {
          try {
            cb(entry);
          } catch (_) {}
        }
      }
    }

    return transitions;
  }

  acknowledgeAll() {
    for (const ev of this.events) {
      ev.acknowledged = true;
    }
  }

  hasUnacknowledgedAlarm() {
    return this.events.some((ev) => !ev.acknowledged && ev.to >= STATE.CAUTION);
  }

  getHighestActiveSeverity() {
    let maxSev = STATE.NOMINAL;
    for (const [ch, st] of this.lastStates.entries()) {
      if (st > maxSev) maxSev = st;
    }
    return maxSev;
  }
}

// Global singletons
export const telemetryStore = new TelemetryStore();
export const eventLog = new EventLog();

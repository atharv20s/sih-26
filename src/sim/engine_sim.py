"""
Physics-based run-to-failure simulator for a MALE UAV aero piston engine.

Purpose
-------
SIH26054 requires a run-to-failure telemetry dataset over the *piston*-engine
sensor set (CHT, EGT, RPM, oil T/P, fuel flow, MAP, AFR, vibration, ...).
No public dataset provides this: C-MAPSS is a turbofan with a different sensor
set, and the IGBT dataset is electronics. This module is the honest fix: it
simulates the piston engine's thermal + mechanical + combustion subsystem with a
first-principles degradation model, so every sample carries a ground-truth RUL
label that we can actually score our models against.

Model structure
---------------
The engine is treated as a small SI piston engine (e.g. 2-stroke, ~1.0-2.0 L
displacement, 4-cylinder, typical of MALE-class UAV piston/fuel-injected powerplants).

Two correlated degradation mechanisms are added after a "burn-in" cycle:

  1. Combustion inefficiency  eta_c(t):  as the engine ages, compression loss +
     injector fouling + valve leakage reduce combustion efficiency. The heat
     released into the cylinder drops for a fixed fuel input, which RAISES EGT
     (later, less complete combustion) and requires a richer mixture (AFR rises)
     and more fuel flow to hold a given power. This also lowers the effective
     MAP (loss of volumetric efficiency).

  2. Mechanical friction/thermal resistance  mu_f(t)  and  R_th(t):  increased
     friction (bearing wear, ring wear) raises the torque needed to turn the
     engine at a given speed, and degraded heat transfer (carbon/lube breakdown)
     raises CHT for a fixed heat flux.

The physics used here is intentionally simple enough to be trained on but rich
enough to produce realistic, physically *plausible* trajectories (monotonic
degradation, cross-correlated channels, and a genuine "thermal runaway" failure
mode that a purely data-driven model would mis-predict). The PINN layer in
`src/pinn` is trained to enforce Fourier's law and monotonic RUL on exactly this
class of signal.

Failure definition
------------------
The engine fails when any of these limits is crossed (whichever happens first):
  * CHT > 260 degC        (thermal limit / knock risk)
  * EGT > 820 degC        (exhaust/valve melt-out)
  * Vibration RMS > 15.0  (bearing damage)
  * Oil pressure < 30 kPa with oil temp > 140 degC (lube starvation)
  * Cumulative mechanical damage D > 1.0
RUL is cycles-to-failure; RUL_F is cycles to the "hard" first-limit failure.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Failure limits (single source of truth for ground truth + PINN constraints)
# ---------------------------------------------------------------------------
FAILURE_LIMITS = {
    "cht": 260.0,          # degC
    "egt": 840.0,          # degC
    "vibration_rms": 15.0, # g RMS
    "oil_temp": 160.0,     # degC   (with oil_pressure < 40 kPa)
    "oil_pressure": 40.0,  # kPa
    "damage": 1.0,         # dimensionless cumulative damage
}

# Nominal baseline (all these degrade from). Ranges roughly match a MALE-class
# air-cooled / liquid-cooled piston UAV powerplant at cruise-dominant operating.
NOMINAL = {
    "rpm":         2850.0,   # cruise rpm
    "cht":         150.0,    # degC
    "egt":         640.0,    # degC
    "oil_temp":     95.0,    # degC
    "oil_pressure": 380.0,   # kPa
    "fuel_flow":     9.5,    # L/h per engine (aggregate, light aircraft scale)
    "vibration_rms": 2.5,    # g RMS
    "map":          62.0,    # kPa manifold absolute pressure (naturally aspirated)
    "afr":          14.6,    # stoichiometric-ish (slightly leaned for cruise)
    "torque":       38.0,    # N*m
    "crank_pos":     0.0,    # deg, crank angle (decorative for dashboard)
    "coolant_temp":  82.0,   # degC
}

# Sensor noise magnitudes (1-sigma, in native units) for the sensor-auditor.
SENSOR_NOISE = {
    "rpm": 30.0, "cht": 1.5, "egt": 3.0, "oil_temp": 0.8,
    "oil_pressure": 4.0, "fuel_flow": 0.05, "vibration_rms": 0.10,
    "map": 0.6, "afr": 0.08, "torque": 0.4, "crank_pos": 8.0,
    "coolant_temp": 0.6,
}

# All sensors, in the canonical order used everywhere (windows, PCA, model input).
SENSOR_ORDER = [
    "rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow",
    "vibration_rms", "map", "afr", "torque", "crank_pos", "coolant_temp",
]


def _ema(x: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Exponential moving average (light sensor smoothing)."""
    out = np.empty_like(x)
    acc = x[0]
    out[0] = acc
    for i in range(1, len(x)):
        acc = alpha * x[i] + (1 - alpha) * acc
        out[i] = acc
    return out


# ---------------------------------------------------------------------------
# Mission profile
# ---------------------------------------------------------------------------
@dataclass
class MissionProfile:
    """A segment-by-segment flight profile. Each segment has a duration in
    cycles (one cycle = one nominal ~1 s duty step at some sample rate) and a
    throttle setting [0,1]. The throttle modulates all engine quantities from
    the run-to-failure trend."""

    name: str
    segments: list  # list of (n_cycles, throttle, eta_demand)

    def total_cycles(self) -> int:
        return sum(s[0] for s in self.segments)

    @classmethod
    def cruise_dominant(cls, total_cycles: int) -> "MissionProfile":
        """Standard surveillance sortie: 91% cruise, with brief takeoff/climb
        (heavy) and occasional loiter. This keeps the model well conditioned
        on the operating point a MALE UAV actually lives at."""
        n_takeoff = max(1, int(0.02 * total_cycles))
        n_climb = max(2, int(0.07 * total_cycles))
        n_cruise = max(5, int(0.83 * total_cycles))
        n_loiter = max(2, total_cycles - n_takeoff - n_climb - n_cruise)
        return cls(
            name="cruise_dominant",
            segments=[
                (n_takeoff, 1.00, 1.00),   # takeoff / full throttle
                (n_climb, 0.85, 0.90),     # climb
                (n_cruise, 0.55, 0.65),    # cruise (endurance setting)
                (n_loiter, 0.40, 0.45),    # loiter descent / calm
            ],
        )

    @classmethod
    def testing(cls, total_cycles: int) -> "MissionProfile":
        """Heavier, more varied test profile (more high-throttle segments) used
        to stress the model out-of-distribution. Mimics a hot, high-load sortie."""
        n_takeoff = max(1, int(0.04 * total_cycles))
        n_climb = max(3, int(0.12 * total_cycles))
        n_cruise = max(5, int(0.68 * total_cycles))
        n_loiter = max(2, total_cycles - n_takeoff - n_climb - n_cruise)
        return cls(
            name="testing",
            segments=[
                (n_takeoff, 1.00, 1.00),
                (n_climb, 0.95, 0.97),
                (n_cruise, 0.70, 0.78),
                (n_loiter, 0.55, 0.60),
            ],
        )


# ---------------------------------------------------------------------------
# Engine run-to-failure simulation
# ---------------------------------------------------------------------------
@dataclass
class EngineSim:
    """Simulate a single engine's run-to-failure telemetry.

    Parameters
    ----------
    total_cycles : int
        Number of cycles (time steps) to simulate.
    seed : int
        Random seed for the sensor noise + degradation stochasticity.
    profile : MissionProfile
        Operating profile.
    initial_health : float in (0, 1]
        Starting health. 1.0 = brand new, e.g. 0.85 = already 15% through life.
    degradation_driver : float
        Multiplier on the base degradation rate (unit = 1.0). Set >1 to make
        a fast-failing engine, <1 to make a long-lived one. Useful to widen the
        RUL distribution across the fleet.
    failure_bias : dict
        Per-mechanism exponent multipliers that steer which failure mode fires
        first. Keys match the names in _degradation_signals (eta_c, mu_f, r_th,
        eta_v, eta_lub, vib). A multiplier < 1 makes that mechanism degrade
        faster (loss^(base_exp * mult) is larger for 0<loss<1 when mult<1).
        Leave empty for the default vibration-dominant behaviour.
    n_cylinders : int = 4
        Used only to scale fuel-flow/torque into realistic SI units.
    """

    total_cycles: int
    seed: int = 0
    profile: MissionProfile = None
    initial_health: float = 1.0
    degradation_driver: float = 1.0
    failure_bias: dict = field(default_factory=dict)
    n_cylinders: int = 4

    # internal state (filled by run_segments)
    data: dict = field(default_factory=dict)
    truth: dict = field(default_factory=dict)   # ground-truth RUL + health + failure cycle
    _rng: np.random.Generator = None
    _seg_labels: np.ndarray = None

    def __post_init__(self):
        if self.profile is None:
            self.profile = MissionProfile.cruise_dominant(self.total_cycles)
        self._rng = np.random.default_rng(self.seed)

    # ---- health deterministic process --------------------------------------
    def _health_trajectory(self, n: int) -> np.ndarray:
        """Health h(t) in [0,1], monotonic non-increasing, with a realistic
        wear curve: gentle early (low degradation rate) and a steep accelerating
        failure tail near end-of-life.

        The engine reaches health ~0 at the END of the requested window, so the
        whole run shows meaningful degradation and the ground-truth RUL spans
        (n-1 ... 0). The `degradation_driver` scales how quickly: driver>1 makes
        the engine fail BEFORE the window ends (RUL hits 0 mid-run), driver<1
        makes it survive the window (RUL stays > 0 everywhere). This produces a
        heterogeneous fleet with a real RUL tail.
        """
        # Life in cycles. driver>1 => shorter life => engine fails before the
        # window; driver<1 => longer life => survives past the window.
        life = float(n) / float(max(self.degradation_driver, 0.2))
        # The failure point (health ~0) sits at 'life'. The wear curve is a
        # logistic-like ramp that is ~flat early and steep near end-of-life.
        t = np.arange(n, dtype=float)
        # Clamp t beyond life so health stays pinned at ~0 (already failed).
        tc = np.clip(t, 0.0, life)
        # Logistic in log space -> broad slow decay then sharp tail.
        # Normalized so h at t=0 equals initial_health and h at t=life ~ 0.
        x = (tc / max(life, 1.0))          # 0..1 (then clipped)
        # Baseline s-shaped wear: gentle start, mild middle, steep end.
        # (1 - x^1.6) alone bottoms at 0 only at x=1, so multiply by a logistic
        # tail that drives h -> ~0 in the final ~20% of life.
        base = 1.0 - x**1.6
        tail = 1.0 / (1.0 + np.exp((x - 0.80) / 0.08))
        h = base * tail
        # Renormalize so h(0)==initial_health and h(life)~=0.
        h = h / np.maximum(h[0], 1e-9) * self.initial_health
        h = np.clip(h, 0.0, 1.0)
        # Enforce strict monotonic non-increasing (required for the PINN's
        # monotonic-decreasing RUL constraint to be well-posed).
        for i in range(1, n):
            if h[i] > h[i - 1]:
                h[i] = h[i - 1]
        return h

    def _degradation_signals(self, h: np.ndarray) -> dict:
        """From health trajectory, derive the *physical* degradation factors
        that shift every sensor channel. All are 1.0 at health=1.

        `failure_bias` maps a mechanism name to {"coef": m, "exp": m}:
          - "coef" scales the degradation magnitude (how far the mechanism can
            push its channel at end-of-life). Needed because some limits are
            unreachable at the base coefficient — e.g. EGT tops out near 792 degC
            against an 840 degC limit unless eta_c is allowed to fall further.
          - "exp" scales the exponent; < 1 accelerates onset (loss^(e*m) is
            larger for 0<loss<1 when m<1), > 1 delays it.

        Steering a mode therefore means boosting the target mechanism AND
        delaying its competitors, since _find_failure returns whichever limit is
        crossed earliest in time.
        """
        loss = 1.0 - h                        # 0 at new, 1 at dead
        b = self.failure_bias                  # mechanism -> {"coef","exp"}

        def cx(name, base_coef, base_exp):
            """Resolve (coefficient, exponent) for one mechanism."""
            spec = b.get(name, {})
            return (base_coef * spec.get("coef", 1.0),
                    base_exp * spec.get("exp", 1.0))

        out = {}
        # Combustion inefficiency: grows non-linearly near end-of-life.
        c, e = cx("eta_c", 0.70, 1.7)
        out["eta_c"] = 1.0 - c * loss**e

        # Mechanical friction: grows with wear (ring/bearing), accelerates late.
        c, e = cx("mu_f", 2.10, 1.4)
        out["mu_f"] = 1.0 + c * loss**e

        # Thermal resistance to coolant/ambient: grows -> CHT rises.
        c, e = cx("r_th", 2.80, 1.6)
        out["r_th"] = 1.0 + c * loss**e

        # Volumetric efficiency / breathing loss.
        c, e = cx("eta_v", 0.35, 1.3)
        out["eta_v"] = 1.0 - c * loss**e

        # Lubrication effectiveness.
        c, e = cx("eta_lub", 0.75, 1.4)
        out["eta_lub"] = 1.0 - c * loss**e

        # Accumulated vibration rise (unbalance + clearance).
        c, e = cx("vib", 7.0, 1.3)
        out["vib"] = 1.0 + c * loss**e

        # Keep efficiency factors physically sane (never <= 0).
        out["eta_c"] = np.maximum(out["eta_c"], 0.12)
        out["eta_v"] = np.maximum(out["eta_v"], 0.15)
        out["eta_lub"] = np.maximum(out["eta_lub"], 0.10)
        return out

    # ---- single-cycle sensor model -----------------------------------------
    def _sensor_step(self, throttle: float, eta_demand: float, deg: dict) -> dict:
        """Produce the *true* (noiseless) sensor values for one cycle from the
        degradation factors `deg` (all 1.0 at health=1). Vectorized-friendly:
        `throttle`, `eta_demand` and the entries of `deg` may be scalars OR
        arrays of length n (see run()); returns scalar or (n,) arrays.

        Design principle: the degradation factors dominate the trajectory so the
        time series is monotonic-in-degradation (this is what makes the PINN's
        monotonic-RUL constraint physically grounded and learnable). Throttle
        only places the operating point on top of that trend.

        References for the physical relations (from the reviewed papers + the
        blueprint's math):
          - Fourier's law / thermal conduction  -> CHT = f(heat_in, R_th)
          - Gas law + exhaust energy             -> EGT rises as combustion
            becomes late/incomplete (eta_c falls)
          - Volumetric efficiency (breathing)    -> MAP ~ eta_v
          - Fuel-air mixing                       -> AFR riches as eta_c falls
          - Mechanical friction                    -> torque & fuel_flow rise
          - Unbalance/clearance growth             -> vibration_rms rises
        """
        # Power demand: throttle * mission demand.
        power = throttle * eta_demand

        # Degradation state: 0 (healthy) -> 1 (dead).
        loss = 1.0 - np.clip(getattr(self, "_last_health", 1.0), 0.0, 1.0)
        eta_c = deg["eta_c"]          # combustion efficiency factor
        mu_f = deg["mu_f"]            # mechanical friction factor
        r_th = deg["r_th"]            # thermal resistance factor
        eta_v = deg["eta_v"]          # volumetric efficiency factor
        eta_lub = deg["eta_lub"]      # lubrication effectiveness factor
        vib = deg["vib"]              # vibration growth factor

        seg = {}

        # RPM: raw engine speed from the sensor (auto/throttle), low at low
        # throttle. Chosen so a healthy cruise sits near 2850 rpm.
        seg["rpm"] = NOMINAL["rpm"] * (0.60 + 0.62 * throttle)

        # Heat released into the cylinder (scale tuned so healthy cruise heat
        # is modest and end-of-life is hot enough to trip CHT/EGT).
        heat = power * (1.0 / np.maximum(eta_c, 0.25)) * 0.85

        # CHT: Fourier-style conduction -> heat * thermal resistance + rpm term.
        seg["cht"] = NOMINAL["cht"] + heat * r_th * 55.0 + 0.30 * (seg["rpm"] / NOMINAL["rpm"]) * 10.0

        # EGT: rises with residual heat (late/incomplete burn).
        seg["egt"] = NOMINAL["egt"] + heat * (1.0 / np.maximum(eta_c, 0.25)) * 45.0

        # Oil temp: heat + friction work.
        seg["oil_temp"] = NOMINAL["oil_temp"] + heat * 18.0 + (mu_f - 1.0) * 90.0

        # Oil pressure: drops as lube degrades and oil heats up.
        seg["oil_pressure"] = NOMINAL["oil_pressure"] * np.maximum(eta_lub, 0.3) \
            * (1.0 - 0.25 * np.maximum(0.0, seg["oil_temp"] - 100.0) / 70.0)

        # Fuel flow: needed fuel to hold power given combustion health.
        # Scale chosen so a HEALTHY cruise yields ~NOMINAL["fuel_flow"] (9.5).
        seg["fuel_flow"] = NOMINAL["fuel_flow"] * power * 2.60 \
            * (1.0 / np.maximum(eta_c, 0.25) - 0.25) * (1.0 + (mu_f - 1.0) * 0.9)

        # Vibration: unbalance/clearance growth, modulated by load.
        seg["vibration_rms"] = NOMINAL["vibration_rms"] * vib * (0.70 + 0.55 * throttle)

        # MAP: volumetric efficiency (and throttle).
        seg["map"] = NOMINAL["map"] * eta_v * (0.50 + 0.75 * throttle)

        # AFR: mixture richens as combustion degrades (leading to knock risk).
        # Stoichiometric ~14.6; a healthy engine runs slightly leaned.
        seg["afr"] = NOMINAL["afr"] * (1.05 - 0.10 * throttle) \
            * np.maximum(1.0 - 0.35 * (1.0 - eta_c), 0.60)

        # Torque: mechanical output; rises with friction (more torque needed to
        # hold speed) at a given power. Scale so healthy cruise ~ NOMINAL (38).
        seg["torque"] = NOMINAL["torque"] * power * mu_f * 3.0

        # Coolant temp (liquid-cooled bank) tracks CHT loosely.
        seg["coolant_temp"] = NOMINAL["coolant_temp"] + (seg["cht"] - NOMINAL["cht"]) * 0.60

        # Crank angle: decorative phase sensor for the dashboard.
        if np.ndim(throttle) == 0:
            seg["crank_pos"] = float(self._rng.uniform(0.0, 720.0))
        else:
            seg["crank_pos"] = self._rng.uniform(0.0, 720.0, size=len(throttle))

        return seg

    # ---- main entry ---------------------------------------------------------
    def run(self, add_noise: bool = True) -> dict:
        """Simulate the full run-to-failure telemetry.

        Returns
        -------
        tuple (data, truth):
            data  : dict[str, np.ndarray], keys = SENSOR_ORDER, len = total_cycles
            truth : dict with standard keys:
                'rul'          : float array, cycles-to-failure (int)
                'rul_f'        : float array, cycles to FIRST failure limit
                'health'       : float array, h(t) in [0,1]
                'failure_cycle': int, the cycle at which it failed
                'failure_type' : str, the limit that triggered it
                'segments'     : int array, per-cycle segment index
        """
        n = self.total_cycles
        h = self._health_trajectory(n)
        deg = self._degradation_signals(h)

        # Build per-cycle throttle/eta from the mission profile (segment based).
        segs = self.profile.segments
        throttle = np.zeros(n)
        eta_dem = np.zeros(n)
        thumbs = np.zeros(n, dtype=int)
        idx = 0
        for si, (dur, thr, eta) in enumerate(segs):
            throttle[idx:idx + dur] = thr
            eta_dem[idx:idx + dur] = eta
            thumbs[idx:idx + dur] = si
            idx += dur
        # Safety: if profile shorter than n, fill with cruise.
        if idx < n:
            throttle[idx:] = 0.55
            eta_dem[idx:] = 0.65
            thumbs[idx:] = segs[-1][0]

        self._seg_labels = thumbs

        # ---- fully vectorized sensor render ---------------------------------
        # All degradation factors are length-n arrays; _sensor_step is
        # elementwise, so we compute all cycles at once.
        self._last_health = h
        seg = self._sensor_step(throttle, eta_dem, {
            "eta_c": deg["eta_c"], "mu_f": deg["mu_f"], "r_th": deg["r_th"],
            "eta_v": deg["eta_v"], "eta_lub": deg["eta_lub"], "vib": deg["vib"],
        })

        cols = {k: np.asarray(seg[k], dtype=float) for k in SENSOR_ORDER}
        # Add sensor noise (monte-carlo per engine).
        if add_noise:
            for k in SENSOR_ORDER:
                if k in SENSOR_NOISE:
                    cols[k] += self._rng.normal(0.0, SENSOR_NOISE[k], size=n)
        # Physics clamp: non-negative for physical quantities (crank_pos may be
        # any phase, but keep it too for readability).
        for k in SENSOR_ORDER:
            if k != "crank_pos":
                cols[k] = np.maximum(cols[k], 0.0)
        # Apply a light EMA over cyclic noise so the telemetry looks like a real
        # sensor bus rather than iid white noise (keeps the audit task honest).
        alpha = 0.15
        for k in SENSOR_ORDER:
            cols[k] = _ema(cols[k], alpha)

        # ---- ground truth RUL ------------------------------------------------
        # Determine failure cycle = first cycle crossing any hard limit.
        fc = self._find_failure(cols)
        rul = np.maximum(fc - np.arange(n), 0).astype(float)
        rul_f = np.maximum(fc - np.arange(n), 0).astype(float)

        truth = {
            "rul": rul,
            "rul_f": rul_f,
            "health": h,
            "failure_cycle": int(fc),
            "failure_type": self._failure_kind(cols, fc),
            "segments": thumbs.copy(),
        }
        self.data = cols
        self.truth = truth
        return cols, truth

    def _find_failure(self, cols: dict) -> int:
        """First cycle index where any failure limit is exceeded."""
        n = len(cols["cht"])
        # damage accumulation normalized to fail at end-of-life (health threshold)
        # First, find health-based failure (last cycle where health still > 0-ish)
        for t in range(n):
            if cols["cht"][t] > FAILURE_LIMITS["cht"]:
                return t
            if cols["egt"][t] > FAILURE_LIMITS["egt"]:
                return t
            if cols["vibration_rms"][t] > FAILURE_LIMITS["vibration_rms"]:
                return t
            if cols["oil_temp"][t] > FAILURE_LIMITS["oil_temp"] and \
               cols["oil_pressure"][t] < FAILURE_LIMITS["oil_pressure"]:
                return t
        return n - 1

    def _failure_kind(self, cols: dict, fc: int) -> str:
        if fc >= len(cols["cht"]) - 1:
            return "none"
        t = fc
        if cols["cht"][t] > FAILURE_LIMITS["cht"]:
            return "cht_over"
        if cols["egt"][t] > FAILURE_LIMITS["egt"]:
            return "egt_over"
        if cols["vibration_rms"][t] > FAILURE_LIMITS["vibration_rms"]:
            return "vibration_over"
        if cols["oil_temp"][t] > FAILURE_LIMITS["oil_temp"] and \
           cols["oil_pressure"][t] < FAILURE_LIMITS["oil_pressure"]:
            return "oil_starvation"
        return "health_exhausted"


# ---------------------------------------------------------------------------
# Fleet generating convenience
# ---------------------------------------------------------------------------
def generate_fleet(n_engines: int, cycles_per_engine: int, seed: int = 0,
                   profile: str = "cruise_dominant",
                   health_range=(0.55, 1.0), driver_range=(0.85, 1.30)) -> list:
    """Generate a heterogeneous fleet of engines with varied starting health,
    degradation rate, and failure archetypes. Returns a list of (engine_id, data, truth).

    Four failure archetypes are distributed roughly equally across the fleet:
      - vibration_over  (default bearing/ring wear — exponent unchanged)
      - cht_over        (thermal runaway — r_th accelerated)
      - egt_over        (combustion failure — eta_c accelerated)
      - oil_starvation  (lube breakdown — eta_lub + mu_f accelerated)

    The diversity lets the PINN's physics guardian and the DRL fault classifier
    distinguish fault modes from sensor trajectories alone — directly matching
    the DRDO requirement for fault *prediction* across failure types.
    """
    # Per-archetype mechanism biasing: {"coef": magnitude, "exp": onset speed}.
    # Each archetype boosts its target mechanism AND delays the competitors,
    # because _find_failure returns whichever limit is crossed earliest in time.
    # Values tuned against scripts/probe_sim.py so all four modes actually fire.
    ARCHETYPES = [
        # vibration_over — bearing/ring unbalance reaches 15 g first.
        {"vib": {"exp": 0.55}},

        # cht_over — thermal runaway: conduction path degrades, CHT hits 260 C.
        {"r_th": {"coef": 1.40, "exp": 0.50},
         "vib": {"exp": 3.00}},

        # egt_over — combustion failure: eta_c must fall below ~0.26 for EGT to
        # be able to reach 840 C at cruise, so its coefficient is raised.
        {"eta_c": {"coef": 1.20, "exp": 0.50},
         "r_th": {"coef": 0.22, "exp": 3.00},
         "vib": {"exp": 3.50}},

        # oil_starvation — lube breakdown + friction: oil_temp > 160 C while
        # oil_pressure < 40 kPa.
        {"eta_lub": {"coef": 1.30, "exp": 0.45},
         "mu_f": {"coef": 1.30, "exp": 0.50},
         "r_th": {"coef": 0.25, "exp": 3.00},
         "vib": {"exp": 3.50}},
    ]

    rng = np.random.default_rng(seed)
    profiles = {
        "cruise_dominant": MissionProfile.cruise_dominant,
        "testing": MissionProfile.testing,
    }
    prof_fn = profiles[profile]

    engines = []
    for i in range(n_engines):
        h0 = float(rng.uniform(*health_range))
        drv = float(rng.uniform(*driver_range))
        prof = prof_fn(cycles_per_engine)
        bias = ARCHETYPES[i % len(ARCHETYPES)]
        sim = EngineSim(
            total_cycles=cycles_per_engine, seed=seed * 1000 + i,
            profile=prof, initial_health=h0, degradation_driver=drv,
            failure_bias=bias,
        )
        data, truth = sim.run(add_noise=True)
        engines.append((f"eng_{i:03d}", data, truth))
    return engines


if __name__ == "__main__":
    # Quick smoke test: one engine, report channels + RUL range.
    sim = EngineSim(total_cycles=600, seed=42, profile=MissionProfile.cruise_dominant(600),
                    initial_health=0.92, degradation_driver=1.05)
    data, truth = sim.run()
    print("cycles:", sim.total_cycles)
    print("failure_cycle:", truth["failure_cycle"], "type:", truth["failure_type"])
    print("RUL range:", truth["rul"][0], "->", truth["rul"][-1])
    print("Example first cycle:", {k: round(float(data[k][0]), 2) for k in SENSOR_ORDER})
    print("Example last cycle:", {k: round(float(data[k][-1]), 2) for k in SENSOR_ORDER})

"""Probe the simulator: at each health level, what do sensors read, and which
failure limit trips first? Use this to tune degradation multipliers so failures
are realistic and varied across the fleet."""
import numpy as np
import sys
sys.path.insert(0, "src")
from sim.engine_sim import EngineSim, MissionProfile, FAILURE_LIMITS, NOMINAL, SENSOR_ORDER


def probe_health(h_vals=(1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05),
                 throttle=0.55, eta=0.65, seed=1):
    sim = EngineSim(total_cycles=100, seed=seed, profile=MissionProfile.cruise_dominant(100),
                    initial_health=1.0, degradation_driver=1.0)
    sim._rng = np.random.default_rng(seed)
    print(f"{'health':>6} {'cht':>7} {'egt':>7} {'oilT':>7} {'oilP':>7} "
          f"{'fuel':>6} {'vib':>6} {'map':>6} {'afr':>6} {'trq':>7} {'flag':>10}")
    for h in h_vals:
        sim._last_health = h
        deg = sim._degradation_signals(np.array([h]))
        seg = sim._sensor_step(throttle, eta, {k: float(v[0]) for k, v in deg.items()})
        flag = ""
        if seg["cht"] > FAILURE_LIMITS["cht"]: flag = "CHT!"
        if seg["egt"] > FAILURE_LIMITS["egt"]: flag += " EGT!"
        if seg["vibration_rms"] > FAILURE_LIMITS["vibration_rms"]: flag += " VIB!"
        if seg["oil_temp"] > FAILURE_LIMITS["oil_temp"] and \
           seg["oil_pressure"] < FAILURE_LIMITS["oil_pressure"]: flag += " OIL!"
        print(f"{h:>6.2f} {seg['cht']:>7.1f} {seg['egt']:>7.1f} {seg['oil_temp']:>7.1f} "
              f"{seg['oil_pressure']:>7.1f} {seg['fuel_flow']:>6.2f} {seg['vibration_rms']:>6.2f} "
              f"{seg['map']:>6.1f} {seg['afr']:>6.2f} {seg['torque']:>7.1f} {flag:>10} ({flag.strip()})")


if __name__ == "__main__":
    probe_health()

"""Aero Piston Engine Prognostic & Control Environment (SIH26054).

Simulates degrading MALE UAV piston engine dynamics with continuous control:
  - State Space (S): Sensor telemetry + PINN physical gradient + PINN RUL + Health index.
  - Action Space (A): Continuous [delta_throttle, delta_mixture] adjustments.
  - PINN Safety Shield: Hard thermodynamic boundary filter preventing catastrophic actions.
  - Reward Function: Rewards flight duration extension while penalizing material thermal & vibration stress.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Safe physical thresholds for MALE UAV aero piston engine
CHT_WARN = 185.0       # °C (approaching threshold)
CHT_CRIT = 210.0       # °C (structural damage limit)
EGT_CRIT = 800.0       # °C (combustion runaway limit)
VIB_WARN = 2.5         # g (bearing wear acceleration)
VIB_CRIT = 3.5         # g (mechanical failure threshold)
OIL_MIN  = 180.0       # kPa (minimum hydrodynamic wedge pressure)


@dataclass
class EngineState:
    cycle: int
    rpm: float
    cht: float
    egt: float
    oil_temp: float
    oil_pressure: float
    fuel_flow: float
    vibration_rms: float
    map_kpa: float
    afr: float
    torque: float
    crank_pos: float
    coolant_temp: float
    pinn_gradient: float
    pinn_rul: float
    health: float


class AeroEngineEnv:
    """Prognostic control environment for deep reinforcement learning."""

    def __init__(self, failure_archetype: str = "cht_over", max_cycles: int = 400):
        self.failure_archetype = failure_archetype
        self.max_cycles = max_cycles
        self.current_cycle = 0
        self.health = 1.0
        self.base_throttle = 0.72   # cruise throttle
        self.base_mixture = 13.8    # air-to-fuel ratio
        self.throttle = self.base_throttle
        self.afr = self.base_mixture

        # Nominal operating point
        self.rpm = 4800.0
        self.cht = 150.0
        self.egt = 650.0
        self.oil_temp = 85.0
        self.oil_pressure = 320.0
        self.fuel_flow = 9.5
        self.vibration_rms = 1.2
        self.map_kpa = 92.0
        self.torque = 24.0
        self.coolant_temp = 78.0
        self.pinn_gradient = 0.0
        self.pinn_rul = float(max_cycles)

        # Safety shield tracking
        self.shield_interventions = 0

    def reset(self, failure_archetype: str = None) -> np.ndarray:
        """Reset engine to nominal start of degradation mission."""
        if failure_archetype:
            self.failure_archetype = failure_archetype
        self.current_cycle = 0
        self.health = 1.0
        self.throttle = self.base_throttle
        self.afr = self.base_mixture
        self.rpm = 4800.0 + np.random.uniform(-50, 50)
        self.cht = 150.0 + np.random.uniform(-2, 2)
        self.egt = 650.0 + np.random.uniform(-5, 5)
        self.oil_temp = 85.0 + np.random.uniform(-1, 1)
        self.oil_pressure = 320.0 + np.random.uniform(-5, 5)
        self.fuel_flow = 9.5 + np.random.uniform(-0.2, 0.2)
        self.vibration_rms = 1.2 + np.random.uniform(-0.05, 0.05)
        self.map_kpa = 92.0
        self.coolant_temp = 78.0
        self.pinn_gradient = 0.0
        self.pinn_rul = float(self.max_cycles)
        self.shield_interventions = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Vector observation (15-dim state space)."""
        return np.array([
            (self.rpm - 4500.0) / 1000.0,
            (self.cht - 150.0) / 50.0,
            (self.egt - 650.0) / 100.0,
            (self.oil_temp - 85.0) / 30.0,
            (self.oil_pressure - 300.0) / 100.0,
            (self.fuel_flow - 9.5) / 5.0,
            (self.vibration_rms - 1.2) / 1.5,
            (self.map_kpa - 90.0) / 20.0,
            (self.afr - 13.5) / 2.0,
            (self.coolant_temp - 78.0) / 20.0,
            self.throttle,
            self.pinn_gradient,
            self.pinn_rul / 400.0,
            self.health,
            1.0 if self.failure_archetype == "cht_over" else 0.0,
        ], dtype=np.float32)

    def apply_pinn_safety_shield(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """PINN Safety Shield: Prevents hazardous exploratory RL actions.

        Predicts one-step thermal & vibration consequence before actuation.
        If action would breach thermal CHT > 210°C or vibration > 3.5g,
        action is bounded to the safe physical operating envelope.
        """
        d_thr, d_mix = action[0], action[1]
        cand_throttle = np.clip(self.throttle + d_thr, 0.50, 0.95)
        cand_afr = np.clip(self.afr + d_mix * 2.0, 11.5, 15.5)

        # Physics projection:
        # Leaner AFR (> 14.5) and high throttle surges CHT & EGT
        lean_surge = max(0.0, cand_afr - 13.8) * 12.0
        thr_surge = (cand_throttle - 0.70) * 45.0
        projected_cht = self.cht + thr_surge + lean_surge

        # High throttle increases vibration
        projected_vib = self.vibration_rms + (cand_throttle - 0.70) * 0.8

        shield_active = False
        safe_action = np.copy(action)

        # Check thermal violation
        if projected_cht > CHT_CRIT:
            shield_active = True
            # Restrict throttle advance and enrich mixture (cool the cylinders)
            safe_action[0] = min(safe_action[0], -0.05)
            safe_action[1] = min(safe_action[1], -0.05)  # enrich mixture

        # Check vibration violation
        if projected_vib > VIB_CRIT:
            shield_active = True
            safe_action[0] = min(safe_action[0], -0.08)

        if shield_active:
            self.shield_interventions += 1

        return safe_action, shield_active

    def step(self, action: np.ndarray, use_shield: bool = True) -> tuple[np.ndarray, float, bool, dict]:
        """Execute action, advance degradation, and calculate reward."""
        self.current_cycle += 1

        shield_triggered = False
        if use_shield:
            action, shield_triggered = self.apply_pinn_safety_shield(action)

        d_throttle = float(np.clip(action[0], -0.15, 0.15))
        d_mixture  = float(np.clip(action[1], -0.10, 0.10))

        self.throttle = float(np.clip(self.throttle + d_throttle, 0.50, 0.95))
        self.afr = float(np.clip(self.afr + d_mixture * 1.5, 11.8, 15.2))

        # Base degradation rate per cycle
        deg_step = 0.0025

        # Mitigating effects of DRL policy:
        # Lowering throttle reduces thermal & mechanical wear
        # Richer mixture (lower AFR) lowers CHT
        thermal_alleviation = max(0.0, (0.75 - self.throttle)) * 0.0015
        if self.afr < 13.5:
            thermal_alleviation += (13.5 - self.afr) * 0.0008

        self.health = max(0.0, self.health - (deg_step - thermal_alleviation))

        # Dynamic physical updates based on degradation archetype
        deg_fraction = 1.0 - self.health

        # Engine physics response
        self.rpm = 4500.0 + self.throttle * 900.0 - deg_fraction * 200.0
        self.fuel_flow = 7.0 + self.throttle * 4.5

        # CHT thermodynamics (Fourier-inspired):
        # heat source proportional to fuel_flow, cooling proportional to (CHT - coolant_temp)
        q_combustion = self.fuel_flow * (14.0 / self.afr) * 12.0
        q_dissipation = 0.65 * (self.cht - self.coolant_temp)

        if self.failure_archetype == "cht_over":
            # Cooling degradation causes thermal runaway
            cht_bias = deg_fraction ** 2 * 90.0
        else:
            cht_bias = deg_fraction * 20.0

        dcht = 0.15 * (q_combustion - q_dissipation) + (self.throttle - 0.72) * 15.0 + cht_bias * 0.05
        self.cht = float(np.clip(self.cht + dcht, 90.0, 240.0))
        self.pinn_gradient = dcht

        # EGT response
        if self.failure_archetype == "egt_over":
            egt_bias = deg_fraction ** 2 * 180.0
        else:
            egt_bias = deg_fraction * 30.0
        self.egt = float(np.clip(580.0 + self.throttle * 120.0 + (self.afr - 13.0) * 25.0 + egt_bias, 500.0, 900.0))

        # Vibration response
        if self.failure_archetype == "vibration_over":
            vib_bias = deg_fraction ** 2.2 * 3.2
        else:
            vib_bias = deg_fraction * 0.5
        self.vibration_rms = float(np.clip(1.0 + (self.throttle - 0.5) * 0.8 + vib_bias, 0.8, 4.5))

        # Oil pressure response
        if self.failure_archetype == "oil_starvation":
            self.oil_pressure = float(max(100.0, 320.0 - deg_fraction * 220.0))
            self.oil_temp = float(85.0 + deg_fraction * 50.0)
        else:
            self.oil_pressure = float(320.0 - deg_fraction * 30.0)
            self.oil_temp = float(85.0 + deg_fraction * 15.0)

        # Update remaining useful life estimate
        self.pinn_rul = max(0.0, self.health * self.max_cycles)

        # Check termination / failure criteria
        failed = (
            self.cht >= CHT_CRIT or
            self.vibration_rms >= VIB_CRIT or
            self.oil_pressure <= OIL_MIN or
            self.egt >= EGT_CRIT or
            self.health <= 0.02
        )
        done = failed or (self.current_cycle >= self.max_cycles)

        # Multi-objective reward calculation:
        # 1. Flight endurance reward (+1.0 for each safe operating cycle)
        reward = 1.0

        # 2. Material wear penalties (quadratic stress penalty)
        if self.cht > CHT_WARN:
            reward -= 5.0 * ((self.cht - CHT_WARN) / (CHT_CRIT - CHT_WARN)) ** 2
        if self.vibration_rms > VIB_WARN:
            reward -= 5.0 * ((self.vibration_rms - VIB_WARN) / (VIB_CRIT - VIB_WARN)) ** 2
        if self.oil_pressure < 220.0:
            reward -= 4.0 * ((220.0 - self.oil_pressure) / 40.0) ** 2

        # 3. Terminal penalty or survival bonus
        if failed:
            reward -= 50.0
        elif self.current_cycle >= self.max_cycles:
            reward += 30.0  # Mission safely completed!

        info = {
            "cycle": self.current_cycle,
            "health": self.health,
            "rul": self.pinn_rul,
            "shield_triggered": shield_triggered,
            "shield_total": self.shield_interventions,
            "failed": failed,
            "cht": self.cht,
            "vibration": self.vibration_rms,
            "oil_pressure": self.oil_pressure,
        }

        return self._get_observation(), reward, done, info

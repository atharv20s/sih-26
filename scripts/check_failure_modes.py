"""Verify the simulator produces all four failure archetypes.

Run:  python scripts/check_failure_modes.py
"""
import sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim.engine_sim import generate_fleet, FAILURE_LIMITS

fleet = generate_fleet(40, 1200, seed=7, profile="cruise_dominant")

counts = collections.Counter(t["failure_type"] for _, _, t in fleet)
print("failure mode mix:", dict(counts))
print()
print("limits:", FAILURE_LIMITS)
print()
hdr = (f"{'engine':8s} {'arch':4s} {'type':16s} {'fc':>5s} "
       f"{'cht':>7s} {'egt':>7s} {'vib':>6s} {'oilT':>6s} {'oilP':>6s}")
print(hdr)
print("-" * len(hdr))
for i in range(8):
    eid, data, t = fleet[i]
    print(f"{eid:8s} {i % 4:<4d} {t['failure_type']:16s} {t['failure_cycle']:5d} "
          f"{data['cht'].max():7.1f} {data['egt'].max():7.1f} "
          f"{data['vibration_rms'].max():6.2f} {data['oil_temp'].max():6.1f} "
          f"{data['oil_pressure'].min():6.1f}")

missing = {"vibration_over", "cht_over", "egt_over", "oil_starvation"} - set(counts)
print()
if missing:
    print("MISSING MODES:", missing)
else:
    print("OK: all four failure modes present")

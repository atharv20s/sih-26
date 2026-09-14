"""Master training pipeline -- runs all steps in order.

Steps:
  1. build_dataset.py    -> data/processed/ (sliding-window arrays + parquet)
  2. train_pinn.py       -> models/checkpoints/pinn_best.pt (+ cnn baselines)
  3. train_classifier.py -> models/checkpoints/fault_classifier.pt
  4. drl_agent.py        -> models/checkpoints/drl_policy.pt

Estimated time (CPU only, default settings):
  build_dataset    ~1-2 min
  train_pinn       ~15-25 min  (60 epochs, 3 models)
  train_classifier ~3-5 min   (40 epochs)
  drl_agent        ~3-8 min   (200 episodes x 300 steps)
  Total            ~22-40 min

Usage (from project root):
    python scripts/run_all_training.py
    python scripts/run_all_training.py --pinn-epochs 30 --drl-episodes 100  # fast demo run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], step_name: str):
    """Run a subprocess command; exit on failure."""
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  STEP: {step_name}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[run_all_training] [FAILED] {step_name} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n[run_all_training] [OK] {step_name} done in {elapsed:.0f}s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engines",      type=int, default=80,  help="Engines per archetype (build_dataset)")
    p.add_argument("--cycles",       type=int, default=600, help="Cycles per engine (build_dataset)")
    p.add_argument("--window",       type=int, default=30,  help="Window length T (build_dataset)")
    p.add_argument("--pinn-epochs",  type=int, default=60,  help="Epochs for PINN + baseline training")
    p.add_argument("--clf-epochs",   type=int, default=40,  help="Epochs for fault classifier")
    p.add_argument("--drl-episodes", type=int, default=200, help="Training episodes for DRL agent")
    p.add_argument("--skip-build",   action="store_true",   help="Skip dataset generation (reuse existing)")
    p.add_argument("--skip-pinn",    action="store_true",   help="Skip PINN training")
    p.add_argument("--skip-clf",     action="store_true",   help="Skip classifier training")
    p.add_argument("--skip-drl",     action="store_true",   help="Skip DRL training")
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    t_start = time.time()
    print(f"\n[run_all_training] SIH26054 full training pipeline")
    print(f"[run_all_training] Project root: {ROOT}")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    if not args.skip_build:
        run([py, str(ROOT / "scripts" / "build_dataset.py"),
             "--engines", str(args.engines),
             "--cycles",  str(args.cycles),
             "--window",  str(args.window)],
            "1/4  build_dataset")
    else:
        print("\n[run_all_training] Skipping dataset build (--skip-build)")

    # ── 2. PINN + CNN baselines ───────────────────────────────────────────────
    if not args.skip_pinn:
        run([py, str(ROOT / "src" / "pinn" / "train_pinn.py"),
             "--epochs", str(args.pinn_epochs),
             "--model",  "all"],
            "2/4  train_pinn (PINN + CNN-PCA + CNN-Raw)")
    else:
        print("\n[run_all_training] Skipping PINN training (--skip-pinn)")

    # ── 3. Fault classifier ───────────────────────────────────────────────────
    if not args.skip_clf:
        run([py, str(ROOT / "scripts" / "train_classifier.py"),
             "--epochs", str(args.clf_epochs)],
            "3/4  train_classifier")
    else:
        print("\n[run_all_training] Skipping classifier training (--skip-clf)")

    # ── 4. DRL PPO agent ──────────────────────────────────────────────────────
    if not args.skip_drl:
        run([py, str(ROOT / "src" / "drl" / "drl_agent.py"),
             "--episodes", str(args.drl_episodes)],
            "4/4  drl_agent (PPO)")
    else:
        print("\n[run_all_training] Skipping DRL training (--skip-drl)")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL DONE  —  total time {elapsed/60:.1f} min")
    print(f"  Checkpoints in: {ROOT / 'models' / 'checkpoints'}")
    print(f"  Start server  : uvicorn src.server.server:app --host 127.0.0.1 --port 8000")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

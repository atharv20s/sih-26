"""Creates all PRAHARI tables in the configured Postgres database (Neon).

Usage (from project root):
    python scripts/init_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from db.session import Base, get_engine
from db import models  # noqa: F401  (registers User/Mission/FaultEvent on Base.metadata)


def main():
    engine = get_engine()
    print(f"[init_db] Creating tables on {engine.url.render_as_string(hide_password=True)} ...")
    Base.metadata.create_all(engine)
    print(f"[init_db] Tables created: {list(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    main()

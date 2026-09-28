"""Prints 'yes' if any PRAHARI user account exists, else 'no'. Used by
setup.ps1/setup.sh to decide whether to seed a demo login — kept as a
standalone script rather than inline Python inside the shell scripts, which
is fragile to quote/indent across PowerShell and bash.

Usage: python scripts/check_login_exists.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select

from db.models import User
from db.session import get_sessionmaker


def main():
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        print("yes" if db.scalar(select(User)) else "no")


if __name__ == "__main__":
    main()

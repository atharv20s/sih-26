"""Seeds (or updates) a PRAHARI login account. There is no public signup form
by design — this script is how demo/judge accounts get created.

Usage (from project root):
    python scripts/create_user.py --email demo@prahari.app --password "..." --name "Demo Judge"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select

from auth.security import hash_password
from db.models import User
from db.session import get_sessionmaker


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--name", default="")
    args = p.parse_args()

    email = args.email.strip().lower()
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        existing = db.scalar(select(User).where(User.email == email))
        if existing:
            existing.hashed_password = hash_password(args.password)
            if args.name:
                existing.full_name = args.name
            db.commit()
            print(f"[create_user] Updated password for existing user: {email}")
        else:
            user = User(email=email, hashed_password=hash_password(args.password), full_name=args.name)
            db.add(user)
            db.commit()
            print(f"[create_user] Created user: {email}")


if __name__ == "__main__":
    main()

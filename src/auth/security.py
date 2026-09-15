"""Password hashing and JWT session tokens for PRAHARI.

Auth model: a single seeded account (or a handful, via scripts/create_user.py)
rather than open signup — there is no /api/auth/signup route by design.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
import jwt
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

JWT_SECRET = os.environ.get("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_HOURS = 24 * 7  # a week — this is a demo console, not a bank

COOKIE_NAME = "prahari_session"

# Using the `bcrypt` package directly rather than passlib's CryptContext:
# passlib 1.7.4's bcrypt backend-detection shim breaks against bcrypt>=4.1
# (it calls a `bcrypt.__about__` attribute bcrypt removed, and a legacy
# wrap-bug probe that raises on bcrypt 5.x's stricter 72-byte check).
_BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    pw_bytes = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    pw_bytes = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(pw_bytes, hashed.encode("utf-8"))
    except ValueError:
        return False


def create_session_token(user_id: str, email: str) -> str:
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET is not set — copy .env.example to .env and fill it in.")
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRES_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_session_token(token: str) -> Optional[dict]:
    if not JWT_SECRET:
        return None
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None

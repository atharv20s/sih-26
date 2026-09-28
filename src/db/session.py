"""SQLAlchemy engine/session setup for PRAHARI's Neon Postgres backend.

Sync engine is intentional: auth and dashboard reads are low-frequency,
and the one hot path (the 10 Hz telemetry WebSocket loop) writes
fire-and-forget via a thread pool so it never blocks streaming — see
`persist.py`'s `run_in_background` helper.

Demo-safety fallback: if DATABASE_URL is missing or unreachable at startup
(e.g. venue wifi drops, Neon is briefly down), PRAHARI falls back to a local
SQLite file instead of crashing the whole server — auth/dashboard/mission
persistence all keep working, just against local storage instead of the
shared cloud database. This is clearly logged and surfaced via
`GET /api/status` so nobody mistakes it for the real thing during a demo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")
FALLBACK_SQLITE_PATH = ROOT / "prahari_fallback.db"
FALLBACK_SQLITE_URL = f"sqlite:///{FALLBACK_SQLITE_PATH}"


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None
_backend_info = {"backend": "unknown", "detail": ""}


def backend_status() -> dict:
    """What database is actually in use right now — surfaced on /api/status
    so a fallback is never silently mistaken for the real Postgres backend."""
    return dict(_backend_info)


def get_engine():
    global _engine, _backend_info
    if _engine is not None:
        return _engine

    if DATABASE_URL:
        try:
            candidate = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)
            # Force an actual connection attempt now, at startup, rather than
            # lazily on first request (which would surface as a confusing
            # 500 mid-demo instead of a clean fallback decided up front).
            with candidate.connect():
                pass
            _engine = candidate
            _backend_info = {"backend": "postgres", "detail": "Connected to configured DATABASE_URL"}
            print("[db] Connected to Postgres.")
            return _engine
        except Exception as e:
            print(f"[db] WARNING: could not connect to DATABASE_URL ({e}). "
                  f"Falling back to local SQLite at {FALLBACK_SQLITE_PATH} for this session.")
    else:
        print("[db] WARNING: DATABASE_URL not set. "
              f"Falling back to local SQLite at {FALLBACK_SQLITE_PATH} for this session.")

    _engine = create_engine(FALLBACK_SQLITE_URL, connect_args={"check_same_thread": False})
    _backend_info = {
        "backend": "sqlite-fallback",
        "detail": "DATABASE_URL unreachable or unset — using local SQLite; data will not persist to the shared Neon database.",
    }
    # SQLite needs its tables created on first use too, same models.
    from . import models  # noqa: F401  (registers tables on Base.metadata)
    Base.metadata.create_all(_engine)
    return _engine


def get_sessionmaker():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_session():
    """FastAPI dependency: yields a session, always closed after the request."""
    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

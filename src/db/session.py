"""SQLAlchemy engine/session setup for PRAHARI's Neon Postgres backend.

Sync engine is intentional: auth and dashboard reads are low-frequency,
and the one hot path (the 10 Hz telemetry WebSocket loop) writes
fire-and-forget via a thread pool so it never blocks streaming — see
`persist.py`'s `run_in_background` helper.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill in "
                "your Postgres connection string."
            )
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)
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

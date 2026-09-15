"""PRAHARI database schema — Users, Missions (one row per live WS session),
and FaultEvents (persisted mirror of the client-side EventLog transitions).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    missions: Mapped[list["Mission"]] = relationship(back_populates="user")


class Mission(Base):
    """One row per live telemetry WebSocket connection ("flight session")."""

    __tablename__ = "missions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    aircraft_tail: Mapped[str] = mapped_column(String(64), default="TAPAS-BH-201")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    frame_count: Mapped[int] = mapped_column(default=0)
    final_fault_archetype: Mapped[str] = mapped_column(String(64), default="nominal")
    final_rul_cycles: Mapped[float] = mapped_column(Float, nullable=True)
    had_critical_event: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped["User"] = relationship(back_populates="missions")
    events: Mapped[list["FaultEvent"]] = relationship(back_populates="mission", order_by="FaultEvent.timestamp")


class FaultEvent(Base):
    """One row per fault/severity state transition during a mission —
    server-side persisted version of telemetry_store.js's EventLog."""

    __tablename__ = "fault_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mission_id: Mapped[str] = mapped_column(String(36), ForeignKey("missions.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    channel: Mapped[str] = mapped_column(String(64), default="")
    fault_archetype: Mapped[str] = mapped_column(String(64), default="nominal")
    severity: Mapped[str] = mapped_column(String(32), default="NOMINAL")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[str] = mapped_column(Text, default="")

    mission: Mapped["Mission"] = relationship(back_populates="events")

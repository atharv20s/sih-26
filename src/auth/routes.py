"""Auth API for PRAHARI. Login-only — no signup route (seeded accounts, see
scripts/create_user.py). Session is an httpOnly JWT cookie.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import User
from db.session import get_session
from .security import (
    COOKIE_NAME,
    JWT_EXPIRES_HOURS,
    create_session_token,
    decode_session_token,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


def get_current_user(request: Request, db: Session = Depends(get_session)) -> Optional[User]:
    """Returns the logged-in User, or None. Does not raise — callers decide
    whether the route is optional-auth or required-auth."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    payload = decode_session_token(token)
    if not payload:
        return None
    return db.get(User, payload.get("sub"))


def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    """Dependency for API routes that must be authenticated -> 401 JSON."""
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@router.post("/login")
def login(req: LoginRequest, response: Response, db: Session = Depends(get_session)):
    user = db.scalar(select(User).where(User.email == req.email.strip().lower()))
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_session_token(user.id, user.email)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,  # dev is plain http://localhost; flip to True behind real TLS
        samesite="lax",
        max_age=JWT_EXPIRES_HOURS * 3600,
        path="/",
    )
    return {"success": True, "email": user.email, "full_name": user.full_name}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"success": True}


@router.get("/me")
def me(user: Optional[User] = Depends(get_current_user)):
    if user is None:
        return {"authenticated": False}
    return {"authenticated": True, "email": user.email, "full_name": user.full_name}

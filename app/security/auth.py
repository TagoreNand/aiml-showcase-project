"""FastAPI auth + RBAC dependencies.

* When ``AUTH_ENABLED`` is False (default for the demo/tests), every request is
  treated as a built-in ``system`` admin so all endpoints stay open.
* When enabled, clients obtain a JWT from ``/auth/token`` (OAuth2 password
  flow) and pass it as ``Authorization: Bearer <token>``. ``require_role``
  enforces the role hierarchy viewer < analyst < admin.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.config import Settings
from app.security import jwt_utils
from app.security.users import Principal, get_user_store

# auto_error=False so we can allow anonymous access when auth is disabled.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)

_SYSTEM_PRINCIPAL = Principal(username="system", role="admin", tenant_id=Settings.DEFAULT_TENANT)


def create_access_token(principal: Principal) -> str:
    return jwt_utils.encode(
        {"sub": principal.username, "role": principal.role, "tenant": principal.tenant_id},
        Settings.JWT_SECRET,
        algorithm=Settings.JWT_ALGORITHM,
        expires_minutes=Settings.JWT_EXPIRE_MINUTES,
    )


def authenticate(username: str, password: str) -> Optional[Principal]:
    return get_user_store().authenticate(username, password)


def get_current_principal(token: Optional[str] = Depends(oauth2_scheme)) -> Principal:
    if not Settings.AUTH_ENABLED:
        return _SYSTEM_PRINCIPAL
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt_utils.decode(token, Settings.JWT_SECRET, algorithm=Settings.JWT_ALGORITHM)
    except jwt_utils.JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal(
        username=payload.get("sub", "unknown"),
        role=payload.get("role", "viewer"),
        tenant_id=payload.get("tenant", Settings.DEFAULT_TENANT),
    )


def require_role(required_role: str):
    """Dependency factory enforcing a minimum role."""

    def _dep(principal: Principal = Depends(get_current_principal)) -> Principal:
        if not principal.can(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires '{required_role}' role; '{principal.role}' is insufficient",
            )
        return principal

    return _dep

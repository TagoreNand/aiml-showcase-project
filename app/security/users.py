"""User directory + RBAC role model.

For the local-first demo, users are loaded from the ``DEMO_USERS`` setting.
Swap ``UserStore`` for a DB/identity-provider implementation in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from app.config import Settings
from app.security.passwords import hash_password, verify_password

# Role hierarchy: higher number = more privilege.
ROLE_LEVELS = {"viewer": 10, "analyst": 20, "admin": 30}


@dataclass
class Principal:
    username: str
    role: str
    tenant_id: str = "public"

    @property
    def level(self) -> int:
        return ROLE_LEVELS.get(self.role, 0)

    def can(self, required_role: str) -> bool:
        return self.level >= ROLE_LEVELS.get(required_role, 99)


@dataclass
class _UserRecord:
    username: str
    password_hash: str
    role: str
    tenant_id: str


class UserStore:
    def __init__(self, spec: Optional[str] = None):
        self._users: Dict[str, _UserRecord] = {}
        self._load(spec if spec is not None else Settings.DEMO_USERS)

    def _load(self, spec: str) -> None:
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) < 3:
                continue
            username, password, role = parts[0], parts[1], parts[2]
            tenant = parts[3] if len(parts) > 3 else Settings.DEFAULT_TENANT
            self._users[username] = _UserRecord(
                username=username,
                password_hash=hash_password(password),
                role=role,
                tenant_id=tenant,
            )

    def authenticate(self, username: str, password: str) -> Optional[Principal]:
        rec = self._users.get(username)
        if rec and verify_password(password, rec.password_hash):
            return Principal(rec.username, rec.role, rec.tenant_id)
        return None

    def __len__(self) -> int:
        return len(self._users)


_STORE: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _STORE
    if _STORE is None:
        _STORE = UserStore()
    return _STORE

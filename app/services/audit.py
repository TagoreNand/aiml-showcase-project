"""Append-only audit logging (JSONL local-first).

Records who did what, to which resource, for which tenant, and the outcome.
Swap the sink for a database/SIEM in production without changing call sites.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._lock = threading.Lock()

    def record(
        self,
        action: str,
        actor: str = "system",
        tenant_id: str = "public",
        resource: Optional[str] = None,
        status: str = "ok",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "tenant_id": tenant_id,
            "action": action,
            "resource": resource,
            "status": status,
            "details": details or {},
        }
        line = json.dumps(entry)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def tail(self, limit: int = 50, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if tenant_id is None or rec.get("tenant_id") == tenant_id:
                    rows.append(rec)
        return rows[-limit:]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

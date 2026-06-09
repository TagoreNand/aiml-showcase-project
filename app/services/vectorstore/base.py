"""Abstract vector-store interface shared by all retrieval backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StoredDocument:
    document_id: str
    text: str
    tenant_id: str = "public"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
        }


class VectorStore(ABC):
    """Tenant-aware document store with similarity search.

    Search results are dicts with keys ``document_id``, ``score`` and
    ``excerpt`` so they map directly onto the existing ``SearchResult`` schema.
    """

    backend: str = "base"

    @abstractmethod
    def add(
        self,
        document_id: str,
        text: str,
        tenant_id: str = "public",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def search(
        self, query: str, top_k: int = 3, tenant_id: str = "public"
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def count(self, tenant_id: Optional[str] = None) -> int:
        ...

    @abstractmethod
    def delete(self, document_id: str, tenant_id: str = "public") -> bool:
        ...

    def tenants(self) -> List[str]:  # optional override
        return []

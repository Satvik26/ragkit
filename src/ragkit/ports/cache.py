"""ACL-scoped query cache port. Cache keys must always include caller ACL tags."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from ragkit.domain.models import Answer


def cache_key(
    query: str,
    allowed_tags: frozenset[str],
    epoch: str = "0",
    model_name: str = "",
) -> str:
    """Key of epoch, sorted ACL tags, model name, and normalized query.

    v2 drops entries written before the model name was part of the key.
    """
    normalized = " ".join(query.lower().split())
    tags_part = ",".join(sorted(allowed_tags))
    model = " ".join(model_name.lower().split())
    raw = f"v2|{epoch}|{tags_part}|{model}|{normalized}".encode()
    return hashlib.sha256(raw).hexdigest()


class QueryCache(ABC):
    @abstractmethod
    def get(self, query: str, allowed_tags: frozenset[str]) -> Answer | None:
        """Return cached Answer for this exact (query, allowed_tags) pair, or None."""

    @abstractmethod
    def set(self, query: str, allowed_tags: frozenset[str], answer: Answer) -> None:
        """Store Answer scoped to (query, allowed_tags)."""

    @abstractmethod
    def invalidate_all(self) -> None:
        """Invalidate cached answers when documents are ingested or updated."""

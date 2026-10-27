"""Small, separate ports (interface segregation): read-only services never see write methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragkit.domain.models import Chunk, ScoredChunk


class VectorWriter(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or overwrite by chunk id. Must be idempotent."""


class VectorSearcher(ABC):
    @abstractmethod
    def search(
        self, vector: Sequence[float], k: int, allowed_tags: frozenset[str]
    ) -> list[ScoredChunk]:
        """Top-k by similarity, best first. The ACL filter is applied inside the search,
        never after it. A chunk is visible iff chunk.acl_tags intersects allowed_tags."""

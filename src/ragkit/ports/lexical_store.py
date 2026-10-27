"""Lexical search ports (interface segregation): writers index chunks, searchers query them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragkit.domain.models import Chunk, ScoredChunk


class LexicalWriter(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Insert or overwrite by chunk id. Must be idempotent."""


class LexicalSearcher(ABC):
    @abstractmethod
    def search(self, query: str, k: int, allowed_tags: frozenset[str]) -> list[ScoredChunk]:
        """Top-k by lexical score (BM25), best first. The ACL filter is applied inside the
        search, never after it. A chunk is visible iff chunk.acl_tags intersects allowed_tags."""

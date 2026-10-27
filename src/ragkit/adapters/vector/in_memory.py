from __future__ import annotations

import math
from collections.abc import Sequence

from ragkit.domain.models import Chunk, ScoredChunk
from ragkit.ports.vector_store import VectorSearcher, VectorWriter


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class InMemoryVectorStore(VectorWriter, VectorSearcher):
    """Reference implementation: used in unit tests and the dev profile."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._items[chunk.id] = (chunk, list(vector))

    def search(
        self, vector: Sequence[float], k: int, allowed_tags: frozenset[str]
    ) -> list[ScoredChunk]:
        if k <= 0 or not allowed_tags:
            return []
        scored = [
            ScoredChunk(chunk, _cosine(vector, vec))
            for chunk, vec in self._items.values()
            if chunk.acl_tags & allowed_tags
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

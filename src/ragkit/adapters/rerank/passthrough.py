from __future__ import annotations

from collections.abc import Sequence

from ragkit.domain.models import ScoredChunk
from ragkit.ports.reranker import Reranker


class PassthroughReranker(Reranker):
    """Keeps the retrieval order. Swap for a cross-encoder adapter once evals justify it."""

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return sorted(candidates, key=lambda s: s.score, reverse=True)

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragkit.domain.models import ScoredChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return candidates re-scored and sorted best first."""

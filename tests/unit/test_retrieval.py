from __future__ import annotations

from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.domain.models import Chunk
from ragkit.services.retrieval import RetrievalService
from tests.fakes import KeywordEmbedder


def test_top_k_and_order() -> None:
    store = InMemoryVectorStore()
    tags = frozenset({"eng"})
    store.upsert(
        [Chunk(str(i), "d", "t", tags) for i in range(3)],
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
    )
    service = RetrievalService(
        KeywordEmbedder({"q": [1.0, 0.0]}), store, PassthroughReranker(), candidates=3, top_k=2
    )
    assert [s.chunk.id for s in service.retrieve("q", tags)] == ["0", "1"]

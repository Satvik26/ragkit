from __future__ import annotations

from typing import Any

import psycopg
import pytest

from ragkit.adapters.lexical.bm25 import InMemoryBM25Store, PostgresBM25Store
from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.domain.models import Chunk, ScoredChunk
from ragkit.services.retrieval import RetrievalService, fuse_rrf
from tests.fakes import KeywordEmbedder


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple[Any, ...]], fail_undefined: bool = False) -> None:
        self._rows = rows
        self._fail_undefined = fail_undefined
        self.closed = False
        self.rolled_back = False

    def execute(self, _sql: str, _params: tuple[Any, ...]) -> _FakeCursor:
        if self._fail_undefined:
            raise psycopg.errors.UndefinedTable("chunks does not exist")
        return _FakeCursor(self._rows)

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def test_hybrid_retrieval_promotes_exact_keyword_hit() -> None:
    tags = frozenset({"eng"})
    c_dense = Chunk("dense-only", "d1", "general cellular mechanism overview", tags)
    c_exact = Chunk("exact-sku", "d2", "biomarker foxp3 inhibits il6 pathway", tags)

    vector_store = InMemoryVectorStore()
    vector_store.upsert([c_dense, c_exact], [[1.0, 0.0], [0.6, 0.4]])

    lexical_store = InMemoryBM25Store()
    lexical_store.upsert([c_dense, c_exact])

    service = RetrievalService(
        KeywordEmbedder({"foxp3 il6": [1.0, 0.0]}),
        vector_store,
        PassthroughReranker(),
        candidates=10,
        top_k=2,
        lexical_searcher=lexical_store,
    )
    hits = service.retrieve("foxp3 il6", tags)
    assert {h.chunk.id for h in hits} == {"dense-only", "exact-sku"}
    assert hits[0].score >= 0.35


def test_fuse_rrf_handles_lexical_only_and_empty_inputs() -> None:
    tags = frozenset({"eng"})
    c1 = Chunk("c1", "d1", "alpha beta", tags)
    lexical_only = fuse_rrf([], [ScoredChunk(c1, 3.0)])
    assert len(lexical_only) == 1
    assert lexical_only[0].score > 0.5
    assert fuse_rrf([], []) == []


def test_postgres_bm25_store_scores_visible_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        ("c1", "doc1", "foxp3 regulates immune tolerance", ["eng"], 0, 4, "h1"),
        ("c2", "doc2", "unrelated travel reimbursement rule", ["eng"], 0, 4, "h2"),
    ]
    conn = _FakeConn(rows)
    store = PostgresBM25Store("postgresql://rag:rag@localhost:5432/rag")
    monkeypatch.setattr(store, "_connect", lambda: conn)
    store.upsert([])
    hits = store.search("foxp3", 5, frozenset({"eng"}))
    assert [h.chunk.id for h in hits] == ["c1"]
    assert conn.closed is True
    assert store.search("foxp3", 0, frozenset({"eng"})) == []


def test_postgres_bm25_returns_empty_when_table_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn([], fail_undefined=True)
    store = PostgresBM25Store("postgresql://rag:rag@localhost:5432/rag")
    monkeypatch.setattr(store, "_connect", lambda: conn)
    assert store.search("foxp3", 5, frozenset({"eng"})) == []
    assert conn.rolled_back is True
    assert conn.closed is True

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import URLError

import pytest

from ragkit.adapters.rerank.http import HttpCrossEncoderReranker
from ragkit.domain.models import Chunk, ScoredChunk


def _response(payload: dict[str, Any]) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode())


def test_rerank_reorders_head_window_and_preserves_tail() -> None:
    seen: dict[str, Any] = {}

    def transport(request: Any, timeout: float) -> io.BytesIO:
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode())
        seen["timeout"] = timeout
        return _response(
            {
                "results": [
                    {"index": 1, "score": 0.99},
                    {"index": 0, "score": 0.01},
                ]
            }
        )

    reranker = HttpCrossEncoderReranker(
        "http://embedder:80/",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        window=2,
        alpha=0.5,
        transport=transport,
    )
    tags = frozenset({"eng"})
    c1 = ScoredChunk(Chunk("c1", "d1", "first doc", tags), 0.70)
    c2 = ScoredChunk(Chunk("c2", "d2", "second doc", tags), 0.60)
    c3 = ScoredChunk(Chunk("c3", "d3", "tail doc", tags), 0.10)

    ranked = reranker.rerank("query text", [c1, c2, c3])
    assert [item.chunk.id for item in ranked] == ["c2", "c1", "c3"]
    assert seen["url"] == "http://embedder:80/v1/rerank"
    assert seen["body"]["documents"] == ["first doc", "second doc"]


def test_rerank_empty_candidates_skips_network() -> None:
    def transport(_request: Any, timeout: float) -> io.BytesIO:
        raise AssertionError(f"unexpected call with timeout {timeout}")

    reranker = HttpCrossEncoderReranker("http://embedder:80", transport=transport)
    assert reranker.rerank("q", []) == []


def test_rerank_rejects_wrong_result_count() -> None:
    reranker = HttpCrossEncoderReranker(
        "http://embedder:80",
        transport=lambda _req, timeout: _response({"results": []}),
    )
    c1 = ScoredChunk(Chunk("c1", "d1", "text", frozenset({"eng"})), 0.8)
    with pytest.raises(ValueError, match="wrong number of scores"):
        reranker.rerank("q", [c1])


def test_reranker_ready_probe() -> None:
    ok_reranker = HttpCrossEncoderReranker(
        "http://embedder:80",
        transport=lambda _req, timeout: _response({"status": "ok"}),
    )
    assert ok_reranker.ready() is True

    def failing(_req: Any, timeout: float) -> io.BytesIO:
        raise URLError(f"down {timeout}")

    bad_reranker = HttpCrossEncoderReranker("http://embedder:80", transport=failing)
    assert bad_reranker.ready() is False

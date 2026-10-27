"""Cross-encoder HTTP reranker. PassthroughReranker stays for unit tests."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from ragkit.domain.models import ScoredChunk
from ragkit.ports.reranker import Reranker


class HttpCrossEncoderReranker(Reranker):
    """Calls POST /v1/rerank on the model sidecar to re-score (query, chunk) pairs."""

    def __init__(
        self,
        base_url: str,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        window: int = 16,
        alpha: float = 0.65,
        timeout: float = 30.0,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._window = window
        self._alpha = alpha
        self._timeout = timeout
        self._transport = transport or urllib.request.urlopen

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
        head = ordered[: self._window]
        tail = ordered[self._window :]
        payload = self._request(
            "POST",
            "/v1/rerank",
            {
                "model": self._model,
                "query": query,
                "documents": [item.chunk.text for item in head],
            },
            self._timeout,
        )
        rows = sorted(payload["results"], key=lambda row: int(row.get("index", 0)))
        if len(rows) != len(head):
            raise ValueError("reranker service returned the wrong number of scores")
        reranked_head = [
            ScoredChunk(
                item.chunk,
                self._alpha * item.score + (1.0 - self._alpha) * float(row["score"]),
            )
            for item, row in zip(head, rows, strict=True)
        ]
        reranked_head.sort(key=lambda item: item.score, reverse=True)
        return reranked_head + tail

    def ready(self) -> bool:
        """A readiness probe. Any failure means the process should not take traffic."""
        try:
            self._request("GET", "/health", None, 2.0)
        except Exception:
            return False
        return True

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None, timeout: float
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self._base + path,
            data=data,
            headers={"content-type": "application/json"},
            method=method,
        )
        with self._transport(request, timeout=timeout) as response:
            return json.loads(response.read().decode())

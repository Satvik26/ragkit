"""OpenAI-compatible embedding client. The hashing stub stays for tests."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from ragkit.ports.embedder import Embedder


class HttpEmbedder(Embedder):
    """Calls POST /v1/embeddings. BGE-M3 is the local model behind that URL."""

    def __init__(
        self,
        base_url: str,
        dim: int,
        model: str = "BAAI/bge-m3",
        timeout: float = 120,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._dim = dim
        self._model = model
        self._timeout = timeout
        self._transport = transport or urllib.request.urlopen

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = self._request(
            "POST",
            "/v1/embeddings",
            {"model": self._model, "input": list(texts)},
            self._timeout,
        )
        rows = sorted(payload["data"], key=lambda row: int(row.get("index", 0)))
        if len(rows) != len(texts):
            raise ValueError("embedding service returned the wrong number of vectors")
        vectors: list[list[float]] = []
        for row in rows:
            vector = [float(value) for value in row["embedding"]]
            if len(vector) != self._dim:
                raise ValueError(f"expected embedding dim {self._dim}, got {len(vector)}")
            vectors.append(vector)
        return vectors

    def ready(self) -> bool:
        """A readiness probe. Any failure means the process should not take traffic."""
        try:
            self._request("GET", "/health", None, 2)
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

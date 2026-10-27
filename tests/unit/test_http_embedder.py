from __future__ import annotations

import json

import pytest

from ragkit.adapters.embed.http import HttpEmbedder


class _Response:
    def __init__(self, body: dict[str, object]) -> None:
        self._body = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_embed_sorts_by_index_and_checks_dim() -> None:
    seen: list[str] = []

    def transport(request: object, timeout: float) -> _Response:
        body = json.loads(request.data.decode())  # type: ignore[attr-defined]
        seen.append(body["model"])
        assert timeout == 120
        return _Response(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )

    embedder = HttpEmbedder("http://embedder", 2, transport=transport)
    assert embedder.embed(["refunds", "days"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert seen == ["BAAI/bge-m3"]


def test_empty_input_does_not_call_the_service() -> None:
    def transport(request: object, timeout: float) -> _Response:
        del request, timeout
        raise AssertionError("empty input must not call the service")

    assert HttpEmbedder("http://embedder", 2, transport=transport).embed([]) == []


def test_wrong_vector_count_is_rejected() -> None:
    def transport(request: object, timeout: float) -> _Response:
        del request, timeout
        return _Response({"data": []})

    embedder = HttpEmbedder("http://embedder", 2, transport=transport)
    with pytest.raises(ValueError, match="wrong number"):
        embedder.embed(["refunds"])


def test_wrong_dim_is_rejected() -> None:
    def transport(request: object, timeout: float) -> _Response:
        del request, timeout
        return _Response({"data": [{"index": 0, "embedding": [1.0]}]})

    embedder = HttpEmbedder("http://embedder", 2, transport=transport)
    with pytest.raises(ValueError, match="dim 2"):
        embedder.embed(["refunds"])


def test_ready_is_false_when_the_probe_fails() -> None:
    def transport(request: object, timeout: float) -> _Response:
        raise ConnectionError("down")

    assert HttpEmbedder("http://embedder", 2, transport=transport).ready() is False


def test_ready_uses_the_health_route() -> None:
    def transport(request: object, timeout: float) -> _Response:
        assert request.get_method() == "GET"  # type: ignore[attr-defined]
        assert request.full_url.endswith("/health")  # type: ignore[attr-defined]
        assert timeout == 2
        return _Response({"status": "ok"})

    assert HttpEmbedder("http://embedder", 2, transport=transport).ready() is True

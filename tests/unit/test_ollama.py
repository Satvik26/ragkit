from __future__ import annotations

import json

import pytest

from ragkit.adapters.llm.ollama import OllamaClient


class _Response:
    def __init__(self, body: dict[str, object]) -> None:
        self._body = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_generate_returns_the_response_text() -> None:
    def transport(request: object, timeout: float) -> _Response:
        body = json.loads(request.data.decode())  # type: ignore[attr-defined]
        assert body["model"] == "gemma4:e2b"
        assert body["stream"] is False
        assert body["think"] is False
        assert body["options"] == {"temperature": 0}
        assert body["prompt"] == "question"
        assert timeout == 120
        return _Response({"response": "Refunds take fourteen days [1]"})

    text = OllamaClient("http://ollama", transport=transport).generate("question")
    assert text == "Refunds take fourteen days [1]"


def test_empty_response_is_rejected() -> None:
    def transport(request: object, timeout: float) -> _Response:
        del request, timeout
        return _Response({"response": "  "})

    client = OllamaClient("http://ollama", transport=transport)
    with pytest.raises(ValueError, match="empty"):
        client.generate("question")


def test_ready_when_the_model_tag_is_present() -> None:
    def transport(request: object, timeout: float) -> _Response:
        assert request.full_url.endswith("/api/tags")  # type: ignore[attr-defined]
        assert timeout == 2
        return _Response({"models": [{"name": "gemma4:e2b"}]})

    assert OllamaClient("http://ollama", transport=transport).ready() is True


def test_ready_accepts_an_untagged_name_with_a_latest_tag() -> None:
    def transport(request: object, timeout: float) -> _Response:
        del request, timeout
        return _Response({"models": [{"name": "llama3.2:latest"}]})

    client = OllamaClient("http://ollama", model="llama3.2", transport=transport)
    assert client.ready() is True


def test_not_ready_when_the_model_is_missing() -> None:
    def transport(request: object, timeout: float) -> _Response:
        del request, timeout
        return _Response({"models": [{"name": "other:latest"}]})

    assert OllamaClient("http://ollama", transport=transport).ready() is False


def test_not_ready_when_the_call_fails() -> None:
    def transport(request: object, timeout: float) -> _Response:
        del request, timeout
        raise ConnectionError("down")

    assert OllamaClient("http://ollama", transport=transport).ready() is False

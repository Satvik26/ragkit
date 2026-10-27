from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from ragkit.adapters.dev.stubs import ExtractiveLLM
from ragkit.adapters.llm.factory import build_llm
from ragkit.adapters.llm.ollama import OllamaClient
from ragkit.adapters.llm.openai_chat import OpenAIChatClient
from ragkit.config.settings import Settings


class _Response:
    def __init__(self, body: bytes | dict[str, object], lines: list[bytes] | None = None) -> None:
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self._lines = lines or []

    def read(self) -> bytes:
        return self._body

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._lines)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_factory_selects_ollama_openai_or_the_local_stub() -> None:
    ollama = build_llm(Settings(llm_provider="ollama", ollama_url="http://127.0.0.1:1"))
    assert isinstance(ollama.client, OllamaClient)
    assert ollama.model_name == "gemma4:e2b"
    assert ollama.ready is not None

    stub = build_llm(Settings(_env_file=None))
    assert isinstance(stub.client, ExtractiveLLM)
    assert stub.ready is None

    with pytest.raises(ValueError, match="RAG_OPENAI_API_KEY"):
        build_llm(Settings(_env_file=None, llm_provider="openai"))

    remote = build_llm(
        Settings(llm_provider="openai", openai_api_key="sk-test", openai_model="gpt-4.1")
    )
    assert isinstance(remote.client, OpenAIChatClient)
    assert remote.model_name == "gpt-4.1"
    assert remote.ready is not None


def test_openai_generate_sends_bearer_token_and_reads_the_message() -> None:
    seen: dict[str, object] = {}

    def transport(request: object, timeout: float) -> _Response:
        seen["url"] = request.full_url  # type: ignore[attr-defined]
        seen["auth"] = request.headers["Authorization"]  # type: ignore[attr-defined]
        seen["body"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        seen["timeout"] = timeout
        return _Response(
            {"choices": [{"message": {"role": "assistant", "content": "No. 2.7 µmol/L [1]."}}]}
        )

    text = OpenAIChatClient("sk-test", transport=transport).generate("question")
    assert text == "No. 2.7 µmol/L [1]."
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["timeout"] == 120
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["temperature"] == 0
    assert body["messages"] == [{"role": "user", "content": "question"}]


def test_openai_stream_yields_deltas_until_done() -> None:
    lines = [
        b'data: {"choices":[{"delta":{"content":"No. "}}]}\n',
        b"\n",
        b'data: {"choices":[{"delta":{"content":"2.7"}}]}\n',
        b"data: [DONE]\n",
    ]

    def transport(request: object, timeout: float) -> _Response:
        del request, timeout
        return _Response(b"", lines)

    chunks = list(OpenAIChatClient("sk-test", transport=transport).generate_stream("q"))
    assert chunks == ["No. ", "2.7"]


def test_openai_rejects_an_empty_or_error_payload() -> None:
    def empty(request: object, timeout: float) -> _Response:
        del request, timeout
        return _Response({"choices": [{"message": {"content": "  "}}]})

    with pytest.raises(ValueError, match="empty"):
        OpenAIChatClient("sk-test", transport=empty).generate("q")

    def failed(request: object, timeout: float) -> _Response:
        del request, timeout
        return _Response({"error": {"message": "invalid api key"}})

    with pytest.raises(ValueError, match="invalid api key"):
        OpenAIChatClient("sk-test", transport=failed).generate("q")


def test_openai_ready_checks_the_models_endpoint() -> None:
    def ok(request: object, timeout: float) -> _Response:
        assert request.full_url.endswith("/models")  # type: ignore[attr-defined]
        assert timeout == 2
        return _Response({"data": [{"id": "gpt-4.1"}]})

    assert OpenAIChatClient("sk-test", transport=ok).ready() is True

    def down(request: object, timeout: float) -> _Response:
        del request, timeout
        raise ConnectionError("down")

    assert OpenAIChatClient("sk-test", transport=down).ready() is False
    assert OpenAIChatClient("  ", transport=ok).ready() is False

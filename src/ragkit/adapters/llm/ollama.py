"""Ollama chat client. Tests stay on ScriptedLLM and ExtractiveLLM."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from ragkit.ports.llm import LLMClient


class OllamaClient(LLMClient):
    """POST /api/generate. Temperature is 0 so citations stay stable."""

    def __init__(
        self,
        base_url: str,
        model: str = "gemma4:e2b",
        timeout: float = 120,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._transport = transport or urllib.request.urlopen

    def generate(self, prompt: str) -> str:
        payload = self._request(
            "POST",
            "/api/generate",
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
            },
            self._timeout,
        )
        text = payload.get("response")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Ollama returned an empty response")
        return text

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Stream NDJSON tokens from Ollama POST /api/generate with stream=True."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            self._base + "/api/generate",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self._transport(request, timeout=self._timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("response")
                if isinstance(piece, str) and piece:
                    yield piece
                if chunk.get("done"):
                    break

    def ready(self) -> bool:
        """Ready only when Ollama is up and the configured model is pulled."""
        try:
            payload = self._request("GET", "/api/tags", None, 2)
        except Exception:
            return False
        names = [str(row.get("name", "")) for row in payload.get("models", [])]
        return _model_present(names, self._model)

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


def _model_present(names: list[str], wanted: str) -> bool:
    for name in names:
        if name == wanted:
            return True
        if ":" not in wanted and name.startswith(wanted + ":"):
            return True
    return False

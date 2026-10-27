"""OpenAI Chat Completions client. The LLMClient port is the strategy; this is one adapter."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from ragkit.ports.llm import LLMClient


class OpenAIChatClient(LLMClient):
    """POST /chat/completions. Temperature is 0 so citations stay stable."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._key = api_key.strip()
        self._model = model
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport or urllib.request.urlopen

    def generate(self, prompt: str) -> str:
        payload = self._post(
            {"model": self._model, "temperature": 0, "messages": _messages(prompt)}
        )
        text = _message_text(payload)
        if not text.strip():
            raise ValueError("OpenAI returned an empty response")
        return text

    def generate_stream(self, prompt: str) -> Iterator[str]:
        body = {
            "model": self._model,
            "temperature": 0,
            "stream": True,
            "messages": _messages(prompt),
        }
        request = urllib.request.Request(
            self._base + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=self._headers(),
            method="POST",
        )
        with self._transport(request, timeout=self._timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
                piece = _stream_piece(line.strip())
                if piece is None:
                    continue
                if piece == "":
                    break
                yield piece

    def ready(self) -> bool:
        """Ready when the API key is accepted. Does not spend a completion."""
        if not self._key:
            return False
        request = urllib.request.Request(
            self._base + "/models",
            headers=self._headers(),
            method="GET",
        )
        try:
            with self._transport(request, timeout=2) as response:
                payload = json.loads(response.read().decode())
        except Exception:
            return False
        return isinstance(payload, dict) and isinstance(payload.get("data"), list)

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self._key}",
        }

    def _post(self, body: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self._base + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=self._headers(),
            method="POST",
        )
        with self._transport(request, timeout=self._timeout) as response:
            payload = json.loads(response.read().decode())
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            message = str(payload["error"].get("message", "OpenAI request failed"))
            raise ValueError(message)
        return payload


def _messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _message_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("OpenAI returned an empty response")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI returned an empty response")
    message = choices[0].get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("OpenAI returned an empty response")
    return content


def _stream_piece(line: str) -> str | None:
    """Return a text delta, '' to stop, or None to skip a keepalive line."""
    if not line or line.startswith(":"):
        return None
    data = line[5:].strip() if line.startswith("data:") else line
    if data == "[DONE]":
        return ""
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = chunk.get("choices") if isinstance(chunk, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    delta = choices[0].get("delta", {})
    content = delta.get("content") if isinstance(delta, dict) else None
    if not isinstance(content, str) or not content:
        return None
    return content

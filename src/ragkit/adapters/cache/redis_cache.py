"""In-memory and Redis RESP query caches with ACL-scoped keys and epoch invalidation."""

from __future__ import annotations

import json
import socket
from urllib.parse import urlparse

from ragkit.domain.models import Answer, Citation
from ragkit.ports.cache import QueryCache, cache_key


def _serialize_answer(answer: Answer) -> str:
    return json.dumps(
        {
            "text": answer.text,
            "citations": [{"chunk_id": c.chunk_id, "doc_id": c.doc_id} for c in answer.citations],
            "abstained": answer.abstained,
            "reason": answer.reason,
        }
    )


def _deserialize_answer(payload: str) -> Answer:
    raw = json.loads(payload)
    return Answer(
        text=str(raw["text"]),
        citations=tuple(
            Citation(chunk_id=str(c["chunk_id"]), doc_id=str(c["doc_id"]))
            for c in raw.get("citations", [])
        ),
        abstained=bool(raw["abstained"]),
        reason=str(raw["reason"]) if raw.get("reason") is not None else None,
    )


class InMemoryQueryCache(QueryCache):
    """Process-local ACL-scoped query cache used in unit tests and dev mode."""

    def __init__(self, model_name: str = "") -> None:
        self._epoch = 0
        self._model_name = model_name
        self._entries: dict[str, Answer] = {}

    def get(self, query: str, allowed_tags: frozenset[str]) -> Answer | None:
        if not allowed_tags:
            return None
        key = cache_key(query, allowed_tags, str(self._epoch), self._model_name)
        return self._entries.get(key)

    def set(self, query: str, allowed_tags: frozenset[str], answer: Answer) -> None:
        if not allowed_tags:
            return
        key = cache_key(query, allowed_tags, str(self._epoch), self._model_name)
        self._entries[key] = answer

    def invalidate_all(self) -> None:
        self._epoch += 1
        self._entries.clear()


class _RespClient:
    """Minimal Redis RESP2 client over stdlib sockets (no external dependency needed)."""

    def __init__(self, redis_url: str, timeout: float = 2.0) -> None:
        parsed = urlparse(redis_url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 6379
        self._timeout = timeout

    def command(self, *parts: str) -> str | int | None:
        encoded = [f"*{len(parts)}\r\n".encode()]
        for part in parts:
            raw = part.encode("utf-8")
            encoded.append(f"${len(raw)}\r\n".encode() + raw + b"\r\n")
        payload = b"".join(encoded)

        with socket.create_connection((self._host, self._port), timeout=self._timeout) as sock:
            sock.sendall(payload)
            reader = sock.makefile("rb")
            line = reader.readline()
            if not line:
                raise ConnectionError("empty response from Redis")
            prefix = line[:1]
            body = line[1:].rstrip(b"\r\n")
            if prefix == b"+":
                return body.decode("utf-8")
            if prefix == b":":
                return int(body)
            if prefix == b"$":
                length = int(body)
                if length < 0:
                    return None
                data = reader.read(length)
                reader.read(2)  # trailing CRLF
                return data.decode("utf-8")
            if prefix == b"-":
                raise RuntimeError(body.decode("utf-8", errors="replace"))
            return None


class RedisQueryCache(QueryCache):
    """Redis-backed ACL-scoped query cache with O(1) epoch invalidation."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 3600,
        client: _RespClient | None = None,
        model_name: str = "",
    ) -> None:
        self._client = client or _RespClient(redis_url)
        self._ttl = ttl_seconds
        self._model_name = model_name
        self._epoch_key = "ragkit:cache:epoch"

    def ready(self) -> bool:
        try:
            return self._client.command("PING") == "PONG"
        except Exception:
            return False

    def _current_epoch(self) -> str:
        try:
            val = self._client.command("GET", self._epoch_key)
            return str(val) if val is not None else "0"
        except Exception:
            return "0"

    def get(self, query: str, allowed_tags: frozenset[str]) -> Answer | None:
        if not allowed_tags:
            return None
        key = "ragkit:q:" + cache_key(
            query, allowed_tags, self._current_epoch(), self._model_name
        )
        try:
            raw = self._client.command("GET", key)
            if not isinstance(raw, str):
                return None
            return _deserialize_answer(raw)
        except Exception:
            return None

    def set(self, query: str, allowed_tags: frozenset[str], answer: Answer) -> None:
        if not allowed_tags:
            return
        key = "ragkit:q:" + cache_key(
            query, allowed_tags, self._current_epoch(), self._model_name
        )
        try:
            self._client.command("SETEX", key, str(self._ttl), _serialize_answer(answer))
        except Exception:
            return

    def invalidate_all(self) -> None:
        try:
            self._client.command("INCR", self._epoch_key)
        except Exception:
            return

"""In-memory and Redis-backed ACL-scoped conversation stores + coreference contextualizer."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque

from ragkit.adapters.cache.redis_cache import _RespClient
from ragkit.adapters.lexical.bm25 import tokenize
from ragkit.ports.conversation import ChatTurn, ConversationStore

_PRONOUN_FOLLOWUP = re.compile(
    r"\b(?:it|its|they|them|their|that|those|this|these|same)\b|"
    r"^\s*(?:and\b|what\s+about\b|how\s+about\b|also\b)",
    re.IGNORECASE,
)
_REQUEST_WORDS = frozenset(
    {
        "about",
        "can",
        "could",
        "deeper",
        "detail",
        "details",
        "elaborate",
        "explain",
        "further",
        "me",
        "more",
        "please",
        "tell",
        "yo",
        "you",
    }
)


def _session_key(session_id: str, allowed_tags: frozenset[str]) -> str:
    tags_part = ",".join(sorted(allowed_tags))
    digest = hashlib.sha256(f"{tags_part}|{session_id.strip()}".encode()).hexdigest()[:32]
    return f"ragkit:conv:{digest}"


def _topic_tokens(query: str) -> list[str]:
    return [token for token in tokenize(query) if token not in _REQUEST_WORDS]


def contextualize_follow_up(query: str, history: list[ChatTurn]) -> str:
    """Resolve a follow-up against the previous user question.

    A pronoun, or an opener such as "what about", keeps the existing anchor rewrite.
    A turn with no topic words left, such as "tell me in more detail", searches the
    previous question plus "in more detail". A named topic stays a new search.
    """
    cleaned = query.strip()
    if not cleaned or not history:
        return query
    if _PRONOUN_FOLLOWUP.search(cleaned):
        last_turn = history[-1]
        prior_tokens = tokenize(last_turn.question)
        current_tokens = set(tokenize(cleaned))
        anchor_words = [token for token in prior_tokens if token not in current_tokens][:6]
        if not anchor_words:
            return cleaned
        return f"{' '.join(anchor_words)}: {cleaned}"
    if _topic_tokens(cleaned):
        return cleaned
    previous = history[-1].question.strip()
    if not previous:
        return cleaned
    return f"{previous} in more detail"


class InMemoryConversationStore(ConversationStore):
    """Process-local ACL-scoped conversation memory."""

    def __init__(self, max_turns: int = 10) -> None:
        self._max_turns = max_turns
        self._sessions: dict[str, deque[ChatTurn]] = {}

    def history(
        self, session_id: str, allowed_tags: frozenset[str], limit: int = 6
    ) -> list[ChatTurn]:
        if not session_id.strip() or not allowed_tags or limit <= 0:
            return []
        turns = list(self._sessions.get(_session_key(session_id, allowed_tags), ()))
        return turns[-limit:]

    def append(self, session_id: str, allowed_tags: frozenset[str], turn: ChatTurn) -> None:
        if not session_id.strip() or not allowed_tags:
            return
        key = _session_key(session_id, allowed_tags)
        bucket = self._sessions.get(key)
        if bucket is None:
            bucket = deque(maxlen=self._max_turns)
            self._sessions[key] = bucket
        bucket.append(turn)

    def clear(self, session_id: str, allowed_tags: frozenset[str]) -> None:
        if not session_id.strip() or not allowed_tags:
            return
        self._sessions.pop(_session_key(session_id, allowed_tags), None)


class RedisConversationStore(ConversationStore):
    """Redis-backed ACL-scoped conversation memory with fallback to in-memory on socket error."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 3600,
        max_turns: int = 10,
        client: _RespClient | None = None,
    ) -> None:
        self._client = client or _RespClient(redis_url)
        self._ttl = ttl_seconds
        self._max_turns = max_turns
        self._fallback = InMemoryConversationStore(max_turns=max_turns)

    def history(
        self, session_id: str, allowed_tags: frozenset[str], limit: int = 6
    ) -> list[ChatTurn]:
        if not session_id.strip() or not allowed_tags or limit <= 0:
            return []
        key = _session_key(session_id, allowed_tags)
        try:
            raw = self._client.command("GET", key)
            if not isinstance(raw, str):
                return self._fallback.history(session_id, allowed_tags, limit)
            items = json.loads(raw)
            turns = [
                ChatTurn(
                    question=str(item["question"]),
                    answer=str(item["answer"]),
                    abstained=bool(item.get("abstained", False)),
                )
                for item in items
            ]
            return turns[-limit:]
        except Exception:
            return self._fallback.history(session_id, allowed_tags, limit)

    def append(self, session_id: str, allowed_tags: frozenset[str], turn: ChatTurn) -> None:
        if not session_id.strip() or not allowed_tags:
            return
        self._fallback.append(session_id, allowed_tags, turn)
        turns = self._fallback.history(session_id, allowed_tags, self._max_turns)
        payload = json.dumps(
            [{"question": t.question, "answer": t.answer, "abstained": t.abstained} for t in turns]
        )
        key = _session_key(session_id, allowed_tags)
        try:
            self._client.command("SETEX", key, str(self._ttl), payload)
        except Exception:
            return

    def clear(self, session_id: str, allowed_tags: frozenset[str]) -> None:
        self._fallback.clear(session_id, allowed_tags)
        if not session_id.strip() or not allowed_tags:
            return
        key = _session_key(session_id, allowed_tags)
        try:
            self._client.command("SETEX", key, "1", "[]")
        except Exception:
            return

"""Multi-turn conversation memory port, isolated by (session_id, acl_tags)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatTurn:
    question: str
    answer: str
    abstained: bool = False


class ConversationStore(ABC):
    @abstractmethod
    def history(
        self, session_id: str, allowed_tags: frozenset[str], limit: int = 6
    ) -> list[ChatTurn]:
        """Return prior turns for this (session_id, allowed_tags) scope, oldest first."""

    @abstractmethod
    def append(self, session_id: str, allowed_tags: frozenset[str], turn: ChatTurn) -> None:
        """Append a completed turn to this ACL-scoped session."""

    @abstractmethod
    def clear(self, session_id: str, allowed_tags: frozenset[str]) -> None:
        """Clear history for this ACL-scoped session."""

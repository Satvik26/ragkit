"""Claim verification port: checks that each generated claim is entailed by its cited chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragkit.domain.models import ScoredChunk


class ClaimVerifier(ABC):
    @abstractmethod
    def verify(self, answer_text: str, evidence: Sequence[ScoredChunk], query: str = "") -> bool:
        """Return True iff every claim in `answer_text` is supported by its cited evidence."""

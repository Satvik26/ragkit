"""Query rewriting port: expands or clarifies a user query before first-stage retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod


class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str) -> str:
        """Return an expanded or clarified query string for retrieval."""

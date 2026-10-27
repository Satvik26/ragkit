"""Authentication & caller ACL resolution port."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AuthError(ValueError):
    """Raised when a bearer token is invalid, expired, or missing required signature."""


class TokenVerifier(ABC):
    @abstractmethod
    def resolve_tags(self, authorization: str, fallback_tags_header: str) -> frozenset[str]:
        """Verify token or dev header and return caller ACL tags."""

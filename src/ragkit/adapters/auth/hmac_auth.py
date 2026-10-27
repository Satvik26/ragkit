"""HMAC-SHA256 signed bearer token verifier with dev-header fallback when secret is unset."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from ragkit.ports.auth import AuthError, TokenVerifier


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + padding).encode("ascii"))


def sign_token(
    secret: str,
    sub: str,
    acl_tags: frozenset[str],
    exp: int | None = None,
) -> str:
    """Create a compact HMAC-SHA256 bearer token carrying `sub` and `acl_tags`."""
    payload = {
        "sub": sub,
        "tags": sorted(acl_tags),
        "exp": exp if exp is not None else int(time.time()) + 3600,
    }
    body = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64url_encode(sig)}"


_DEFAULT_DEV_SECRET = "ragkit-dev-secret"


class HmacTokenVerifier(TokenVerifier):
    """Verifies signed Bearer tokens; requires Bearer when `secret` is explicitly set."""

    def __init__(self, secret: str = "") -> None:
        self._require_bearer = bool(secret.strip())
        self._secret = secret.strip() or _DEFAULT_DEV_SECRET

    def resolve_tags(self, authorization: str, fallback_tags_header: str) -> frozenset[str]:
        auth_val = authorization.strip()
        if auth_val.lower().startswith("bearer "):
            token = auth_val[7:].strip()
            return self._verify(token)

        if self._require_bearer:
            raise AuthError("Bearer token is required when RAG_AUTH_SECRET is set")

        return frozenset(part.strip() for part in fallback_tags_header.split(",") if part.strip())

    def _verify(self, token: str) -> frozenset[str]:
        parts = token.split(".")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise AuthError("malformed bearer token")
        body_b64, sig_b64 = parts
        expected_sig = hmac.new(
            self._secret.encode("utf-8"), body_b64.encode("ascii"), hashlib.sha256
        ).digest()
        try:
            actual_sig = _b64url_decode(sig_b64)
        except Exception as exc:
            raise AuthError("invalid token signature encoding") from exc
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise AuthError("invalid token signature")
        try:
            payload = json.loads(_b64url_decode(body_b64).decode("utf-8"))
        except Exception as exc:
            raise AuthError("invalid token payload") from exc
        exp = int(payload.get("exp", 0))
        if exp < int(time.time()):
            raise AuthError("token has expired")
        raw_tags = payload.get("tags", [])
        if not isinstance(raw_tags, list):
            raise AuthError("token tags claim must be a list")
        return frozenset(str(t).strip() for t in raw_tags if str(t).strip())

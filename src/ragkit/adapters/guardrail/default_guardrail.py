"""Deterministic prompt-injection blocker and GDPR PII redactor."""

from __future__ import annotations

import re

from ragkit.ports.guardrail import Guardrail, GuardrailResult

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"(?:reveal|show|print|output)\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"forget\s+(?:all\s+)?(?:your\s+)?(?:rules|instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:dan|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"pretend\s+(?:the\s+)?evidence\s+says", re.IGNORECASE),
    re.compile(r"drop\s+table\s+\w+", re.IGNORECASE),
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")


class PassthroughGuardrail(Guardrail):
    """Allows all inputs unchanged."""

    def check_input(self, text: str) -> GuardrailResult:
        return GuardrailResult(allowed=True, sanitized_text=text)


class DefaultGuardrail(Guardrail):
    """Blocks prompt-injection/jailbreak payloads and masks GDPR PII (emails, cards, IBANs)."""

    def check_input(self, text: str) -> GuardrailResult:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                return GuardrailResult(
                    allowed=False,
                    sanitized_text=text,
                    reason="guardrail_blocked:prompt_injection",
                )

        sanitized = text
        redacted = False

        if _EMAIL_RE.search(sanitized):
            sanitized = _EMAIL_RE.sub("[REDACTED_EMAIL]", sanitized)
            redacted = True
        if _CARD_RE.search(sanitized):
            sanitized = _CARD_RE.sub("[REDACTED_CARD]", sanitized)
            redacted = True
        if _IBAN_RE.search(sanitized):
            sanitized = _IBAN_RE.sub("[REDACTED_IBAN]", sanitized)
            redacted = True

        return GuardrailResult(
            allowed=True,
            sanitized_text=sanitized,
            pii_redacted=redacted,
        )

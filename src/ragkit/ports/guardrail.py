"""Input and output guardrail port: blocks prompt injections and redacts GDPR PII."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    allowed: bool
    sanitized_text: str
    reason: str | None = None
    pii_redacted: bool = False


class Guardrail(ABC):
    @abstractmethod
    def check_input(self, text: str) -> GuardrailResult:
        """Inspect user input for prompt injection and redact PII before retrieval/LLM."""

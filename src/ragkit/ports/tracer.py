"""Observability & tracing port: records LangGraph node spans, sessions, and Langfuse scores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TraceSpan:
    name: str
    duration_ms: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class TraceScore:
    trace_id: str
    name: str
    value: float
    comment: str = ""


@dataclass(frozen=True, slots=True)
class TraceRecord:
    trace_id: str
    query: str
    route_kind: str
    abstained: bool
    reason: str | None
    total_ms: float
    nodes: tuple[str, ...] = ()
    spans: tuple[TraceSpan, ...] = field(default_factory=tuple)
    session_id: str | None = None
    user_id: str | None = None
    top_score: float = 0.0
    chunks_passed_gate: int = 0
    guardrail_status: str = "passed"
    output: str = ""
    generation: str = ""


class Tracer(ABC):
    @abstractmethod
    def record(self, trace: TraceRecord) -> None:
        """Persist or export a completed request trace and its node spans."""

    @abstractmethod
    def recent(self, limit: int = 20) -> list[TraceRecord]:
        """Return the most recent traces, newest first."""

    def score(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        """Record an evaluation or user-feedback score against `trace_id`."""
        del trace_id, name, value, comment

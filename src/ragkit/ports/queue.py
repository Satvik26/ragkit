"""Durable ingestion queue port with retry budget and dead-letter queue (DLQ)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

JobStatus = Literal["queued", "completed", "dead_letter"]


@dataclass(frozen=True, slots=True)
class IngestionJob:
    job_id: str
    doc_id: str
    text: str
    acl_tags: frozenset[str]
    attempts: int = 0
    max_attempts: int = 3
    status: JobStatus = "queued"
    last_error: str | None = None


class IngestionQueue(ABC):
    @abstractmethod
    def enqueue(
        self, doc_id: str, text: str, acl_tags: frozenset[str], max_attempts: int = 3
    ) -> IngestionJob:
        """Enqueue an idempotent document ingestion job (keyed by doc_id + content hash)."""

    @abstractmethod
    def process_next(self) -> IngestionJob | None:
        """Pop and process one queued job; retry on failure or move to dead_letter."""

    @abstractmethod
    def dead_letters(self) -> list[IngestionJob]:
        """Return jobs that exhausted their retry budget."""

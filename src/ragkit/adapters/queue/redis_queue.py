"""In-memory and Redis-backed durable ingestion queues with retries and dead-letter queue."""

from __future__ import annotations

from collections import deque

from ragkit.domain.catalog import content_hash
from ragkit.ports.queue import IngestionJob, IngestionQueue
from ragkit.services.ingestion import IngestionService


class InMemoryIngestionQueue(IngestionQueue):
    """Durable-semantics ingestion queue with idempotent job IDs, retries, and DLQ."""

    def __init__(self, ingestion: IngestionService) -> None:
        self._ingestion = ingestion
        self._pending: deque[str] = deque()
        self._jobs: dict[str, IngestionJob] = {}
        self._dlq: list[str] = []

    def enqueue(
        self, doc_id: str, text: str, acl_tags: frozenset[str], max_attempts: int = 3
    ) -> IngestionJob:
        job_id = content_hash(f"{doc_id}|{','.join(sorted(acl_tags))}|{text}")[:24]
        existing = self._jobs.get(job_id)
        if existing is not None and existing.status in ("queued", "completed"):
            return existing
        job = IngestionJob(
            job_id=job_id,
            doc_id=doc_id,
            text=text,
            acl_tags=acl_tags,
            attempts=0,
            max_attempts=max(1, max_attempts),
            status="queued",
        )
        self._jobs[job_id] = job
        self._pending.append(job_id)
        return job

    def process_next(self) -> IngestionJob | None:
        if not self._pending:
            return None
        job_id = self._pending.popleft()
        job = self._jobs[job_id]
        attempt = job.attempts + 1
        try:
            self._ingestion.ingest(job.doc_id, job.text, job.acl_tags)
            done = IngestionJob(
                job_id=job.job_id,
                doc_id=job.doc_id,
                text=job.text,
                acl_tags=job.acl_tags,
                attempts=attempt,
                max_attempts=job.max_attempts,
                status="completed",
            )
            self._jobs[job_id] = done
            return done
        except Exception as exc:
            if attempt < job.max_attempts:
                retried = IngestionJob(
                    job_id=job.job_id,
                    doc_id=job.doc_id,
                    text=job.text,
                    acl_tags=job.acl_tags,
                    attempts=attempt,
                    max_attempts=job.max_attempts,
                    status="queued",
                    last_error=str(exc),
                )
                self._jobs[job_id] = retried
                self._pending.append(job_id)
                return retried
            dead = IngestionJob(
                job_id=job.job_id,
                doc_id=job.doc_id,
                text=job.text,
                acl_tags=job.acl_tags,
                attempts=attempt,
                max_attempts=job.max_attempts,
                status="dead_letter",
                last_error=str(exc),
            )
            self._jobs[job_id] = dead
            self._dlq.append(job_id)
            return dead

    def dead_letters(self) -> list[IngestionJob]:
        return [self._jobs[jid] for jid in self._dlq if jid in self._jobs]

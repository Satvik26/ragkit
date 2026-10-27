"""Process-local catalog. Used when RAG_DATABASE_URL is empty, including unit tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from ragkit.domain.catalog import Clock, StoredChunk, StoredDocument, chunk_record, document_record
from ragkit.domain.models import Chunk
from ragkit.ports.catalog import DocumentCatalog


class InMemoryDocumentCatalog(DocumentCatalog):
    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._documents: dict[str, StoredDocument] = {}
        self._chunks: dict[str, StoredChunk] = {}

    def save(
        self, doc_id: str, text: str, acl_tags: frozenset[str], chunks: Sequence[Chunk]
    ) -> None:
        now = self._clock()
        previous = self._documents.get(doc_id)
        created_at = previous.created_at if previous is not None else now
        self._documents[doc_id] = document_record(doc_id, text, acl_tags, created_at, now)
        for chunk in chunks:
            record = chunk_record(chunk)
            self._chunks[record.id] = record

    def document(self, doc_id: str) -> StoredDocument | None:
        return self._documents.get(doc_id)

    def chunks_for(self, doc_id: str) -> list[StoredChunk]:
        rows = [chunk for chunk in self._chunks.values() if chunk.doc_id == doc_id]
        return sorted(rows, key=lambda chunk: chunk.start)

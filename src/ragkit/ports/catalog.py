"""Durable record of ingested documents. Search stays on the vector store."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragkit.domain.catalog import StoredChunk, StoredDocument
from ragkit.domain.models import Chunk


class DocumentCatalog(ABC):
    @abstractmethod
    def save(
        self, doc_id: str, text: str, acl_tags: frozenset[str], chunks: Sequence[Chunk]
    ) -> None:
        """Insert or overwrite the document and these chunks. Same ids overwrite."""

    @abstractmethod
    def document(self, doc_id: str) -> StoredDocument | None:
        """The document row, or None when it was never stored."""

    @abstractmethod
    def chunks_for(self, doc_id: str) -> list[StoredChunk]:
        """Chunk rows for one document, in window order."""

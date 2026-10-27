from __future__ import annotations

from collections.abc import Sequence

from ragkit.domain.models import Chunk
from ragkit.ports.cache import QueryCache
from ragkit.ports.catalog import DocumentCatalog
from ragkit.ports.chunker import Chunker
from ragkit.ports.embedder import Embedder
from ragkit.ports.lexical_store import LexicalWriter
from ragkit.ports.vector_store import VectorWriter


class IngestionService:
    """Turns one document into stored vectors and a durable row. Depends only on ports."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        writer: VectorWriter,
        catalog: DocumentCatalog,
        lexical_writer: LexicalWriter | None = None,
        cache: QueryCache | None = None,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._writer = writer
        self._catalog = catalog
        self._lexical_writer = lexical_writer
        self._cache = cache

    def ingest(self, doc_id: str, text: str, acl_tags: frozenset[str]) -> list[Chunk]:
        chunks = self._chunker.chunk(doc_id, text, acl_tags)
        if not chunks:
            return []
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        # Catalog first. If the vector write then fails, a retry overwrites the same ids.
        self._catalog.save(doc_id, text, acl_tags, chunks)
        self._writer.upsert(chunks, vectors)
        if self._lexical_writer is not None:
            self._lexical_writer.upsert(chunks)
        if self._cache is not None:
            self._cache.invalidate_all()
        return chunks

    def ingest_many(
        self, documents: Sequence[tuple[str, str]], acl_tags: frozenset[str]
    ) -> list[tuple[str, list[Chunk]]]:
        """Batch-embed and upsert a group of documents to minimize network round-trips."""
        prepared: list[tuple[str, str, list[Chunk]]] = []
        all_chunks: list[Chunk] = []
        for doc_id, text in documents:
            chunks = self._chunker.chunk(doc_id, text, acl_tags)
            if chunks:
                prepared.append((doc_id, text, chunks))
                all_chunks.extend(chunks)
        if not all_chunks:
            return []
        vectors = self._embedder.embed([chunk.text for chunk in all_chunks])
        for doc_id, text, chunks in prepared:
            self._catalog.save(doc_id, text, acl_tags, chunks)
        self._writer.upsert(all_chunks, vectors)
        if self._lexical_writer is not None:
            self._lexical_writer.upsert(all_chunks)
        if self._cache is not None:
            self._cache.invalidate_all()
        return [(doc_id, chunks) for doc_id, _text, chunks in prepared]

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ragkit.adapters.catalog.memory import InMemoryDocumentCatalog
from ragkit.adapters.chunk.fixed_size import FixedSizeChunker
from ragkit.adapters.dev.stubs import HashingEmbedder
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.domain.models import Chunk
from ragkit.ports.catalog import DocumentCatalog
from ragkit.ports.embedder import Embedder
from ragkit.services.ingestion import IngestionService

ENG = frozenset({"eng"})


class RecordingEmbedder(Embedder):
    def __init__(self) -> None:
        self.texts: list[str] = []
        self._inner = HashingEmbedder(32)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return self._inner.embed(texts)


def service() -> tuple[
    IngestionService, InMemoryVectorStore, RecordingEmbedder, InMemoryDocumentCatalog
]:
    store = InMemoryVectorStore()
    embedder = RecordingEmbedder()
    catalog = InMemoryDocumentCatalog()
    ingest = IngestionService(FixedSizeChunker(size=4, overlap=0), embedder, store, catalog)
    return ingest, store, embedder, catalog


def test_ingest_stores_a_searchable_chunk() -> None:
    ingest, store, embedder, catalog = service()
    chunks = ingest.ingest("policy", "Refunds take fourteen days.", ENG)
    assert len(chunks) == 1
    assert embedder.texts == [chunks[0].text]
    [vector] = embedder.embed([chunks[0].text])
    hits = store.search(vector, 5, ENG)
    assert [hit.chunk.id for hit in hits] == [chunks[0].id]
    assert hits[0].chunk.doc_id == "policy"
    stored = catalog.document("policy")
    assert stored is not None and stored.acl_tags == ENG
    assert [row.text for row in catalog.chunks_for("policy")] == [chunks[0].text]


def test_second_ingest_overwrites_instead_of_duplicating() -> None:
    ingest, store, embedder, catalog = service()
    text = "Refunds take fourteen days."
    ingest.ingest("policy", text, ENG)
    ingest.ingest("policy", text, ENG)
    [vector] = embedder.embed([text])
    assert len(store.search(vector, 10, ENG)) == 1
    assert len(catalog.chunks_for("policy")) == 1


def test_other_acl_group_cannot_see_the_document() -> None:
    ingest, store, embedder, _catalog = service()
    chunks = ingest.ingest("policy", "Refunds take fourteen days.", ENG)
    [vector] = embedder.embed([chunks[0].text])
    assert store.search(vector, 5, frozenset({"hr"})) == []


def test_blank_text_writes_nothing() -> None:
    ingest, store, embedder, catalog = service()
    assert ingest.ingest("policy", "   ", ENG) == []
    assert embedder.texts == []
    assert store.search([1.0, 0.0], 5, ENG) == []
    assert catalog.document("policy") is None


def test_catalog_failure_does_not_write_vectors() -> None:
    store = InMemoryVectorStore()
    embedder = RecordingEmbedder()

    class Broken(DocumentCatalog):
        def save(
            self, doc_id: str, text: str, acl_tags: frozenset[str], chunks: Sequence[Chunk]
        ) -> None:
            raise RuntimeError("postgres down")

        def document(self, doc_id: str) -> None:
            return None

        def chunks_for(self, doc_id: str) -> list[object]:
            return []

    ingest = IngestionService(FixedSizeChunker(size=4, overlap=0), embedder, store, Broken())
    with pytest.raises(RuntimeError, match="postgres down"):
        ingest.ingest("policy", "Refunds take fourteen days.", ENG)
    assert store.search(embedder.embed(["Refunds take fourteen days."])[0], 5, ENG) == []


def test_long_text_embeds_each_chunk() -> None:
    ingest, store, embedder, catalog = service()
    text = "one two three four five six seven eight"
    chunks = ingest.ingest("policy", text, ENG)
    assert [c.text for c in chunks] == ["one two three four", "five six seven eight"]
    assert embedder.texts == [c.text for c in chunks]
    [vector] = HashingEmbedder(32).embed([chunks[1].text])
    hits = store.search(vector, 2, ENG)
    assert hits[0].chunk.id == chunks[1].id
    assert [row.start for row in catalog.chunks_for("policy")] == [0, 4]

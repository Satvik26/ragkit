from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragkit.adapters.catalog.memory import InMemoryDocumentCatalog
from ragkit.adapters.chunk.fixed_size import FixedSizeChunker
from ragkit.adapters.dev.stubs import ExtractiveLLM, HashingEmbedder
from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.api.main import create_app
from ragkit.services.answering import AnswerService
from ragkit.services.folder_ingest import FolderIngestError, ingest_folder
from ragkit.services.ingestion import IngestionService
from ragkit.services.retrieval import RetrievalService

ENG = frozenset({"eng"})


def _stack() -> tuple[IngestionService, InMemoryDocumentCatalog]:
    embedder = HashingEmbedder(32)
    store = InMemoryVectorStore()
    catalog = InMemoryDocumentCatalog()
    ingestion = IngestionService(FixedSizeChunker(size=500, overlap=0), embedder, store, catalog)
    return ingestion, catalog


def _client(root: Path) -> TestClient:
    embedder = HashingEmbedder(32)
    store = InMemoryVectorStore()
    retrieval = RetrievalService(embedder, store, PassthroughReranker())
    ingestion = IngestionService(FixedSizeChunker(), embedder, store, InMemoryDocumentCatalog())
    app = create_app(AnswerService(retrieval, ExtractiveLLM()), ingestion, corpus_dir=str(root))
    return TestClient(app)


def test_folder_stores_text_and_markdown_only(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "policy.md").write_text("Refunds take fourteen days.", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("Salary bands stay confidential.", encoding="utf-8")
    (tmp_path / "scan.pdf").write_text("not ingested", encoding="utf-8")
    (tmp_path / ".secret.md").write_text("hidden", encoding="utf-8")
    ingestion, catalog = _stack()
    stored = ingest_folder(ingestion, tmp_path, "", ENG)
    assert [(item.doc_id, item.chunks) for item in stored] == [
        ("notes/policy.md", 1),
        ("readme.txt", 1),
    ]
    assert catalog.document("notes/policy.md") is not None
    assert catalog.document("scan.pdf") is None


def test_second_pass_overwrites_the_same_chunk(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("Salary bands stay confidential.", encoding="utf-8")
    ingestion, catalog = _stack()
    ingest_folder(ingestion, tmp_path, "", ENG)
    ingest_folder(ingestion, tmp_path, "", ENG)
    assert len(catalog.chunks_for("readme.txt")) == 1


def test_blank_file_writes_no_chunks(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    ingestion, catalog = _stack()
    stored = ingest_folder(ingestion, tmp_path, "", ENG)
    assert stored == []
    assert catalog.document("empty.md") is None


def test_absolute_path_and_missing_directory_are_rejected(tmp_path: Path) -> None:
    ingestion, _catalog = _stack()
    with pytest.raises(FolderIngestError, match="relative"):
        ingest_folder(ingestion, tmp_path, "/etc", ENG)
    with pytest.raises(FolderIngestError, match="not a directory"):
        ingest_folder(ingestion, tmp_path, "missing", ENG)


def test_non_utf8_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "broken.txt").write_bytes(b"\xff")
    ingestion, _catalog = _stack()
    with pytest.raises(FolderIngestError, match="UTF-8"):
        ingest_folder(ingestion, tmp_path, "", ENG)


def test_path_that_escapes_the_corpus_is_rejected(tmp_path: Path) -> None:
    ingestion, _catalog = _stack()
    with pytest.raises(FolderIngestError, match="escapes"):
        ingest_folder(ingestion, tmp_path, "../outside", ENG)


def test_api_ingests_a_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "policy.md").write_text("Refunds take fourteen days.", encoding="utf-8")
    response = _client(tmp_path).post(
        "/v1/documents/folder",
        json={"path": "notes"},
        headers={"x-user-tags": "eng"},
    )
    assert response.status_code == 200
    assert response.json() == {"files": [{"doc_id": "notes/policy.md", "chunks": 1}]}


def test_api_rejects_a_missing_corpus_and_missing_tags(tmp_path: Path) -> None:
    embedder = HashingEmbedder(8)
    store = InMemoryVectorStore()
    retrieval = RetrievalService(embedder, store, PassthroughReranker())
    ingestion = IngestionService(FixedSizeChunker(), embedder, store, InMemoryDocumentCatalog())
    client = TestClient(create_app(AnswerService(retrieval, ExtractiveLLM()), ingestion))
    missing_root = client.post(
        "/v1/documents/folder", json={"path": ""}, headers={"x-user-tags": "eng"}
    )
    assert missing_root.status_code == 400
    missing_tags = _client(tmp_path).post("/v1/documents/folder", json={"path": ""})
    assert missing_tags.status_code == 400

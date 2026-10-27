from __future__ import annotations

from fastapi.testclient import TestClient

from ragkit.adapters.catalog.memory import InMemoryDocumentCatalog
from ragkit.adapters.chunk.fixed_size import FixedSizeChunker
from ragkit.adapters.dev.stubs import ExtractiveLLM, HashingEmbedder
from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.api.main import create_app
from ragkit.domain.models import Chunk
from ragkit.services.answering import AnswerService
from ragkit.services.ingestion import IngestionService
from ragkit.services.retrieval import RetrievalService


def make_client(seed: bool = True) -> TestClient:
    embedder = HashingEmbedder(64)
    store = InMemoryVectorStore()
    if seed:
        text = "Refunds take fourteen days"
        store.upsert([Chunk("c1", "policy.pdf", text, frozenset({"eng"}))], embedder.embed([text]))
    retrieval = RetrievalService(embedder, store, PassthroughReranker())
    ingestion = IngestionService(FixedSizeChunker(), embedder, store, InMemoryDocumentCatalog())
    return TestClient(create_app(AnswerService(retrieval, ExtractiveLLM()), ingestion))


def test_health_and_ready() -> None:
    client = make_client()
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_ready_fails_when_the_vector_store_is_down() -> None:
    embedder = HashingEmbedder(8)
    store = InMemoryVectorStore()
    retrieval = RetrievalService(embedder, store, PassthroughReranker())
    ingestion = IngestionService(FixedSizeChunker(), embedder, store, InMemoryDocumentCatalog())
    client = TestClient(
        create_app(AnswerService(retrieval, ExtractiveLLM()), ingestion, lambda: False)
    )
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert client.get("/health").status_code == 200


def test_query_with_permission_returns_cited_answer() -> None:
    r = make_client().post(
        "/v1/query", json={"query": "refunds fourteen days"}, headers={"x-user-tags": "eng"}
    )
    body = r.json()
    assert r.status_code == 200 and not body["abstained"]
    assert body["citations"][0]["doc_id"] == "policy.pdf"
    assert "x-request-id" in r.headers


def test_query_without_tags_abstains() -> None:
    r = make_client().post("/v1/query", json={"query": "refunds fourteen days"})
    assert r.json()["abstained"] is True


def test_empty_query_rejected() -> None:
    assert make_client().post("/v1/query", json={"query": ""}).status_code == 422


def test_ingested_document_can_be_queried() -> None:
    client = make_client(seed=False)
    stored = client.post(
        "/v1/documents",
        json={"doc_id": "policy", "text": "Refunds take fourteen days"},
        headers={"x-user-tags": "eng"},
    )
    assert stored.status_code == 200
    assert stored.json() == {"doc_id": "policy", "chunks": 1}
    answer = client.post(
        "/v1/query",
        json={"query": "refunds fourteen days"},
        headers={"x-user-tags": "eng"},
    )
    body = answer.json()
    assert answer.status_code == 200 and not body["abstained"]
    assert body["citations"][0]["doc_id"] == "policy"


def test_document_without_tags_is_rejected() -> None:
    response = make_client(seed=False).post(
        "/v1/documents", json={"doc_id": "policy", "text": "Refunds take fourteen days"}
    )
    assert response.status_code == 400


def test_other_group_cannot_query_ingested_document() -> None:
    client = make_client(seed=False)
    client.post(
        "/v1/documents",
        json={"doc_id": "policy", "text": "Refunds take fourteen days"},
        headers={"x-user-tags": "eng"},
    )
    answer = client.post(
        "/v1/query",
        json={"query": "refunds fourteen days"},
        headers={"x-user-tags": "hr"},
    )
    assert answer.json()["abstained"] is True


def test_inspector_ui_and_retrieve_endpoint() -> None:
    client = make_client()
    ui = client.get("/")
    assert ui.status_code == 200
    assert "Retrieval &amp; Grounding Inspector" in ui.text

    ret = client.post(
        "/v1/retrieve",
        json={"query": "refunds fourteen days"},
        headers={"x-user-tags": "eng"},
    )
    assert ret.status_code == 200
    body = ret.json()
    assert len(body["retrieved_chunks"]) == 1
    assert body["retrieved_chunks"][0]["doc_id"] == "policy.pdf"
    assert body["retrieved_chunks"][0]["passed_gate"] is True

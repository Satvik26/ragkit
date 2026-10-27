from __future__ import annotations

from ragkit.adapters.embed.http import HttpEmbedder
from ragkit.adapters.llm.ollama import OllamaClient
from ragkit.adapters.rerank.http import HttpCrossEncoderReranker
from ragkit.config.container import build_services
from ragkit.config.settings import Settings


def test_ready_is_false_when_postgres_refuses_the_connection() -> None:
    _answer, _ingestion, ready = build_services(
        Settings(database_url="postgresql://rag:rag@127.0.0.1:1/rag")
    )
    assert ready is not None
    assert ready() is False


def test_embedding_url_uses_the_http_embedder() -> None:
    _answer, ingestion, ready = build_services(Settings(embedding_url="http://127.0.0.1:1"))
    assert isinstance(ingestion._embedder, HttpEmbedder)
    assert ready is not None
    assert ready() is False


def test_reranker_url_uses_the_http_cross_encoder_reranker() -> None:
    answer, _ingestion, ready = build_services(Settings(reranker_url="http://127.0.0.1:1"))
    assert isinstance(answer._retrieval._reranker, HttpCrossEncoderReranker)
    assert ready is not None
    assert ready() is False


def test_ollama_url_uses_the_ollama_client() -> None:
    answer, _ingestion, ready = build_services(
        Settings(llm_provider="ollama", ollama_url="http://127.0.0.1:1")
    )
    assert isinstance(answer._llm, OllamaClient)
    assert ready is not None
    assert ready() is False

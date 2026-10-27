from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from ragkit.adapters.auth.hmac_auth import HmacTokenVerifier, sign_token
from ragkit.adapters.cache.redis_cache import InMemoryQueryCache, RedisQueryCache, _RespClient
from ragkit.adapters.catalog.memory import InMemoryDocumentCatalog
from ragkit.adapters.chunk.fixed_size import FixedSizeChunker
from ragkit.adapters.conversation.store import InMemoryConversationStore
from ragkit.adapters.dev.stubs import ExtractiveLLM, HashingEmbedder
from ragkit.adapters.queue.redis_queue import InMemoryIngestionQueue
from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.api.main import create_app
from ragkit.config.container import build_services
from ragkit.config.settings import Settings
from ragkit.domain.models import Answer, Chunk, Citation
from ragkit.ports.auth import AuthError
from ragkit.ports.cache import cache_key
from ragkit.services.answering import AnswerService
from ragkit.services.ingestion import IngestionService
from ragkit.services.retrieval import RetrievalService
from tests.fakes import KeywordEmbedder, ScriptedLLM

ENG = frozenset({"eng"})
FINANCE = frozenset({"finance"})


class _FakeRespClient(_RespClient):
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.epoch = 0
        self.fail = False

    def command(self, *parts: str) -> str | int | None:
        if self.fail:
            raise ConnectionError("redis down")
        op = parts[0].upper()
        if op == "PING":
            return "PONG"
        if op == "GET":
            return self.store.get(parts[1])
        if op == "SETEX":
            self.store[parts[1]] = parts[3]
            return "OK"
        if op == "INCR":
            self.epoch += 1
            self.store[parts[1]] = str(self.epoch)
            return self.epoch
        return None


def test_acl_scoped_cache_prevents_cross_tenant_leak_and_invalidates_on_ingest() -> None:
    store = InMemoryVectorStore()
    catalog = InMemoryDocumentCatalog()
    cache = InMemoryQueryCache()
    embedder = KeywordEmbedder({"refund?": [1.0, 0.0], "Refunds take 14 days.": [1.0, 0.0]})
    ingestion = IngestionService(FixedSizeChunker(), embedder, store, catalog, cache=cache)
    ingestion.ingest("refunds.md", "Refunds take 14 days.", ENG)

    llm = ScriptedLLM("Refunds take 14 days [1].")
    retrieval = RetrievalService(embedder, store, PassthroughReranker())
    service = AnswerService(retrieval, llm, min_evidence_score=0.5, cache=cache)

    first = service.run_traced("refund?", ENG)
    assert not first.answer.abstained
    assert len(llm.prompts) == 1

    # Second call with same ACL tag hits cache without calling LLM
    second = service.run_traced("refund?", ENG)
    assert second.answer == first.answer
    assert second.trace.nodes == ("cache_hit",)
    assert len(llm.prompts) == 1

    # Different ACL tag (FINANCE) MUST NOT see ENG cached answer
    finance_run = service.run_traced("refund?", FINANCE)
    assert finance_run.answer.abstained is True
    assert finance_run.trace.nodes != ("cache_hit",)

    # Re-ingesting invalidates cache
    ingestion.ingest("refunds.md", "Refunds take 14 days.", ENG)
    after_ingest = service.run_traced("refund?", ENG)
    assert after_ingest.trace.nodes != ("cache_hit",)
    assert len(llm.prompts) == 2


def test_stream_and_session_repeat_serve_acl_cache() -> None:
    store = InMemoryVectorStore()
    catalog = InMemoryDocumentCatalog()
    cache = InMemoryQueryCache()
    conv = InMemoryConversationStore()
    embedder = KeywordEmbedder(
        {
            "refund?": [1.0, 0.0],
            "Refunds take 14 days.": [1.0, 0.0],
            "refund: And what about it?": [1.0, 0.0],
            "refund? in more detail": [1.0, 0.0],
            "What is the hotel cap?": [0.0, 1.0],
        }
    )
    ingestion = IngestionService(FixedSizeChunker(), embedder, store, catalog, cache=cache)
    ingestion.ingest("refunds.md", "Refunds take 14 days.", ENG)
    llm = ScriptedLLM("Refunds take 14 days [1].")
    retrieval = RetrievalService(embedder, store, PassthroughReranker())
    service = AnswerService(
        retrieval, llm, min_evidence_score=0.5, cache=cache, conversations=conv
    )

    first_events = list(service.stream_traced("refund?", ENG, session_id="sess-1"))
    first_done = next(event["result"] for event in first_events if event["type"] == "done")
    assert "cache_hit" not in first_done.trace.nodes
    assert any(event["type"] == "token" for event in first_events)
    assert len(llm.prompts) == 1

    # Identical question in the same session must hit cache and skip the LLM.
    repeat_events = list(service.stream_traced("refund?", ENG, session_id="sess-1"))
    assert [event["type"] for event in repeat_events] == ["done"]
    repeat = repeat_events[0]["result"]
    assert repeat.trace.nodes == ("cache_hit",)
    assert repeat.answer == first_done.answer
    assert len(llm.prompts) == 1

    # Pronoun follow-up rewrites first, so it must miss the prior turn's cache key.
    follow_events = list(service.stream_traced("And what about it?", ENG, session_id="sess-1"))
    follow = next(event["result"] for event in follow_events if event["type"] == "done")
    assert "cache_hit" not in follow.trace.nodes
    assert "contextualize" in follow.trace.nodes
    assert len(llm.prompts) == 2

    # Non-stream path also serves the cached exact repeat while history exists.
    again = service.run_traced("refund?", ENG, session_id="sess-1")
    assert again.trace.nodes == ("cache_hit",)
    assert again.answer == first_done.answer
    assert len(llm.prompts) == 2

    list(service.stream_traced("refund?", ENG, session_id="sess-detail"))
    detail_events = list(
        service.stream_traced("can yo tell about in detail?", ENG, session_id="sess-detail")
    )
    detail = next(event["result"] for event in detail_events if event["type"] == "done")
    assert "contextualize" in detail.trace.nodes
    assert detail.contextualized_query == "refund? in more detail"
    assert "cache_hit" not in detail.trace.nodes

    topic_events = list(
        service.stream_traced("What is the hotel cap?", ENG, session_id="sess-detail")
    )
    topic = next(event["result"] for event in topic_events if event["type"] == "done")
    assert "contextualize" not in topic.trace.nodes
    assert topic.contextualized_query == "What is the hotel cap?"


def test_cache_key_separates_models_sharing_one_store() -> None:
    assert cache_key("refund?", ENG, "0", "gemma4:e2b") != cache_key("refund?", ENG, "0", "gpt-4.1")
    fake = _FakeRespClient()
    gemma = RedisQueryCache("redis://localhost:6379/0", client=fake, model_name="gemma4:e2b")
    gpt = RedisQueryCache("redis://localhost:6379/0", client=fake, model_name="gpt-4.1")
    ans = Answer(text="14 days [1].", citations=(Citation("c1", "d1"),), abstained=False)
    gemma.set("refund?", ENG, ans)
    assert gemma.get("refund?", ENG) == ans
    assert gpt.get("refund?", ENG) is None


def test_redis_query_cache_roundtrip_and_failure_tolerance() -> None:
    fake = _FakeRespClient()
    cache = RedisQueryCache("redis://localhost:6379/0", client=fake)
    assert cache.ready() is True

    ans = Answer(text="14 days [1].", citations=(Citation("c1", "d1"),), abstained=False)
    assert cache.get("q", frozenset()) is None
    cache.set("q", frozenset(), ans)

    cache.set("q", ENG, ans)
    assert cache.get("q", ENG) == ans
    assert cache.get("q", FINANCE) is None

    cache.invalidate_all()
    assert cache.get("q", ENG) is None

    fake.fail = True
    assert cache.ready() is False
    assert cache.get("q", ENG) is None
    cache.set("q", ENG, ans)
    cache.invalidate_all()


def test_ingestion_queue_retries_and_moves_poison_job_to_dlq() -> None:
    calls = 0

    class _FlakyIngestion(IngestionService):
        def __init__(self) -> None:
            pass

        def ingest(self, doc_id: str, text: str, acl_tags: frozenset[str]) -> list[Chunk]:
            nonlocal calls
            calls += 1
            if doc_id == "poison.md":
                raise RuntimeError("corrupt payload")
            return [Chunk("c1", doc_id, text, acl_tags)]

    queue = InMemoryIngestionQueue(_FlakyIngestion())
    j1 = queue.enqueue("ok.md", "hello", ENG)
    j1_dup = queue.enqueue("ok.md", "hello", ENG)
    assert j1.job_id == j1_dup.job_id

    done = queue.process_next()
    assert done is not None and done.status == "completed"
    assert queue.process_next() is None

    # Poison job with max_attempts=2 fails twice then lands in dead_letter
    queue.enqueue("poison.md", "bad", ENG, max_attempts=2)
    attempt1 = queue.process_next()
    assert attempt1 is not None and attempt1.status == "queued" and attempt1.attempts == 1
    attempt2 = queue.process_next()
    assert attempt2 is not None and attempt2.status == "dead_letter" and attempt2.attempts == 2
    assert len(queue.dead_letters()) == 1
    assert queue.dead_letters()[0].doc_id == "poison.md"


def test_hmac_token_verifier_and_api_auth_enforcement() -> None:
    secret = "super-secret-test-key"
    verifier = HmacTokenVerifier(secret)
    token = sign_token(secret, sub="user-1", acl_tags=ENG)
    assert verifier.resolve_tags(f"Bearer {token}", "") == ENG

    # Missing bearer when secret is configured -> AuthError
    with pytest.raises(AuthError, match="Bearer token is required"):
        verifier.resolve_tags("", "eng")

    # Tampered signature -> AuthError
    with pytest.raises(AuthError, match="signature"):
        verifier.resolve_tags(f"Bearer {token}tampered", "")

    # Expired token -> AuthError
    expired = sign_token(secret, sub="user-1", acl_tags=ENG, exp=int(time.time()) - 10)
    with pytest.raises(AuthError, match="expired"):
        verifier.resolve_tags(f"Bearer {expired}", "")

    # API integration with HmacTokenVerifier and Job Queue endpoints
    embedder = HashingEmbedder(32)
    store = InMemoryVectorStore()
    retrieval = RetrievalService(embedder, store, PassthroughReranker())
    svc = AnswerService(retrieval, ExtractiveLLM())
    ing = IngestionService(FixedSizeChunker(), embedder, store, InMemoryDocumentCatalog())
    client = TestClient(create_app(svc, ing, token_verifier=verifier))

    unauth = client.post("/v1/query", json={"query": "test"}, headers={"x-user-tags": "eng"})
    assert unauth.status_code == 401

    job_resp = client.post(
        "/v1/jobs/documents",
        json={"doc_id": "policy.md", "text": "Refunds take fourteen days"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert job_resp.status_code == 200
    assert job_resp.json()["status"] == "completed"

    dlq_resp = client.get("/v1/jobs/dead-letter")
    assert dlq_resp.status_code == 200
    assert dlq_resp.json() == {"dead_letters": []}


def test_container_wires_redis_cache_when_redis_url_set() -> None:
    answer, _ing, ready = build_services(Settings(redis_url="redis://127.0.0.1:1/0"))
    assert isinstance(answer._cache, RedisQueryCache)
    assert ready is not None
    assert ready() is False

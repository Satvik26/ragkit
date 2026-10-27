from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import URLError

from fastapi.testclient import TestClient

from ragkit.adapters.agent.langgraph_crag import LangGraphCRAGRunner
from ragkit.adapters.catalog.memory import InMemoryDocumentCatalog
from ragkit.adapters.chunk.fixed_size import FixedSizeChunker
from ragkit.adapters.dev.stubs import ExtractiveLLM, HashingEmbedder
from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.rewrite.rewriter import TechnicalQueryRewriter
from ragkit.adapters.trace.langfuse import InMemoryTracer, LangfuseHttpTracer
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.adapters.verify.verifier import GroundingClaimVerifier
from ragkit.api.main import create_app
from ragkit.config.container import build_services
from ragkit.config.settings import Settings
from ragkit.domain.models import Chunk
from ragkit.services.answering import AnswerService
from ragkit.services.ingestion import IngestionService
from ragkit.services.retrieval import RetrievalService
from tests.fakes import KeywordEmbedder, ScriptedLLM

ENG = frozenset({"eng"})


def test_langgraph_crag_happy_path_and_trace_nodes() -> None:
    store = InMemoryVectorStore()
    store.upsert([Chunk("c1", "refunds.md", "Refunds take fourteen days.", ENG)], [[1.0, 0.0]])
    retrieval = RetrievalService(
        KeywordEmbedder({"refunds?": [1.0, 0.0]}), store, PassthroughReranker()
    )
    tracer = InMemoryTracer()
    runner = LangGraphCRAGRunner(
        retrieval=retrieval,
        llm=ScriptedLLM("Refunds take 14 days [1]."),
        min_evidence_score=0.5,
        verifier=GroundingClaimVerifier(),
        tracer=tracer,
    )
    res = runner.run("refunds?", ENG, trace_id="trace-123")
    assert not res.answer.abstained
    assert res.trace.trace_id == "trace-123"
    assert res.trace.nodes == ("route", "retrieve", "generate", "verify")
    assert [s.name for s in res.trace.spans] == ["route", "retrieve", "generate", "verify"]
    assert len(tracer.recent(5)) == 1
    assert tracer.recent(0) == []


def test_langgraph_crag_self_correcting_rewrite_retry_loop() -> None:
    store = InMemoryVectorStore()
    store.upsert([Chunk("c1", "refunds.md", "Refunds take fourteen days.", ENG)], [[1.0, 0.0]])
    rewriter = TechnicalQueryRewriter(extra_expansions={"ref-pol": "refunds fourteen days"})
    expanded = rewriter.rewrite("ref-pol")
    retrieval = RetrievalService(
        KeywordEmbedder(
            {
                "ref-pol": [0.0, 1.0],  # first attempt scores 0.0 (< 0.5 gate)
                expanded: [1.0, 0.0],  # rewritten attempt scores 1.0 (>= 0.5 gate)
            }
        ),
        store,
        PassthroughReranker(),
    )
    runner = LangGraphCRAGRunner(
        retrieval=retrieval,
        llm=ScriptedLLM("Refunds take fourteen days [1]."),
        min_evidence_score=0.5,
        verifier=GroundingClaimVerifier(),
        corrective_rewriter=rewriter,
    )
    res = runner.run("ref-pol", ENG)
    assert not res.answer.abstained
    assert res.trace.nodes == (
        "route",
        "retrieve",
        "rewrite_query",
        "retrieve",
        "generate",
        "verify",
    )


def test_langfuse_http_tracer_ships_batch_and_tolerates_network_failure() -> None:
    captured: dict[str, Any] = {}

    def ok_transport(request: Any, timeout: float) -> io.BytesIO:
        captured["url"] = request.full_url
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return io.BytesIO(b'{"status":"ok"}')

    tracer = LangfuseHttpTracer(
        "https://cloud.langfuse.com/",
        public_key="pk-lf-123",
        secret_key="sk-lf-456",
        transport=ok_transport,
    )
    store = InMemoryVectorStore()
    store.upsert([Chunk("c1", "refunds.md", "Refunds take fourteen days.", ENG)], [[1.0, 0.0]])
    retrieval = RetrievalService(
        KeywordEmbedder({"refunds?": [1.0, 0.0]}), store, PassthroughReranker()
    )
    service = AnswerService(retrieval, ScriptedLLM("Refunds take 14 days [1]."), tracer=tracer)
    service.answer("refunds?", ENG)

    assert captured["url"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert captured["auth"].startswith("Basic ")
    spans = captured["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert spans[0]["name"] == "ragkit.query"
    def _string_attr(span: dict[str, Any], key: str) -> str:
        for item in span["attributes"]:
            if item["key"] == key:
                return str(item["value"]["stringValue"])
        raise AssertionError(f"missing {key}")

    root_output = json.loads(_string_attr(spans[0], "langfuse.trace.output"))
    assert root_output["answer"] == "Refunds take 14 days [1]."
    assert root_output["abstained"] is False
    generate = next(span for span in spans if span["name"] == "langgraph.generate")
    assert _string_attr(generate, "langfuse.observation.output") == "Refunds take 14 days [1]."
    assert any(s["name"] == "langgraph.retrieve" for s in spans)
    assert len(tracer.recent(10)) == 1

    def failing_transport(_request: Any, timeout: float) -> io.BytesIO:
        raise URLError(f"offline {timeout}")

    resilient = LangfuseHttpTracer(
        "https://cloud.langfuse.com", "pk", "sk", transport=failing_transport
    )
    resilient.record(tracer.recent(1)[0])
    assert len(resilient.recent(1)) == 1


def test_api_traces_endpoint_and_container_langfuse_wiring() -> None:
    answer, _ing, _ready = build_services(
        Settings(
            langfuse_url="http://127.0.0.1:9",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )
    )
    assert isinstance(answer._tracer, LangfuseHttpTracer)

    embedder = HashingEmbedder(32)
    store = InMemoryVectorStore()
    text = "Refunds take fourteen days"
    store.upsert([Chunk("c1", "policy.md", text, ENG)], embedder.embed([text]))
    retrieval = RetrievalService(embedder, store, PassthroughReranker())
    tracer = InMemoryTracer()
    svc = AnswerService(retrieval, ExtractiveLLM(), tracer=tracer)
    ing = IngestionService(FixedSizeChunker(), embedder, store, InMemoryDocumentCatalog())
    client = TestClient(create_app(svc, ing))

    q_resp = client.post("/v1/query", json={"query": text}, headers={"x-user-tags": "eng"})
    assert q_resp.status_code == 200
    q_body = q_resp.json()
    assert q_body["nodes"] == ["route", "retrieve", "generate", "verify"]
    assert len(q_body["spans"]) == 4

    t_resp = client.get("/v1/traces")
    assert t_resp.status_code == 200
    traces = t_resp.json()["traces"]
    assert len(traces) == 1
    assert traces[0]["nodes"] == ["route", "retrieve", "generate", "verify"]

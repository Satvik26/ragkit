from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable, Iterator, Sequence
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ragkit.adapters.auth.hmac_auth import HmacTokenVerifier, sign_token
from ragkit.adapters.queue.redis_queue import InMemoryIngestionQueue
from ragkit.api.ui import INSPECTOR_HTML
from ragkit.config.container import build_services
from ragkit.config.settings import Settings
from ragkit.domain.models import Citation, ScoredChunk
from ragkit.ports.auth import AuthError, TokenVerifier
from ragkit.ports.queue import IngestionQueue
from ragkit.services.answering import AnswerService
from ragkit.services.folder_ingest import FolderIngestError, ingest_folder
from ragkit.services.ingestion import IngestionService


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=100)


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=100)
    value: float = Field(ge=0.0, le=1.0)
    comment: str = Field(default="", max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    trace_id: str
    value: float


class ChatTurnOut(BaseModel):
    question: str
    answer: str
    abstained: bool


class DocumentRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=100_000)


class DocumentResponse(BaseModel):
    doc_id: str
    chunks: int


class JobOut(BaseModel):
    job_id: str
    doc_id: str
    status: str
    attempts: int
    max_attempts: int
    last_error: str | None = None


class DeadLetterResponse(BaseModel):
    dead_letters: list[JobOut]


class TokenRequest(BaseModel):
    sub: str = Field(default="satvik", min_length=1, max_length=100)
    tags: list[str] = Field(default_factory=lambda: ["eng"])


class TokenResponse(BaseModel):
    token: str
    sub: str
    tags: list[str]


class FolderRequest(BaseModel):
    path: str = ""


class FolderFileOut(BaseModel):
    doc_id: str
    chunks: int


class FolderResponse(BaseModel):
    files: list[FolderFileOut]


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str


class SpanOut(BaseModel):
    name: str
    duration_ms: float
    detail: str = ""


class TraceOut(BaseModel):
    trace_id: str
    query: str
    route_kind: str
    abstained: bool
    reason: str | None = None
    total_ms: float
    nodes: list[str]
    spans: list[SpanOut]
    session_id: str | None = None
    guardrail_status: str = "passed"


class TracesResponse(BaseModel):
    traces: list[TraceOut]


class RetrievedChunkOut(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    text: str
    score: float
    passed_gate: bool
    cited: bool
    acl_tags: list[str]
    metadata: dict[str, str]


class RetrieveResponse(BaseModel):
    route_kind: str = "single"
    sub_queries: list[str] = Field(default_factory=list)
    min_evidence_score: float
    retrieved_chunks: list[RetrievedChunkOut]


class QueryResponse(BaseModel):
    trace_id: str = ""
    session_id: str | None = None
    answer: str
    citations: list[CitationOut]
    abstained: bool
    reason: str | None = None
    route_kind: str = "single"
    sub_queries: list[str] = Field(default_factory=list)
    contextualized_query: str = ""
    guardrail_status: str = "passed"
    min_evidence_score: float = 0.35
    retrieved_chunks: list[RetrievedChunkOut] = Field(default_factory=list)
    nodes: list[str] = Field(default_factory=list)
    spans: list[SpanOut] = Field(default_factory=list)
    history: list[ChatTurnOut] = Field(default_factory=list)


def _format_retrieved(
    scored: Sequence[ScoredChunk],
    min_score: float,
    citations: Sequence[Citation],
) -> list[RetrievedChunkOut]:
    cited_ids = {c.chunk_id for c in citations}
    return [
        RetrievedChunkOut(
            rank=rank,
            chunk_id=item.chunk.id,
            doc_id=item.chunk.doc_id,
            text=item.chunk.text,
            score=item.score,
            passed_gate=item.score >= min_score,
            cited=item.chunk.id in cited_ids,
            acl_tags=sorted(item.chunk.acl_tags),
            metadata=dict(item.chunk.metadata),
        )
        for rank, item in enumerate(scored, start=1)
    ]


def create_app(
    service: AnswerService,
    ingestion: IngestionService,
    ready_check: Callable[[], bool] | None = None,
    corpus_dir: str | None = None,
    token_verifier: TokenVerifier | None = None,
    queue: IngestionQueue | None = None,
) -> FastAPI:
    app = FastAPI(title="ragkit", version="0.1.0")
    auth = token_verifier or HmacTokenVerifier()
    job_queue = queue or InMemoryIngestionQueue(ingestion)

    def _resolve_caller_tags(authorization: str, x_user_tags: str) -> frozenset[str]:
        try:
            return auth.resolve_tags(authorization, x_user_tags)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id", uuid.uuid4().hex)
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    @app.get("/", response_class=HTMLResponse)
    def inspector_ui() -> HTMLResponse:
        return HTMLResponse(INSPECTOR_HTML)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        # /health means the process is up. /ready means Qdrant, Postgres, Redis, and models answer.
        if ready_check is not None and not ready_check():
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    @app.post("/v1/auth/token", response_model=TokenResponse)
    def mint_token(body: TokenRequest) -> TokenResponse:
        secret = getattr(auth, "_secret", "ragkit-dev-secret")
        clean_tags = frozenset(t.strip() for t in body.tags if t.strip())
        tok = sign_token(secret, sub=body.sub, acl_tags=clean_tags)
        return TokenResponse(token=tok, sub=body.sub, tags=sorted(clean_tags))

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    def feedback(body: FeedbackRequest) -> FeedbackResponse:
        service.record_feedback(body.trace_id, body.value, body.comment)
        return FeedbackResponse(status="recorded", trace_id=body.trace_id, value=body.value)

    @app.get("/v1/traces", response_model=TracesResponse)
    def traces(limit: int = 20) -> TracesResponse:
        records = service.recent_traces(limit)
        return TracesResponse(
            traces=[
                TraceOut(
                    trace_id=rec.trace_id,
                    query=rec.query,
                    route_kind=rec.route_kind,
                    abstained=rec.abstained,
                    reason=rec.reason,
                    total_ms=rec.total_ms,
                    nodes=list(rec.nodes),
                    spans=[
                        SpanOut(name=s.name, duration_ms=s.duration_ms, detail=s.detail)
                        for s in rec.spans
                    ],
                    session_id=rec.session_id,
                    guardrail_status=rec.guardrail_status,
                )
                for rec in records
            ]
        )

    @app.post("/v1/documents", response_model=DocumentResponse)
    def documents(
        body: DocumentRequest,
        authorization: str = Header(default=""),
        x_user_tags: str = Header(default=""),
    ) -> DocumentResponse:
        tags = _resolve_caller_tags(authorization, x_user_tags)
        if not tags:
            raise HTTPException(
                status_code=400, detail="x-user-tags is required to store a document"
            )
        stored = ingestion.ingest(body.doc_id, body.text, tags)
        return DocumentResponse(doc_id=body.doc_id, chunks=len(stored))

    @app.post("/v1/jobs/documents", response_model=JobOut)
    def enqueue_document_job(
        body: DocumentRequest,
        authorization: str = Header(default=""),
        x_user_tags: str = Header(default=""),
    ) -> JobOut:
        tags = _resolve_caller_tags(authorization, x_user_tags)
        if not tags:
            raise HTTPException(
                status_code=400, detail="x-user-tags is required to store a document"
            )
        job = job_queue.enqueue(body.doc_id, body.text, tags)
        processed = job_queue.process_next() or job
        return JobOut(
            job_id=processed.job_id,
            doc_id=processed.doc_id,
            status=processed.status,
            attempts=processed.attempts,
            max_attempts=processed.max_attempts,
            last_error=processed.last_error,
        )

    @app.get("/v1/jobs/dead-letter", response_model=DeadLetterResponse)
    def dead_letters() -> DeadLetterResponse:
        return DeadLetterResponse(
            dead_letters=[
                JobOut(
                    job_id=j.job_id,
                    doc_id=j.doc_id,
                    status=j.status,
                    attempts=j.attempts,
                    max_attempts=j.max_attempts,
                    last_error=j.last_error,
                )
                for j in job_queue.dead_letters()
            ]
        )

    @app.post("/v1/documents/folder", response_model=FolderResponse)
    def folder(
        body: FolderRequest,
        authorization: str = Header(default=""),
        x_user_tags: str = Header(default=""),
    ) -> FolderResponse:
        tags = _resolve_caller_tags(authorization, x_user_tags)
        if not tags:
            raise HTTPException(
                status_code=400, detail="x-user-tags is required to store a document"
            )
        if not corpus_dir:
            raise HTTPException(status_code=400, detail="RAG_CORPUS_DIR is not set")
        try:
            stored = ingest_folder(ingestion, Path(corpus_dir), body.path, tags)
        except FolderIngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FolderResponse(
            files=[FolderFileOut(doc_id=item.doc_id, chunks=item.chunks) for item in stored]
        )

    @app.post("/v1/retrieve", response_model=RetrieveResponse)
    def retrieve(
        body: QueryRequest,
        authorization: str = Header(default=""),
        x_user_tags: str = Header(default=""),
    ) -> RetrieveResponse:
        tags = _resolve_caller_tags(authorization, x_user_tags)
        route_kind, sub_queries = service.plan_route(body.query)
        scored = service.retrieve_ranked(body.query, tags)
        return RetrieveResponse(
            route_kind=route_kind,
            sub_queries=sub_queries,
            min_evidence_score=service.min_evidence_score,
            retrieved_chunks=_format_retrieved(scored, service.min_evidence_score, ()),
        )

    @app.post("/v1/query/stream")
    def query_stream(
        body: QueryRequest,
        request: Request,
        authorization: str = Header(default=""),
        x_user_tags: str = Header(default=""),
    ) -> StreamingResponse:
        tags = _resolve_caller_tags(authorization, x_user_tags)
        trace_id = getattr(request.state, "request_id", None)

        def event_stream() -> Iterator[str]:
            for event in service.stream_traced(
                body.query, tags, trace_id=trace_id, session_id=body.session_id
            ):
                kind = event["type"]
                if kind == "retrieval":
                    payload = {
                        "type": "retrieval",
                        "route_kind": event["route_kind"],
                        "sub_queries": event["sub_queries"],
                        "contextualized_query": event["contextualized_query"],
                        "guardrail_status": event["guardrail_status"],
                        "min_evidence_score": service.min_evidence_score,
                        "retrieved_chunks": [
                            c.model_dump()
                            for c in _format_retrieved(
                                event["retrieved"], service.min_evidence_score, ()
                            )
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                elif kind == "token":
                    yield f"data: {json.dumps({'type': 'token', 'delta': event['delta']})}\n\n"
                elif kind == "done":
                    run = event["result"]
                    result = run.answer
                    turns = (
                        service.session_history(body.session_id, tags) if body.session_id else []
                    )
                    done_payload = QueryResponse(
                        trace_id=run.trace.trace_id,
                        session_id=body.session_id,
                        answer=result.text,
                        citations=[
                            CitationOut(chunk_id=c.chunk_id, doc_id=c.doc_id)
                            for c in result.citations
                        ],
                        abstained=result.abstained,
                        reason=result.reason,
                        route_kind=run.route_kind,
                        sub_queries=run.sub_queries,
                        contextualized_query=run.contextualized_query,
                        guardrail_status=run.guardrail_status,
                        min_evidence_score=service.min_evidence_score,
                        retrieved_chunks=_format_retrieved(
                            run.retrieved, service.min_evidence_score, result.citations
                        ),
                        nodes=list(run.trace.nodes),
                        spans=[
                            SpanOut(name=s.name, duration_ms=s.duration_ms, detail=s.detail)
                            for s in run.trace.spans
                        ],
                        history=[
                            ChatTurnOut(question=t.question, answer=t.answer, abstained=t.abstained)
                            for t in turns
                        ],
                    ).model_dump()
                    done_payload["type"] = "done"
                    yield f"data: {json.dumps(done_payload)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/query", response_model=QueryResponse)
    def query(
        body: QueryRequest,
        request: Request,
        authorization: str = Header(default=""),
        x_user_tags: str = Header(default=""),
    ) -> QueryResponse:
        tags = _resolve_caller_tags(authorization, x_user_tags)
        trace_id = getattr(request.state, "request_id", None)
        run = service.run_traced(
            body.query,
            tags,
            trace_id=trace_id,
            session_id=body.session_id,
        )
        result = run.answer
        turns = service.session_history(body.session_id, tags) if body.session_id else []
        return QueryResponse(
            trace_id=run.trace.trace_id,
            session_id=body.session_id,
            answer=result.text,
            citations=[CitationOut(chunk_id=c.chunk_id, doc_id=c.doc_id) for c in result.citations],
            abstained=result.abstained,
            reason=result.reason,
            route_kind=run.route_kind,
            sub_queries=run.sub_queries,
            contextualized_query=run.contextualized_query,
            guardrail_status=run.guardrail_status,
            min_evidence_score=service.min_evidence_score,
            retrieved_chunks=_format_retrieved(
                run.retrieved, service.min_evidence_score, result.citations
            ),
            nodes=list(run.trace.nodes),
            spans=[
                SpanOut(name=s.name, duration_ms=s.duration_ms, detail=s.detail)
                for s in run.trace.spans
            ],
            history=[
                ChatTurnOut(question=t.question, answer=t.answer, abstained=t.abstained)
                for t in turns
            ],
        )

    return app


_settings = Settings()
app = create_app(
    *build_services(_settings),
    corpus_dir=_settings.corpus_dir,
    token_verifier=HmacTokenVerifier(_settings.auth_secret),
)

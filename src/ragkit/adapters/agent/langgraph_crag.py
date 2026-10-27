"""LangGraph Corrective RAG (CRAG) state machine with Guardrails and Multi-Turn Context.

Graph topology:
  START -> (guard_input if guardrail configured) -> (contextualize if session history)
        -> route -> retrieve -> (evidence gate check)
            - if evidence >= threshold           -> generate -> verify -> END
            - if weak evidence & retry available -> rewrite_query -> retrieve
            - if weak evidence & no retry left   -> abstain -> END
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ragkit.adapters.conversation.store import contextualize_follow_up
from ragkit.domain.models import Answer, Citation, ScoredChunk
from ragkit.ports.conversation import ChatTurn
from ragkit.ports.guardrail import Guardrail
from ragkit.ports.llm import LLMClient
from ragkit.ports.query_rewriter import QueryRewriter
from ragkit.ports.tracer import Tracer, TraceRecord, TraceSpan
from ragkit.ports.verifier import ClaimVerifier
from ragkit.services.prompts import NOT_FOUND_TOKEN, build_prompt, format_model_answer
from ragkit.services.retrieval import RetrievalService

NOT_FOUND_TEXT = "I could not find this in the available documents."
_CITATION = re.compile(r"\[(\d+)\]")


class CRAGState(TypedDict):
    query: str
    active_query: str
    allowed_tags: frozenset[str]
    history: list[ChatTurn]
    guardrail_status: str
    route_kind: str
    sub_queries: list[str]
    retrieved: list[ScoredChunk]
    evidence: list[ScoredChunk]
    attempts: int
    raw_generation: str
    answer: Answer | None
    nodes: list[str]
    spans: list[TraceSpan]


@dataclass(frozen=True, slots=True)
class CRAGRunResult:
    answer: Answer
    retrieved: list[ScoredChunk]
    route_kind: str
    sub_queries: list[str]
    trace: TraceRecord
    contextualized_query: str = ""
    guardrail_status: str = "passed"


class LangGraphCRAGRunner:
    """Executes the Corrective RAG workflow using a compiled LangGraph StateGraph."""

    def __init__(
        self,
        retrieval: RetrievalService,
        llm: LLMClient,
        min_evidence_score: float = 0.35,
        verifier: ClaimVerifier | None = None,
        corrective_rewriter: QueryRewriter | None = None,
        tracer: Tracer | None = None,
        guardrail: Guardrail | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._min_score = min_evidence_score
        self._verifier = verifier
        self._corrective_rewriter = corrective_rewriter
        self._tracer = tracer
        self._guardrail = guardrail
        self._graph: Any = self._compile_graph()

    def _compile_graph(self) -> Any:
        builder: Any = StateGraph(CRAGState)
        builder.add_node("guard_input", self._node_guard_input)
        builder.add_node("contextualize", self._node_contextualize)
        builder.add_node("route", self._node_route)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("rewrite_query", self._node_rewrite)
        builder.add_node("generate", self._node_generate)
        builder.add_node("verify", self._node_verify)
        builder.add_node("abstain", self._node_abstain_no_evidence)

        builder.add_conditional_edges(
            START,
            self._decide_entry,
            {
                "guard_input": "guard_input",
                "contextualize": "contextualize",
                "route": "route",
            },
        )
        builder.add_conditional_edges(
            "guard_input",
            self._decide_after_guard,
            {
                "blocked": END,
                "contextualize": "contextualize",
                "route": "route",
            },
        )
        builder.add_edge("contextualize", "route")
        builder.add_edge("route", "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            self._decide_after_retrieve,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query",
                "abstain": "abstain",
            },
        )
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("generate", "verify")
        builder.add_edge("verify", END)
        builder.add_edge("abstain", END)
        return builder.compile()

    def _decide_entry(self, state: CRAGState) -> str:
        if self._guardrail is not None:
            return "guard_input"
        if state["history"]:
            return "contextualize"
        return "route"

    def _node_guard_input(self, state: CRAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        assert self._guardrail is not None
        res = self._guardrail.check_input(state["query"])
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if not res.allowed:
            reason = res.reason or "guardrail_blocked:prompt_injection"
            span = TraceSpan("guard_input", ms, reason)
            ans = Answer(text=NOT_FOUND_TEXT, citations=(), abstained=True, reason=reason)
            return {
                "guardrail_status": reason,
                "answer": ans,
                "nodes": [*state["nodes"], "guard_input"],
                "spans": [*state["spans"], span],
            }
        status = "pii_redacted" if res.pii_redacted else "passed"
        if res.pii_redacted:
            span = TraceSpan("guard_input", ms, "pii_redacted")
            return {
                "query": res.sanitized_text,
                "active_query": res.sanitized_text,
                "guardrail_status": status,
                "nodes": [*state["nodes"], "guard_input"],
                "spans": [*state["spans"], span],
            }
        return {
            "guardrail_status": status,
        }

    def _decide_after_guard(self, state: CRAGState) -> str:
        if state["answer"] is not None:
            return "blocked"
        if state["history"]:
            return "contextualize"
        return "route"

    def _node_contextualize(self, state: CRAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        resolved = contextualize_follow_up(state["active_query"], state["history"])
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if resolved != state["active_query"]:
            span = TraceSpan("contextualize", ms, resolved[:80])
            return {
                "active_query": resolved,
                "nodes": [*state["nodes"], "contextualize"],
                "spans": [*state["spans"], span],
            }
        return {}

    def _node_route(self, state: CRAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        plan = self._retrieval.plan_route(state["active_query"])
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        span = TraceSpan("route", ms, f"{plan.kind} ({len(plan.sub_queries)} hop)")
        return {
            "route_kind": plan.kind,
            "sub_queries": list(plan.sub_queries),
            "nodes": [*state["nodes"], "route"],
            "spans": [*state["spans"], span],
        }

    def _node_retrieve(self, state: CRAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        retrieved = self._retrieval.retrieve(state["active_query"], state["allowed_tags"])
        evidence = [item for item in retrieved if item.score >= self._min_score]
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        top_score = f"{retrieved[0].score:.3f}" if retrieved else "0.000"
        span = TraceSpan(
            "retrieve",
            ms,
            f"{len(retrieved)} fetched, {len(evidence)} passed gate (top={top_score})",
        )
        return {
            "retrieved": retrieved,
            "evidence": evidence,
            "attempts": state["attempts"] + 1,
            "nodes": [*state["nodes"], "retrieve"],
            "spans": [*state["spans"], span],
        }

    def _decide_after_retrieve(self, state: CRAGState) -> str:
        if state["evidence"]:
            return "generate"
        if state["attempts"] == 1 and self._corrective_rewriter is not None:
            rewritten = self._corrective_rewriter.rewrite(state["query"])
            if rewritten != state["active_query"]:
                return "rewrite_query"
        return "abstain"

    def _node_rewrite(self, state: CRAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        rewritten = (
            self._corrective_rewriter.rewrite(state["query"])
            if self._corrective_rewriter is not None
            else state["query"]
        )
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        span = TraceSpan("rewrite_query", ms, rewritten[:80])
        return {
            "active_query": rewritten,
            "nodes": [*state["nodes"], "rewrite_query"],
            "spans": [*state["spans"], span],
        }

    def _node_abstain_no_evidence(self, state: CRAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        ans = Answer(text=NOT_FOUND_TEXT, citations=(), abstained=True, reason="no_evidence")
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        span = TraceSpan("abstain", ms, "reason=no_evidence")
        return {
            "answer": ans,
            "nodes": [*state["nodes"], "abstain"],
            "spans": [*state["spans"], span],
        }

    def _node_generate(self, state: CRAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        prompt = build_prompt(state["active_query"], state["evidence"])
        raw = format_model_answer(self._llm.generate(prompt))
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        span = TraceSpan("generate", ms, f"{len(raw)} chars")
        return {
            "raw_generation": raw,
            "nodes": [*state["nodes"], "generate"],
            "spans": [*state["spans"], span],
        }

    def _node_verify(self, state: CRAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw = state["raw_generation"]
        evidence = state["evidence"]

        if raw == NOT_FOUND_TOKEN:
            ans = Answer(text=NOT_FOUND_TEXT, citations=(), abstained=True, reason="llm_abstained")
            detail = "abstained=llm_abstained"
        else:
            numbers = {int(n) for n in _CITATION.findall(raw)}
            valid_bounds = bool(numbers) and all(1 <= n <= len(evidence) for n in numbers)
            verified = valid_bounds and (
                self._verifier.verify(raw, evidence, query=state["active_query"])
                if self._verifier is not None
                else True
            )
            if not verified:
                ans = Answer(
                    text=NOT_FOUND_TEXT,
                    citations=(),
                    abstained=True,
                    reason="unsupported_answer",
                )
                detail = "abstained=unsupported_answer"
            else:
                citations = tuple(
                    Citation(evidence[n - 1].chunk.id, evidence[n - 1].chunk.doc_id)
                    for n in sorted(numbers)
                )
                ans = Answer(text=raw, citations=citations, abstained=False)
                detail = f"verified ({len(citations)} citations)"

        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        span = TraceSpan("verify", ms, detail)
        return {
            "answer": ans,
            "nodes": [*state["nodes"], "verify"],
            "spans": [*state["spans"], span],
        }

    def run(
        self,
        query: str,
        allowed_tags: frozenset[str],
        trace_id: str | None = None,
        history: list[ChatTurn] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> CRAGRunResult:
        t0 = time.perf_counter()
        initial: CRAGState = {
            "query": query,
            "active_query": query,
            "allowed_tags": allowed_tags,
            "history": list(history) if history else [],
            "guardrail_status": "passed",
            "route_kind": "single",
            "sub_queries": [query],
            "retrieved": [],
            "evidence": [],
            "attempts": 0,
            "raw_generation": "",
            "answer": None,
            "nodes": [],
            "spans": [],
        }
        final: CRAGState = self._graph.invoke(initial)
        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        answer = final["answer"] or Answer(
            text=NOT_FOUND_TEXT, citations=(), abstained=True, reason="no_evidence"
        )
        retrieved = final["retrieved"]
        top_score = retrieved[0].score if retrieved else 0.0
        passed_gate = len(final["evidence"])
        trace = TraceRecord(
            trace_id=trace_id or uuid.uuid4().hex,
            query=final["query"],
            route_kind=final["route_kind"],
            abstained=answer.abstained,
            reason=answer.reason,
            total_ms=total_ms,
            nodes=tuple(final["nodes"]),
            spans=tuple(final["spans"]),
            session_id=session_id,
            user_id=user_id,
            top_score=top_score,
            chunks_passed_gate=passed_gate,
            guardrail_status=final["guardrail_status"],
            output=answer.text,
            generation=final.get("raw_generation") or "",
        )
        if self._tracer is not None:
            self._tracer.record(trace)
        return CRAGRunResult(
            answer=answer,
            retrieved=retrieved,
            route_kind=final["route_kind"],
            sub_queries=final["sub_queries"],
            trace=trace,
            contextualized_query=final["active_query"],
            guardrail_status=final["guardrail_status"],
        )

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator
from typing import Any

from ragkit.adapters.agent.langgraph_crag import CRAGRunResult, LangGraphCRAGRunner
from ragkit.adapters.conversation.store import contextualize_follow_up
from ragkit.domain.models import Answer, Citation, ScoredChunk
from ragkit.ports.cache import QueryCache
from ragkit.ports.conversation import ChatTurn, ConversationStore
from ragkit.ports.guardrail import Guardrail
from ragkit.ports.llm import LLMClient
from ragkit.ports.query_rewriter import QueryRewriter
from ragkit.ports.tracer import Tracer, TraceRecord, TraceSpan
from ragkit.ports.verifier import ClaimVerifier
from ragkit.services.prompts import NOT_FOUND_TOKEN, build_prompt, format_model_answer
from ragkit.services.retrieval import RetrievalService

NOT_FOUND_TEXT = "I could not find this in the available documents."
_CITATION = re.compile(r"\[(\d+)\]")


class AnswerService:
    """Grounded answering with three guards: evidence gate, cite-only prompt, claim verification.

    Orchestrated through the compiled LangGraph Corrective-RAG state machine
    (`LangGraphCRAGRunner`), with input guardrails (`Guardrail`), multi-turn session
    memory (`ConversationStore`), ACL-scoped caching (`QueryCache`), and node-level
    tracing exported via the `Tracer` port.
    """

    def __init__(
        self,
        retrieval: RetrievalService,
        llm: LLMClient,
        min_evidence_score: float = 0.35,
        verifier: ClaimVerifier | None = None,
        corrective_rewriter: QueryRewriter | None = None,
        tracer: Tracer | None = None,
        cache: QueryCache | None = None,
        guardrail: Guardrail | None = None,
        conversations: ConversationStore | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._min_score = min_evidence_score
        self._verifier = verifier
        self._tracer = tracer
        self._cache = cache
        self._guardrail = guardrail
        self._conversations = conversations
        self._runner = LangGraphCRAGRunner(
            retrieval=retrieval,
            llm=llm,
            min_evidence_score=min_evidence_score,
            verifier=verifier,
            corrective_rewriter=corrective_rewriter,
            tracer=tracer,
            guardrail=guardrail,
        )

    @property
    def min_evidence_score(self) -> float:
        return self._min_score

    def plan_route(self, query: str) -> tuple[str, list[str]]:
        plan = self._retrieval.plan_route(query)
        return plan.kind, list(plan.sub_queries)

    def retrieve_ranked(self, query: str, allowed_tags: frozenset[str]) -> list[ScoredChunk]:
        """Scored chunks in retrieval order, before the evidence gate."""
        return self._retrieval.retrieve(query, allowed_tags)

    def ranked_chunk_ids(self, query: str, allowed_tags: frozenset[str]) -> list[str]:
        """Chunk ids in retrieval order, before the evidence gate. Used by the golden-set run."""
        return [item.chunk.id for item in self._retrieval.retrieve(query, allowed_tags)]

    def session_history(
        self, session_id: str, allowed_tags: frozenset[str], limit: int = 6
    ) -> list[ChatTurn]:
        if self._conversations is None or not session_id:
            return []
        return self._conversations.history(session_id, allowed_tags, limit=limit)

    def record_feedback(
        self, trace_id: str, value: float, comment: str = "", name: str = "user_feedback"
    ) -> None:
        if self._tracer is not None:
            self._tracer.score(trace_id=trace_id, name=name, value=value, comment=comment)

    def _serve_cache_hit(
        self,
        cache_query: str,
        allowed_tags: frozenset[str],
        *,
        trace_id: str | None,
        session_id: str | None,
        user_id: str | None = None,
        extra_nodes: tuple[str, ...] = (),
        extra_spans: tuple[TraceSpan, ...] = (),
        guardrail_status: str = "passed",
        total_ms: float | None = None,
    ) -> CRAGRunResult | None:
        """Return a cached answer keyed by the resolved query and ACL tags, or None."""
        if self._cache is None:
            return None
        t0 = time.perf_counter()
        cached = self._cache.get(cache_query, allowed_tags)
        if cached is None:
            return None
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        kind, sub_qs = self.plan_route(cache_query)
        span = TraceSpan("cache_hit", ms, f"acl={','.join(sorted(allowed_tags))}")
        trace = TraceRecord(
            trace_id=trace_id or uuid.uuid4().hex,
            query=cache_query,
            route_kind=kind,
            abstained=cached.abstained,
            reason=cached.reason,
            total_ms=ms if total_ms is None else total_ms,
            nodes=(*extra_nodes, "cache_hit"),
            spans=(*extra_spans, span),
            session_id=session_id,
            user_id=user_id,
            guardrail_status=guardrail_status,
            output=cached.text,
        )
        if self._tracer is not None:
            self._tracer.record(trace)
        if self._conversations is not None and session_id:
            self._conversations.append(
                session_id,
                allowed_tags,
                ChatTurn(cache_query, cached.text, cached.abstained),
            )
        return CRAGRunResult(
            answer=cached,
            retrieved=[],
            route_kind=kind,
            sub_queries=sub_qs,
            trace=trace,
            contextualized_query=cache_query,
            guardrail_status=guardrail_status,
        )

    def stream_traced(
        self,
        query: str,
        allowed_tags: frozenset[str],
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream retrieval event, live LLM tokens, and final verified answer event."""
        tid = trace_id or uuid.uuid4().hex
        t_start = time.perf_counter()
        nodes: list[str] = []
        spans: list[TraceSpan] = []
        guard_status = "passed"
        active_query = query

        # 1. Input Guardrail
        if self._guardrail is not None:
            t0 = time.perf_counter()
            gres = self._guardrail.check_input(query)
            gms = round((time.perf_counter() - t0) * 1000.0, 2)
            if not gres.allowed:
                reason = gres.reason or "guardrail_blocked:prompt_injection"
                nodes.append("guard_input")
                spans.append(TraceSpan("guard_input", gms, reason))
                ans = Answer(text=NOT_FOUND_TEXT, citations=(), abstained=True, reason=reason)
                trace = TraceRecord(
                    tid,
                    query,
                    "single",
                    True,
                    reason,
                    gms,
                    tuple(nodes),
                    tuple(spans),
                    session_id=session_id,
                    guardrail_status=reason,
                    output=ans.text,
                )
                if self._tracer is not None:
                    self._tracer.record(trace)
                yield {
                    "type": "done",
                    "result": CRAGRunResult(ans, [], "single", [query], trace, query, reason),
                }
                return
            if gres.pii_redacted:
                guard_status = "pii_redacted"
                active_query = gres.sanitized_text
                nodes.append("guard_input")
                spans.append(TraceSpan("guard_input", gms, "pii_redacted"))

        # 2. Multi-turn Contextualize
        history = (
            self._conversations.history(session_id, allowed_tags)
            if self._conversations is not None and session_id
            else []
        )
        if history:
            t0 = time.perf_counter()
            resolved = contextualize_follow_up(active_query, history)
            cms = round((time.perf_counter() - t0) * 1000.0, 2)
            if resolved != active_query:
                active_query = resolved
                nodes.append("contextualize")
                spans.append(TraceSpan("contextualize", cms, resolved[:80]))

        # 2b. ACL-scoped cache after pronouns are resolved so exact repeats hit Redis
        cached = self._serve_cache_hit(
            active_query,
            allowed_tags,
            trace_id=tid,
            session_id=session_id,
            extra_nodes=tuple(nodes),
            extra_spans=tuple(spans),
            guardrail_status=guard_status,
            total_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
        )
        if cached is not None:
            yield {"type": "done", "result": cached}
            return

        # 3. Route & Retrieve
        t0 = time.perf_counter()
        plan = self._retrieval.plan_route(active_query)
        rms = round((time.perf_counter() - t0) * 1000.0, 2)
        nodes.append("route")
        spans.append(TraceSpan("route", rms, f"{plan.kind} ({len(plan.sub_queries)} hop)"))

        t0 = time.perf_counter()
        retrieved = self._retrieval.retrieve(active_query, allowed_tags)
        evidence = [item for item in retrieved if item.score >= self._min_score]
        ret_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        top_s = f"{retrieved[0].score:.3f}" if retrieved else "0.000"
        nodes.append("retrieve")
        spans.append(
            TraceSpan(
                "retrieve",
                ret_ms,
                f"{len(retrieved)} fetched, {len(evidence)} passed gate (top={top_s})",
            )
        )

        # Immediately emit retrieval candidates so the UI shows chunks while LLM streams
        yield {
            "type": "retrieval",
            "route_kind": plan.kind,
            "sub_queries": list(plan.sub_queries),
            "contextualized_query": active_query,
            "guardrail_status": guard_status,
            "retrieved": retrieved,
        }

        if not evidence:
            nodes.append("abstain")
            spans.append(TraceSpan("abstain", 0.1, "reason=no_evidence"))
            ans = Answer(text=NOT_FOUND_TEXT, citations=(), abstained=True, reason="no_evidence")
            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            trace = TraceRecord(
                tid,
                active_query,
                plan.kind,
                True,
                "no_evidence",
                total_ms,
                tuple(nodes),
                tuple(spans),
                session_id=session_id,
                guardrail_status=guard_status,
                output=ans.text,
            )
            if self._tracer is not None:
                self._tracer.record(trace)
            yield {
                "type": "done",
                "result": CRAGRunResult(
                    ans,
                    retrieved,
                    plan.kind,
                    list(plan.sub_queries),
                    trace,
                    active_query,
                    guard_status,
                ),
            }
            return

        # 4. Stream Generation Tokens
        t0 = time.perf_counter()
        prompt = build_prompt(active_query, evidence)
        chunks_buf: list[str] = []
        for delta in self._llm.generate_stream(prompt):
            chunks_buf.append(delta)
            yield {"type": "token", "delta": delta}
        raw = format_model_answer("".join(chunks_buf))
        gen_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        nodes.append("generate")
        spans.append(TraceSpan("generate", gen_ms, f"{len(raw)} chars"))

        # 5. Claim Verification
        t0 = time.perf_counter()
        if raw == NOT_FOUND_TOKEN or raw.startswith(NOT_FOUND_TOKEN):
            ans = Answer(text=NOT_FOUND_TEXT, citations=(), abstained=True, reason="llm_abstained")
            v_detail = "abstained=llm_abstained"
        else:
            numbers = {int(n) for n in _CITATION.findall(raw)}
            valid_bounds = bool(numbers) and all(1 <= n <= len(evidence) for n in numbers)
            verified = valid_bounds and (
                self._verifier.verify(raw, evidence, query=active_query)
                if self._verifier is not None
                else True
            )
            if not verified:
                ans = Answer(
                    text=NOT_FOUND_TEXT, citations=(), abstained=True, reason="unsupported_answer"
                )
                v_detail = "abstained=unsupported_answer"
            else:
                citations = tuple(
                    Citation(evidence[n - 1].chunk.id, evidence[n - 1].chunk.doc_id)
                    for n in sorted(numbers)
                )
                ans = Answer(text=raw, citations=citations, abstained=False)
                v_detail = f"verified ({len(citations)} citations)"
        ver_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        nodes.append("verify")
        spans.append(TraceSpan("verify", ver_ms, v_detail))

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        trace = TraceRecord(
            tid,
            active_query,
            plan.kind,
            ans.abstained,
            ans.reason,
            total_ms,
            tuple(nodes),
            tuple(spans),
            session_id=session_id,
            top_score=retrieved[0].score if retrieved else 0.0,
            chunks_passed_gate=len(evidence),
            guardrail_status=guard_status,
            output=ans.text,
            generation=raw,
        )
        if self._tracer is not None:
            self._tracer.record(trace)
        if self._cache is not None and guard_status == "passed":
            self._cache.set(active_query, allowed_tags, ans)
        if self._conversations is not None and session_id:
            self._conversations.append(
                session_id, allowed_tags, ChatTurn(active_query, ans.text, ans.abstained)
            )
        yield {
            "type": "done",
            "result": CRAGRunResult(
                ans,
                retrieved,
                plan.kind,
                list(plan.sub_queries),
                trace,
                active_query,
                guard_status,
            ),
        }

    def run_traced(
        self,
        query: str,
        allowed_tags: frozenset[str],
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> CRAGRunResult:
        """Execute the LangGraph CRAG workflow (with guardrails, chat history, and cache)."""
        history = (
            self._conversations.history(session_id, allowed_tags)
            if self._conversations is not None and session_id
            else []
        )
        cache_query = query
        extra_nodes: list[str] = []
        extra_spans: list[TraceSpan] = []
        if history:
            t0 = time.perf_counter()
            resolved = contextualize_follow_up(query, history)
            cms = round((time.perf_counter() - t0) * 1000.0, 2)
            if resolved != query:
                cache_query = resolved
                extra_nodes.append("contextualize")
                extra_spans.append(TraceSpan("contextualize", cms, resolved[:80]))

        # Lookup the resolved query so identical repeats hit even when the session has history.
        # Pronoun follow-ups rewrite first and therefore miss the prior turn's cache key.
        cached = self._serve_cache_hit(
            cache_query,
            allowed_tags,
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            extra_nodes=tuple(extra_nodes),
            extra_spans=tuple(extra_spans),
        )
        if cached is not None:
            return cached

        res = self._runner.run(
            query,
            allowed_tags,
            trace_id=trace_id,
            history=history,
            session_id=session_id,
            user_id=user_id,
        )
        if self._cache is not None and res.guardrail_status == "passed":
            self._cache.set(res.contextualized_query or query, allowed_tags, res.answer)
        blocked = res.guardrail_status == "guardrail_blocked:prompt_injection"
        if self._conversations is not None and session_id and not blocked:
            self._conversations.append(
                session_id,
                allowed_tags,
                ChatTurn(res.trace.query, res.answer.text, res.answer.abstained),
            )
        return res

    def recent_traces(self, limit: int = 20) -> list[TraceRecord]:
        if self._tracer is None:
            return []
        return self._tracer.recent(limit)

    def answer_with_evidence(
        self, query: str, allowed_tags: frozenset[str]
    ) -> tuple[Answer, list[ScoredChunk]]:
        """Run retrieval once and return both the final Answer and all retrieved candidates."""
        result = self.run_traced(query, allowed_tags)
        return result.answer, result.retrieved

    def answer(self, query: str, allowed_tags: frozenset[str]) -> Answer:
        return self.run_traced(query, allowed_tags).answer

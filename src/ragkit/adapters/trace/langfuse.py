"""In-memory ring-buffer tracer and Langfuse OpenTelemetry (OTLP/HTTP) + Scores exporter."""

from __future__ import annotations

import base64
import json
import secrets
import time
import urllib.request
from collections import deque
from collections.abc import Callable
from typing import Any

from ragkit.ports.tracer import Tracer, TraceRecord, TraceScore

_MAX_TRACE_TEXT = 8000


def _clip_trace_text(text: str) -> str:
    if len(text) <= _MAX_TRACE_TEXT:
        return text
    return text[:_MAX_TRACE_TEXT] + "…"


def _hex_trace_id(raw_id: str) -> str:
    cleaned = "".join(c for c in raw_id.lower() if c in "0123456789abcdef")
    if len(cleaned) >= 32:
        return cleaned[:32]
    return (cleaned + secrets.token_hex(16))[:32]


def _str_attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _int_attr(key: str, value: int) -> dict[str, Any]:
    return {"key": key, "value": {"intValue": str(value)}}


def _obs_type(span_name: str) -> str:
    if span_name == "generate":
        return "generation"
    if span_name == "retrieve":
        return "retriever"
    if span_name in ("verify", "guard_input"):
        return "evaluator"
    return "span"


class InMemoryTracer(Tracer):
    """Keeps a bounded ring buffer of traces and scores in memory for UI and tests."""

    def __init__(self, max_items: int = 100) -> None:
        self._items: deque[TraceRecord] = deque(maxlen=max_items)
        self._scores: deque[TraceScore] = deque(maxlen=max_items)

    def record(self, trace: TraceRecord) -> None:
        self._items.appendleft(trace)

    def recent(self, limit: int = 20) -> list[TraceRecord]:
        if limit <= 0:
            return []
        return list(self._items)[:limit]

    def score(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        self._scores.appendleft(
            TraceScore(trace_id=trace_id, name=name, value=value, comment=comment)
        )

    def recent_scores(self, limit: int = 20) -> list[TraceScore]:
        return list(self._scores)[: max(0, limit)]


class LangfuseHttpTracer(Tracer):
    """Records traces in memory and exports OTLP/HTTP JSON spans + scores to Langfuse."""

    def __init__(
        self,
        base_url: str,
        public_key: str,
        secret_key: str,
        model_name: str = "gemma4:e2b",
        max_items: int = 100,
        timeout: float = 5.0,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._base = base_url.strip().strip('"').strip("'").rstrip("/")
        pk = public_key.strip().strip('"').strip("'")
        sk = secret_key.strip().strip('"').strip("'")
        token = f"{pk}:{sk}".encode()
        self._auth = "Basic " + base64.b64encode(token).decode("ascii")
        self._model = model_name
        self._memory = InMemoryTracer(max_items=max_items)
        self._timeout = timeout
        self._transport = transport or urllib.request.urlopen

    def record(self, trace: TraceRecord) -> None:
        self._memory.record(trace)
        trace_id_hex = _hex_trace_id(trace.trace_id)
        root_span_id = trace_id_hex[:16]
        end_ns = time.time_ns()
        total_ns = max(1_000_000, int(trace.total_ms * 1_000_000))
        start_ns = end_ns - total_ns

        output_json = json.dumps(
            {
                "answer": _clip_trace_text(trace.output),
                "abstained": trace.abstained,
                "reason": trace.reason,
                "nodes": list(trace.nodes),
            }
        )
        root_attrs: list[dict[str, Any]] = [
            _str_attr("langfuse.trace.name", "ragkit.query"),
            _str_attr("langfuse.trace.input", trace.query),
            _str_attr("langfuse.trace.output", output_json),
            _str_attr("langfuse.observation.type", "agent"),
            _str_attr("langfuse.observation.input", trace.query),
            _str_attr("langfuse.observation.output", output_json),
            _str_attr("ragkit.route_kind", trace.route_kind),
            _str_attr("ragkit.guardrail_status", trace.guardrail_status),
            _str_attr("ragkit.top_retrieval_score", f"{trace.top_score:.3f}"),
            _int_attr("ragkit.chunks_passed_gate", trace.chunks_passed_gate),
        ]
        if trace.session_id:
            root_attrs.append(_str_attr("langfuse.session.id", trace.session_id))
        if trace.user_id:
            root_attrs.append(_str_attr("langfuse.user.id", trace.user_id))

        otel_spans: list[dict[str, Any]] = [
            {
                "traceId": trace_id_hex,
                "spanId": root_span_id,
                "name": "ragkit.query",
                "kind": 1,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(end_ns),
                "attributes": root_attrs,
                "status": {"code": 1},
            }
        ]

        cursor_ns = start_ns
        for idx, span in enumerate(trace.spans, start=1):
            dur_ns = max(100_000, int(span.duration_ms * 1_000_000))
            span_end_ns = cursor_ns + dur_ns
            child_span_id = f"{idx:04x}{root_span_id[4:]}"
            obs_type = _obs_type(span.name)
            span_output = (
                _clip_trace_text(trace.generation)
                if span.name == "generate" and trace.generation
                else span.detail
            )
            attrs: list[dict[str, Any]] = [
                _str_attr("langfuse.observation.type", obs_type),
                _str_attr("langfuse.observation.input", trace.query),
                _str_attr("langfuse.observation.output", span_output),
            ]
            if obs_type == "generation":
                prompt_tokens = max(10, len(trace.query.split()) * 18)
                completion_tokens = max(5, len(span_output.split()) * 4)
                attrs.extend(
                    [
                        _str_attr("gen_ai.request.model", self._model),
                        _str_attr("langfuse.observation.model.name", self._model),
                        _int_attr("gen_ai.usage.input_tokens", prompt_tokens),
                        _int_attr("gen_ai.usage.output_tokens", completion_tokens),
                    ]
                )
            otel_spans.append(
                {
                    "traceId": trace_id_hex,
                    "spanId": child_span_id,
                    "parentSpanId": root_span_id,
                    "name": f"langgraph.{span.name}",
                    "kind": 1,
                    "startTimeUnixNano": str(cursor_ns),
                    "endTimeUnixNano": str(span_end_ns),
                    "attributes": attrs,
                    "status": {"code": 1},
                }
            )
            cursor_ns = span_end_ns

        otlp_payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [_str_attr("service.name", "ragkit")],
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "ragkit.langgraph", "version": "0.1.0"},
                            "spans": otel_spans,
                        }
                    ],
                }
            ]
        }
        self._post_json("/api/public/otel/v1/traces", otlp_payload)

    def score(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        self._memory.score(trace_id, name, value, comment)
        trace_id_hex = _hex_trace_id(trace_id)
        payload: dict[str, Any] = {
            "traceId": trace_id_hex,
            "name": name,
            "value": value,
            "dataType": "NUMERIC",
        }
        if comment:
            payload["comment"] = comment
        self._post_json("/api/public/scores", payload)

    def _post_json(self, path: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            self._base + path,
            data=data,
            headers={
                "content-type": "application/json",
                "authorization": self._auth,
                "user-agent": "ragkit/0.1.0",
            },
            method="POST",
        )
        try:
            with self._transport(request, timeout=self._timeout) as response:
                response.read()
        except Exception:
            # Observability export failures must never crash user queries.
            return

    def recent(self, limit: int = 20) -> list[TraceRecord]:
        return self._memory.recent(limit)

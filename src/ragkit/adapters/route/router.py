"""Rule-based and LLM-assisted retrieval routers for single-pass vs multi-hop queries."""

from __future__ import annotations

import re

from ragkit.ports.router import RetrievalRouter, RoutePlan

_MULTI_QUESTION = re.compile(r"\?\s+(?=[A-Z])")
_IMPERATIVE_TAIL = re.compile(
    r"^(?:explain|describe|summarize|elaborate|justify|state|give|provide|tell)\b",
    re.IGNORECASE,
)
_CONJUNCTION_SPLIT = re.compile(
    r"\s+and\s+(?=(?:how|what|who|when|where|which)\b)|\b(?:and also|as well as|versus|vs\.?)\b",
    re.IGNORECASE,
)
_COMPARE_PREFIX = re.compile(
    r"^\s*(?:compare|what is the difference between)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)\??\s*$",
    re.IGNORECASE,
)


class PassthroughRouter(RetrievalRouter):
    """Always routes to a single retrieval pass."""

    def route(self, query: str) -> RoutePlan:
        return RoutePlan(kind="single", sub_queries=(query,))


class RuleBasedRetrievalRouter(RetrievalRouter):
    """Routes direct lookups to `single` and decomposes compound/comparison queries to `multi_hop`.

    Avoids paying an LLM planner call on the 80%+ of queries that are a single lookup.
    """

    def __init__(self, max_sub_queries: int = 3) -> None:
        self._max_sub_queries = max_sub_queries

    def route(self, query: str) -> RoutePlan:
        cleaned = query.strip()
        if not cleaned:
            return RoutePlan(kind="single", sub_queries=(query,))

        compare_match = _COMPARE_PREFIX.match(cleaned)
        if compare_match:
            left = compare_match.group(1).strip()
            right = compare_match.group(2).strip()
            if left and right:
                return RoutePlan(kind="multi_hop", sub_queries=(left, right))

        parts = [p.strip() for p in _MULTI_QUESTION.split(cleaned) if p.strip()]
        if len(parts) > 1 and not any(_IMPERATIVE_TAIL.match(p) for p in parts[1:]):
            return RoutePlan(
                kind="multi_hop",
                sub_queries=tuple(parts[: self._max_sub_queries]),
            )

        conj_parts = [p.strip(" ,;?") for p in _CONJUNCTION_SPLIT.split(cleaned) if p.strip(" ,;?")]
        if len(conj_parts) > 1 and all(len(p.split()) >= 2 for p in conj_parts):
            return RoutePlan(
                kind="multi_hop",
                sub_queries=tuple(conj_parts[: self._max_sub_queries]),
            )

        return RoutePlan(kind="single", sub_queries=(cleaned,))

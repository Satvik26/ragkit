from __future__ import annotations

from collections.abc import Sequence

from ragkit.domain.models import Chunk, ScoredChunk
from ragkit.ports.catalog import DocumentCatalog
from ragkit.ports.embedder import Embedder
from ragkit.ports.lexical_store import LexicalSearcher
from ragkit.ports.query_rewriter import QueryRewriter
from ragkit.ports.reranker import Reranker
from ragkit.ports.router import RetrievalRouter, RoutePlan
from ragkit.ports.vector_store import VectorSearcher

_RRF_K = 60


def fuse_rrf(
    dense_hits: Sequence[ScoredChunk],
    lexical_hits: Sequence[ScoredChunk],
    rrf_k: int = _RRF_K,
    lexical_weight: float = 0.10,
) -> list[ScoredChunk]:
    """Score-calibrated hybrid fusion over dense and BM25 candidates.

    Preserves the dense cosine margin and adds a normalized BM25 bonus
    (`lexical_weight * (bm25 / max_bm25)`) plus a tiny RRF tie-breaker so
    high-confidence dense hits are never overturned by weak keyword overlaps,
    while exact-term matches break close semantic ties and promote missed chunks.
    """
    if not lexical_hits:
        return list(dense_hits)
    if not dense_hits:
        peak = max((item.score / (item.score + 1.0) for item in lexical_hits), default=0.0)
        top_raw = lexical_hits[0].score
        return [
            ScoredChunk(
                item.chunk,
                peak * (item.score / top_raw) if top_raw > 0 else 0.0,
            )
            for item in lexical_hits
        ]

    chunks_by_id: dict[str, Chunk] = {}
    dense_scores: dict[str, float] = {}
    lexical_scores: dict[str, float] = {}
    rrf_scores: dict[str, float] = {}

    for rank, item in enumerate(dense_hits, start=1):
        cid = item.chunk.id
        chunks_by_id[cid] = item.chunk
        dense_scores[cid] = item.score
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    max_lex = max((item.score for item in lexical_hits), default=1.0)
    for rank, item in enumerate(lexical_hits, start=1):
        cid = item.chunk.id
        chunks_by_id[cid] = item.chunk
        lexical_scores[cid] = item.score / max_lex if max_lex > 0 else 0.0
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    dense_floor = min((item.score for item in dense_hits), default=0.45)
    combined: dict[str, float] = {}
    for cid in chunks_by_id:
        base = dense_scores.get(cid, dense_floor)
        lex_bonus = lexical_weight * lexical_scores.get(cid, 0.0)
        combined[cid] = base + lex_bonus + 1e-3 * rrf_scores[cid]

    ordered_ids = sorted(combined, key=lambda cid: combined[cid], reverse=True)
    return [ScoredChunk(chunks_by_id[cid], combined[cid]) for cid in ordered_ids]


def expand_parent_chunks(
    ranked: Sequence[ScoredChunk],
    catalog: DocumentCatalog,
    allowed_tags: frozenset[str],
    top_k: int,
    parent_docs_limit: int = 2,
) -> list[ScoredChunk]:
    """Expand multi-chunk parent documents so sibling chunks of top hits are kept together."""
    if not ranked or top_k <= 0:
        return []
    expanded: list[ScoredChunk] = []
    seen_ids: set[str] = set()
    expanded_docs = 0

    for item in ranked:
        if len(expanded) >= top_k:
            break
        if item.chunk.id not in seen_ids:
            expanded.append(item)
            seen_ids.add(item.chunk.id)
        if expanded_docs < parent_docs_limit:
            siblings = catalog.chunks_for(item.chunk.doc_id)
            if len(siblings) > 1:
                expanded_docs += 1
                for idx, sib in enumerate(siblings, start=1):
                    if len(expanded) >= top_k:
                        break
                    if sib.id in seen_ids or not (sib.acl_tags & allowed_tags):
                        continue
                    sib_chunk = Chunk(
                        id=sib.id,
                        doc_id=sib.doc_id,
                        text=sib.text,
                        acl_tags=sib.acl_tags,
                        metadata={
                            "start": str(sib.start),
                            "end": str(sib.end),
                            "content_hash": sib.text_hash,
                        },
                    )
                    expanded.append(ScoredChunk(sib_chunk, max(0.0, item.score - 1e-4 * idx)))
                    seen_ids.add(sib.id)
    return expanded[:top_k]


def interleave_multi_hop(
    sub_results: Sequence[Sequence[ScoredChunk]],
    top_k: int,
) -> list[ScoredChunk]:
    """Round-robin interleave ranked chunks from each sub-query so every hop gets evidence."""
    merged: list[ScoredChunk] = []
    seen: set[str] = set()
    max_len = max((len(r) for r in sub_results), default=0)
    for rank_idx in range(max_len):
        for sub_list in sub_results:
            if rank_idx >= len(sub_list):
                continue
            item = sub_list[rank_idx]
            if item.chunk.id not in seen:
                seen.add(item.chunk.id)
                merged.append(item)
                if len(merged) >= top_k:
                    return merged
    return merged


class RetrievalService:
    """Depends only on ports (dependency inversion); concrete adapters are injected."""

    def __init__(
        self,
        embedder: Embedder,
        searcher: VectorSearcher,
        reranker: Reranker,
        candidates: int = 100,
        top_k: int = 8,
        lexical_searcher: LexicalSearcher | None = None,
        query_rewriter: QueryRewriter | None = None,
        catalog: DocumentCatalog | None = None,
        router: RetrievalRouter | None = None,
    ) -> None:
        self._embedder = embedder
        self._searcher = searcher
        self._reranker = reranker
        self._candidates = candidates
        self._top_k = top_k
        self._lexical_searcher = lexical_searcher
        self._query_rewriter = query_rewriter
        self._catalog = catalog
        self._router = router

    def plan_route(self, query: str) -> RoutePlan:
        if self._router is None:
            return RoutePlan(kind="single", sub_queries=(query,))
        return self._router.route(query)

    def _retrieve_single(self, query: str, allowed_tags: frozenset[str]) -> list[ScoredChunk]:
        lexical_query = (
            self._query_rewriter.rewrite(query) if self._query_rewriter is not None else query
        )
        [query_vector] = self._embedder.embed([query])
        dense_candidates = self._searcher.search(query_vector, self._candidates, allowed_tags)
        if self._lexical_searcher is None:
            candidates = dense_candidates
        else:
            lexical_candidates = self._lexical_searcher.search(
                lexical_query, self._candidates, allowed_tags
            )
            candidates = fuse_rrf(dense_candidates, lexical_candidates)[: self._candidates]
        reranked = self._reranker.rerank(query, candidates)[: self._top_k]
        if self._catalog is not None:
            return expand_parent_chunks(reranked, self._catalog, allowed_tags, self._top_k)
        return reranked

    def retrieve(self, query: str, allowed_tags: frozenset[str]) -> list[ScoredChunk]:
        plan = self.plan_route(query)
        if plan.kind == "multi_hop" and len(plan.sub_queries) > 1:
            sub_results = [self._retrieve_single(sub_q, allowed_tags) for sub_q in plan.sub_queries]
            return interleave_multi_hop(sub_results, self._top_k)
        return self._retrieve_single(query, allowed_tags)

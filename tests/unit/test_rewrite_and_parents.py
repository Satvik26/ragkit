from __future__ import annotations

from ragkit.adapters.catalog.memory import InMemoryDocumentCatalog
from ragkit.adapters.chunk.fixed_size import FixedSizeChunker
from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.rewrite.rewriter import (
    LLMQueryRewriter,
    PassthroughQueryRewriter,
    TechnicalQueryRewriter,
)
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.domain.models import Chunk, ScoredChunk
from ragkit.services.prompts import NOT_FOUND_TOKEN
from ragkit.services.retrieval import RetrievalService, expand_parent_chunks
from tests.fakes import KeywordEmbedder, ScriptedLLM


def test_passthrough_rewriter_leaves_query_unchanged() -> None:
    rewriter = PassthroughQueryRewriter()
    assert rewriter.rewrite("How long do refunds take?") == "How long do refunds take?"


def test_technical_rewriter_expands_hyphens_and_glossary() -> None:
    rewriter = TechnicalQueryRewriter(extra_expansions={"il-6": "interleukin 6 cytokine"})
    expanded = rewriter.rewrite("Signaling of IL-6 in T-cells.")
    assert expanded.startswith("Signaling of IL-6 in T-cells.")
    assert "IL 6" in expanded
    assert "T cells" in expanded
    assert "interleukin 6 cytokine" in expanded
    assert rewriter.rewrite("   ") == "   "
    assert rewriter.rewrite("plain query") == "plain query"


def test_llm_query_rewriter_appends_keywords_and_handles_not_found() -> None:
    rewriter = LLMQueryRewriter(ScriptedLLM("refund window fourteen days"))
    assert rewriter.rewrite("refunds?") == "refunds? (refund window fourteen days)"

    abstained = LLMQueryRewriter(ScriptedLLM(NOT_FOUND_TOKEN))
    assert abstained.rewrite("refunds?") == "refunds?"
    assert abstained.rewrite("   ") == "   "


def test_parent_child_expansion_pulls_in_sibling_chunks() -> None:
    tags = frozenset({"eng"})
    catalog = InMemoryDocumentCatalog()
    chunker = FixedSizeChunker(size=4, overlap=1)
    text = "w0 w1 w2 w3 w4 w5 w6 w7 w8"
    chunks = chunker.chunk("long-doc.md", text, tags)
    assert len(chunks) == 3
    catalog.save("long-doc.md", text, tags, chunks)

    other = Chunk(
        "other-1",
        "other.md",
        "unrelated text",
        tags,
        {"start": "0", "end": "2", "content_hash": "h"},
    )
    ranked = [ScoredChunk(chunks[0], 0.90), ScoredChunk(other, 0.50)]
    expanded = expand_parent_chunks(ranked, catalog, tags, top_k=4)
    assert [item.chunk.id for item in expanded] == [
        chunks[0].id,
        chunks[1].id,
        chunks[2].id,
        "other-1",
    ]
    assert expand_parent_chunks([], catalog, tags, top_k=4) == []


def test_retrieval_service_uses_rewriter_and_expands_parents() -> None:
    tags = frozenset({"eng"})
    catalog = InMemoryDocumentCatalog()
    c1 = Chunk(
        "c1",
        "doc-p",
        "t-cell receptor signaling",
        tags,
        {"start": "0", "end": "3", "content_hash": "h1"},
    )
    c2 = Chunk(
        "c2",
        "doc-p",
        "sibling window text",
        tags,
        {"start": "3", "end": "6", "content_hash": "h2"},
    )
    catalog.save("doc-p", "t-cell receptor signaling sibling window text", tags, [c1, c2])

    store = InMemoryVectorStore()
    store.upsert([c1], [[1.0, 0.0]])
    rewriter = TechnicalQueryRewriter()

    from ragkit.adapters.lexical.bm25 import InMemoryBM25Store

    lexical = InMemoryBM25Store()
    lexical.upsert([c1])

    service = RetrievalService(
        KeywordEmbedder({"t-cell signaling": [1.0, 0.0]}),
        store,
        PassthroughReranker(),
        top_k=4,
        lexical_searcher=lexical,
        query_rewriter=rewriter,
        catalog=catalog,
    )
    hits = service.retrieve("t-cell signaling", tags)
    assert [h.chunk.id for h in hits] == ["c1", "c2"]

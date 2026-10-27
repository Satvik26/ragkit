from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.adapters.dev.stubs import ExtractiveLLM
from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.config.settings import Settings
from ragkit.domain.models import Chunk
from ragkit.evals.run import (
    GoldenQuestion,
    evaluate,
    format_report,
    load_questions,
    main,
)
from ragkit.services.answering import AnswerService
from ragkit.services.retrieval import RetrievalService
from tests.fakes import KeywordEmbedder


def test_unanswerable_questions_stay_out_of_the_average() -> None:
    questions = [
        GoldenQuestion("a", "known", frozenset({"hit"}), True),
        GoldenQuestion("b", "unknown", frozenset(), False),
    ]
    report = evaluate(
        questions,
        lambda question: ["hit"] if question == "known" else ["other"],
        8,
    )
    assert report.questions == 2
    assert report.scored == 1
    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0


def test_recall_stops_at_k_and_mrr_uses_the_rank() -> None:
    questions = [GoldenQuestion("a", "q", frozenset({"late"}), True)]
    report = evaluate(questions, lambda _question: ["x", "late"], 1)
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.5


def test_load_questions_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "golden_set.jsonl"
    path.write_text(
        '{"id":"q1","question":"How long?","relevant_chunk_ids":["c1"],"answerable":true}\n'
        '{"id":"q2","question":"Who won?","relevant_chunk_ids":[],"answerable":false}\n',
        encoding="utf-8",
    )
    loaded = load_questions(path)
    assert loaded[0].relevant_chunk_ids == frozenset({"c1"})
    assert loaded[1].answerable is False


def test_format_report_names_k() -> None:
    questions = [GoldenQuestion("a", "q", frozenset({"hit"}), True)]
    text = format_report(evaluate(questions, lambda _question: ["hit"], 8))
    assert "recall@8 1.000" in text
    assert "mrr 1.000" in text


def test_ranked_chunk_ids_follow_retrieval_order() -> None:
    store = InMemoryVectorStore()
    tags = frozenset({"eng"})
    store.upsert(
        [Chunk("a", "d", "t", tags), Chunk("b", "d", "t", tags)],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    service = AnswerService(
        RetrievalService(KeywordEmbedder({"q": [1.0, 0.0]}), store, PassthroughReranker()),
        ExtractiveLLM(),
    )
    assert service.ranked_chunk_ids("q", tags) == ["a", "b"]


def test_main_refuses_the_in_memory_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "golden_set.jsonl"
    path.write_text(
        '{"id":"q1","question":"How long?","relevant_chunk_ids":["c1"],"answerable":true}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ragkit.evals.run.Settings",
        lambda: Settings(qdrant_url=None, embedding_url=None),
    )
    with pytest.raises(SystemExit, match="RAG_QDRANT_URL"):
        main(["--path", str(path)])

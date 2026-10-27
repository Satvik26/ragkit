from __future__ import annotations

import pytest

from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.domain.models import Chunk
from ragkit.services.answering import NOT_FOUND_TEXT, AnswerService
from ragkit.services.retrieval import RetrievalService
from tests.fakes import KeywordEmbedder, ScriptedLLM

ENG = frozenset({"eng"})


def build(reply: str, min_score: float = 0.5) -> tuple[AnswerService, ScriptedLLM]:
    store = InMemoryVectorStore()
    store.upsert(
        [
            Chunk("c1", "policy.pdf", "Refunds take 14 days.", frozenset({"eng"})),
            Chunk("c2", "hr.pdf", "Salary bands are confidential.", frozenset({"hr"})),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    table = {"refund?": [1.0, 0.0], "salary?": [0.0, 1.0], "weather?": [0.1, 0.9]}
    llm = ScriptedLLM(reply)
    retrieval = RetrievalService(KeywordEmbedder(table), store, PassthroughReranker())
    return AnswerService(retrieval, llm, min_score), llm


def test_grounded_answer_has_citations() -> None:
    service, _ = build("Refunds take 14 days [1].")
    result = service.answer("refund?", ENG)
    assert not result.abstained
    assert [c.chunk_id for c in result.citations] == ["c1"]
    assert result.citations[0].doc_id == "policy.pdf"


def test_prompt_contains_only_permitted_evidence() -> None:
    service, llm = build("Refunds take 14 days [1].")
    service.answer("refund?", ENG)
    assert "Refunds take 14 days." in llm.prompts[0]
    assert "Salary" not in llm.prompts[0]


def test_acl_blocks_restricted_document() -> None:
    service, llm = build("anything [1]")
    result = service.answer("salary?", ENG)
    assert result.abstained and result.reason == "no_evidence"
    assert llm.prompts == []  # the LLM is never called without permitted evidence


def test_low_similarity_triggers_abstention() -> None:
    service, llm = build("made up [1]", min_score=0.5)
    result = service.answer("weather?", ENG)  # nearest permitted chunk scores about 0.11
    assert result.abstained and result.text == NOT_FOUND_TEXT
    assert llm.prompts == []


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("Refunds take 14 days.", "unsupported_answer"),  # no citation
        ("Refunds take 14 days [7].", "unsupported_answer"),  # citation to nothing
        ("NOT_FOUND", "llm_abstained"),
    ],
)
def test_bad_llm_output_abstains(reply: str, reason: str) -> None:
    service, _ = build(reply)
    result = service.answer("refund?", ENG)
    assert result.abstained and result.reason == reason

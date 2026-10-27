from __future__ import annotations

from ragkit.adapters.rerank.passthrough import PassthroughReranker
from ragkit.adapters.route.router import PassthroughRouter, RuleBasedRetrievalRouter
from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.adapters.verify.verifier import (
    GroundingClaimVerifier,
    LLMClaimVerifier,
    PassthroughVerifier,
)
from ragkit.domain.models import Chunk, ScoredChunk
from ragkit.services.answering import AnswerService
from ragkit.services.retrieval import RetrievalService
from tests.fakes import KeywordEmbedder, ScriptedLLM

ENG = frozenset({"eng"})


def _evidence(text: str = "Refunds take fourteen days.") -> list[ScoredChunk]:
    return [ScoredChunk(Chunk("c1", "refunds.md", text, ENG), 0.90)]


def test_grounding_verifier_accepts_supported_claim_and_word_number_match() -> None:
    verifier = GroundingClaimVerifier()
    assert verifier.verify("Refunds take 14 days [1].", _evidence()) is True
    assert verifier.verify("Refunds take fourteen days [1].", _evidence()) is True


def test_grounding_verifier_rejects_hallucinated_numbers() -> None:
    verifier = GroundingClaimVerifier()
    assert verifier.verify("Refunds take 48 hours [1].", _evidence()) is False
    assert verifier.verify("Refunds take 14 days with a 5% fee [1].", _evidence()) is False
    assert (
        verifier.verify(
            "Folic acid dosage was 0.8 mg [1].",
            _evidence("Folic acid dosage was 0.8 mg."),
            query="What dosage of vitamin B12 was given?",
        )
        is False
    )


def test_grounding_verifier_rejects_unrelated_claim_with_fake_citation() -> None:
    verifier = GroundingClaimVerifier()
    assert verifier.verify("The Eiffel Tower stands in Paris [1].", _evidence()) is False
    assert verifier.verify("No citations here.", _evidence()) is False
    assert verifier.verify("Out of bounds [9].", _evidence()) is False


def test_passthrough_and_llm_claim_verifier() -> None:
    ev = _evidence()
    assert PassthroughVerifier().verify("Anything [1]", ev) is True

    supported = LLMClaimVerifier(ScriptedLLM("SUPPORTED"))
    assert supported.verify("Refunds take 14 days [1].", ev) is True

    unsupported = LLMClaimVerifier(ScriptedLLM("UNSUPPORTED"))
    assert unsupported.verify("Refunds take 14 days [1].", ev) is False

    # Fails grounding check before calling LLM
    llm = ScriptedLLM("SUPPORTED")
    assert LLMClaimVerifier(llm).verify("Refunds take 99 days [1].", ev) is False
    assert llm.prompts == []


def test_answer_service_abstains_when_verifier_rejects_hallucinated_number() -> None:
    store = InMemoryVectorStore()
    store.upsert([Chunk("c1", "refunds.md", "Refunds take fourteen days.", ENG)], [[1.0, 0.0]])
    retrieval = RetrievalService(
        KeywordEmbedder({"refund?": [1.0, 0.0]}), store, PassthroughReranker()
    )
    service = AnswerService(
        retrieval,
        ScriptedLLM("Refunds are paid in 24 hours [1]."),
        min_evidence_score=0.5,
        verifier=GroundingClaimVerifier(),
    )
    result = service.answer("refund?", ENG)
    assert result.abstained is True
    assert result.reason == "unsupported_answer"


def test_rule_based_router_classifies_single_and_multi_hop() -> None:
    router = RuleBasedRetrievalRouter()
    assert PassthroughRouter().route("q").kind == "single"
    assert router.route("   ").kind == "single"

    single = router.route("How long do refunds take?")
    assert single.kind == "single"
    assert single.sub_queries == ("How long do refunds take?",)

    compare = router.route("Compare severity one response and severity two response?")
    assert compare.kind == "multi_hop"
    assert compare.sub_queries == ("severity one response", "severity two response")

    two_q = router.route("How long do refunds take? What is the postmortem deadline?")
    assert two_q.kind == "multi_hop"
    assert len(two_q.sub_queries) == 2

    conj = router.route("How long do refunds take and what is the postmortem deadline?")
    assert conj.kind == "multi_hop"
    assert len(conj.sub_queries) == 2


def test_multi_hop_retrieval_interleaves_evidence_from_each_sub_query() -> None:
    store = InMemoryVectorStore()
    c_refund = Chunk("c-ref", "refunds.md", "Refunds take fourteen days.", ENG)
    c_post = Chunk("c-post", "postmortems.md", "Postmortems are due within five days.", ENG)
    store.upsert([c_refund, c_post], [[1.0, 0.0], [0.0, 1.0]])

    embedder = KeywordEmbedder(
        {
            "How long do refunds take": [1.0, 0.0],
            "what is the postmortem deadline": [0.0, 1.0],
        }
    )
    service = RetrievalService(
        embedder,
        store,
        PassthroughReranker(),
        top_k=2,
        router=RuleBasedRetrievalRouter(),
    )
    hits = service.retrieve("How long do refunds take and what is the postmortem deadline?", ENG)
    assert [h.chunk.id for h in hits] == ["c-ref", "c-post"]

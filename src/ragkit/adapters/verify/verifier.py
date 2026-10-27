"""Claim verifiers: passthrough, deterministic lexical/numeric grounding, and LLM judge."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragkit.adapters.lexical.bm25 import tokenize
from ragkit.domain.models import ScoredChunk
from ragkit.ports.llm import LLMClient
from ragkit.ports.verifier import ClaimVerifier

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER_TOKEN = re.compile(r"\b\d+(?:\.\d+)?\b")
# Alphanumeric technical/biomedical identifiers like B12, PGE2, V600E, G12D, 3DFC, H1N1
_ALNUM_ID = re.compile(r"\b(?:[a-z]+\d+[a-z0-9]*|\d+[a-z]+[a-z0-9]*)\b")

_WORD_TO_NUM: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "ninety": "90",
    "hundred": "100",
    "five hundred": "500",
    "thousand": "1000",
    "ten thousand": "10000",
}


def _extract_numbers(text: str) -> set[str]:
    cleaned = _CITATION.sub("", text).lower()
    # Normalize comma-separated thousands (e.g. 4,000 -> 4000, 314,019 -> 314019)
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", cleaned)
    nums = set(_NUMBER_TOKEN.findall(cleaned))
    for word, digit in _WORD_TO_NUM.items():
        if re.search(rf"\b{re.escape(word)}\b", cleaned):
            nums.add(digit)
    return nums


class PassthroughVerifier(ClaimVerifier):
    """Accepts any syntactically cited answer."""

    def verify(self, answer_text: str, evidence: Sequence[ScoredChunk], query: str = "") -> bool:
        del answer_text, evidence, query
        return True


class GroundingClaimVerifier(ClaimVerifier):
    """Checks that each cited claim sentence shares content tokens and numbers with its chunks.

    1. Every claim sentence with a citation `[n]` is checked against the union of chunks
       cited in that sentence (or all cited chunks in the answer if `[n]` is at the end).
    2. Any number in the claim (e.g. `48` or `fourteen` -> `14`) must appear in the cited
       evidence text. This catches hallucinated figures, dates, and dosages immediately.
    3. Any alphanumeric identifier in the query (e.g. `B12`, `PGE2`, `V600E`) must appear
       in the cited evidence text so the LLM cannot substitute a different entity (`folic acid`
       for `vitamin B12`).
    4. At least one content token from the claim must appear in the cited evidence text.
    """

    def verify(self, answer_text: str, evidence: Sequence[ScoredChunk], query: str = "") -> bool:
        all_indices = {int(n) for n in _CITATION.findall(answer_text)}
        if not all_indices or any(n < 1 or n > len(evidence) for n in all_indices):
            return False

        all_cited_text = " ".join(evidence[n - 1].chunk.text for n in sorted(all_indices)).lower()
        query_ids = set(_ALNUM_ID.findall(query.lower()))
        if query_ids and not query_ids.issubset(set(_ALNUM_ID.findall(all_cited_text))):
            return False

        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer_text.strip()) if s.strip()]
        if not sentences:
            return False

        for sentence in sentences:
            local_indices = {int(n) for n in _CITATION.findall(sentence)}
            target_indices = local_indices if local_indices else all_indices
            if any(n < 1 or n > len(evidence) for n in target_indices):
                return False

            cited_text = " ".join(evidence[n - 1].chunk.text for n in sorted(target_indices))
            claim_without_cites = _CITATION.sub("", sentence).strip()
            if not claim_without_cites:
                continue

            claim_numbers = _extract_numbers(claim_without_cites)
            evidence_numbers = _extract_numbers(cited_text)
            if not claim_numbers.issubset(evidence_numbers):
                return False

            claim_tokens = set(tokenize(claim_without_cites))
            evidence_tokens = set(tokenize(cited_text))
            for word, digit in _WORD_TO_NUM.items():
                if digit in evidence_numbers:
                    evidence_tokens.update(tokenize(word))
                    evidence_tokens.add(digit)
            if claim_tokens and not (claim_tokens & evidence_tokens):
                return False

        return True


class LLMClaimVerifier(ClaimVerifier):
    """Runs GroundingClaimVerifier first, then asks an LLMClient entailment judge."""

    def __init__(self, llm: LLMClient) -> None:
        self._grounding = GroundingClaimVerifier()
        self._llm = llm

    def verify(self, answer_text: str, evidence: Sequence[ScoredChunk], query: str = "") -> bool:
        if not self._grounding.verify(answer_text, evidence, query=query):
            return False
        indices = sorted({int(n) for n in _CITATION.findall(answer_text)})
        passages = "\n".join(f"[{n}] {evidence[n - 1].chunk.text}" for n in indices)
        prompt = (
            "Does the Evidence below fully support the Answer without any unsupported claims?\n"
            "Reply with ONLY one word: SUPPORTED or UNSUPPORTED.\n\n"
            f"Evidence:\n{passages}\n\n"
            f"Answer:\n{answer_text}"
        )
        verdict = self._llm.generate(prompt).strip().upper()
        return verdict.startswith("SUPPORTED")

"""Score the golden set with the same retrieval the API uses.

Unanswerable rows have no relevant chunk, so they stay out of recall and MRR.
Recall is measured on the ranked list retrieval returns, before the evidence gate.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragkit.config.container import build_services
from ragkit.config.settings import Settings
from ragkit.evals.metrics import recall_at_k, reciprocal_rank

Retrieve = Callable[[str], Sequence[str]]


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    id: str
    question: str
    relevant_chunk_ids: frozenset[str]
    answerable: bool


@dataclass(frozen=True, slots=True)
class EvalReport:
    questions: int
    scored: int
    recall_at_k: float
    mrr: float
    k: int


def load_questions(path: Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        questions.append(
            GoldenQuestion(
                id=str(raw["id"]),
                question=str(raw["question"]),
                relevant_chunk_ids=frozenset(str(item) for item in raw["relevant_chunk_ids"]),
                answerable=bool(raw["answerable"]),
            )
        )
    return questions


def evaluate(questions: Sequence[GoldenQuestion], retrieve: Retrieve, k: int) -> EvalReport:
    """Average recall@k and MRR over answerable questions that name at least one chunk."""
    scored = [item for item in questions if item.answerable and item.relevant_chunk_ids]
    if not scored:
        return EvalReport(len(questions), 0, 0.0, 0.0, k)
    recalls: list[float] = []
    ranks: list[float] = []
    for item in scored:
        retrieved = list(retrieve(item.question))
        relevant = set(item.relevant_chunk_ids)
        recalls.append(recall_at_k(retrieved, relevant, k))
        ranks.append(reciprocal_rank(retrieved, relevant))
    count = len(scored)
    return EvalReport(len(questions), count, sum(recalls) / count, sum(ranks) / count, k)


def format_report(report: EvalReport) -> str:
    return (
        f"questions {report.questions}\n"
        f"scored {report.scored}\n"
        f"recall@{report.k} {report.recall_at_k:.3f}\n"
        f"mrr {report.mrr:.3f}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print recall@k and MRR for the golden set.")
    parser.add_argument("--path", type=Path, default=Path("evals/golden_set.jsonl"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--tags", default="eng")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.k < 1:
        raise SystemExit("k must be at least 1")
    if not args.path.is_file():
        raise SystemExit(f"golden set not found: {args.path}")
    settings = Settings()
    if not settings.qdrant_url or not settings.embedding_url:
        raise SystemExit(
            "Set RAG_QDRANT_URL and RAG_EMBEDDING_URL. Empty values use the in-memory stub."
        )
    tags = frozenset(part.strip() for part in args.tags.split(",") if part.strip())
    if not tags:
        raise SystemExit("at least one tag is required")
    answer, _ingestion, _ready = build_services(settings)
    report = evaluate(
        load_questions(args.path),
        lambda question: answer.ranked_chunk_ids(question, tags),
        args.k,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()

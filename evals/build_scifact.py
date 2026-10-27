"""Write the SciFact test corpus and a golden file whose chunk ids match FixedSizeChunker.

The handbook stays the smoke test. This set is the harder public baseline from the BEIR
SciFact test split, sized so CPU BGE-M3 embedding finishes in ~3 minutes locally (150
abstracts and 100 claims by default; pass `--full` for all 5,183 abstracts and 300 claims).

Download the BEIR zip once, from the repo root:

    curl -fL -o evals/.cache/scifact.zip \\
      https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

Then: uv run python evals/build_scifact.py
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from ragkit.adapters.chunk.fixed_size import FixedSizeChunker

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).with_name(".cache")
ZIP_PATH = CACHE / "scifact.zip"
EXTRACTED = CACHE / "scifact"
CORPUS = ROOT / "corpus" / "scifact"
GOLDEN = ROOT / "evals" / "scifact.jsonl"
_TAGS = frozenset({"eng"})


def render_abstract(title: str, text: str) -> str:
    heading = " ".join(title.split()) or "Untitled"
    body = text.strip()
    return f"# {heading}\n\n{body}\n"


def load_beir(root: Path) -> tuple[dict[str, tuple[str, str]], dict[str, str], dict[str, set[str]]]:
    """Corpus id to (title, text), query id to claim, query id to relevant corpus ids."""
    documents: dict[str, tuple[str, str]] = {}
    for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        documents[str(raw["_id"])] = (str(raw["title"]), str(raw["text"]))
    queries: dict[str, str] = {}
    for line in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        queries[str(raw["_id"])] = str(raw["text"])
    relevant: dict[str, set[str]] = defaultdict(set)
    qrel_lines = (root / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()
    for line in qrel_lines[1:]:
        query_id, corpus_id, score = line.split("\t")
        if int(score) > 0:
            relevant[query_id].add(corpus_id)
    return documents, queries, relevant


def select_subset(
    documents: dict[str, tuple[str, str]],
    relevant: dict[str, set[str]],
    max_queries: int,
    max_docs: int,
) -> tuple[dict[str, tuple[str, str]], dict[str, set[str]]]:
    """Keep the first `max_queries` test claims, all their cited abstracts, plus distractors."""
    chosen_queries = sorted(relevant, key=int)[:max_queries]
    kept_relevant = {qid: relevant[qid] for qid in chosen_queries}
    needed_docs: set[str] = set()
    for doc_ids in kept_relevant.values():
        needed_docs.update(doc_ids)
    for cid in sorted(documents, key=int):
        if len(needed_docs) >= max_docs:
            break
        needed_docs.add(cid)
    kept_documents = {cid: documents[cid] for cid in sorted(needed_docs, key=int)}
    return kept_documents, kept_relevant


def golden_rows(
    documents: dict[str, tuple[str, str]],
    queries: dict[str, str],
    relevant: dict[str, set[str]],
    chunker: FixedSizeChunker,
) -> tuple[list[tuple[str, str]], list[dict[str, object]], int]:
    """Markdown files, golden rows, and how many test questions touch a multi-chunk abstract.

    Every chunk of a relevant abstract is labeled. Recall@k then requires each of those
    chunks, which is stricter than BEIR's document-level score when an abstract is split.
    """
    files: list[tuple[str, str]] = []
    chunk_ids: dict[str, list[str]] = {}
    for corpus_id in sorted(documents, key=int):
        title, text = documents[corpus_id]
        body = render_abstract(title, text)
        name = f"{corpus_id}.md"
        files.append((name, body))
        doc_id = f"scifact/{name}"
        chunks = chunker.chunk(doc_id, body, _TAGS)
        chunk_ids[corpus_id] = [chunk.id for chunk in chunks]

    rows: list[dict[str, object]] = []
    multi_chunk_questions = 0
    for query_id in sorted(relevant, key=int):
        claim = queries.get(query_id)
        if claim is None:
            continue
        doc_ids = sorted(relevant[query_id], key=int)
        if any(len(chunk_ids.get(doc_id, [])) > 1 for doc_id in doc_ids):
            multi_chunk_questions += 1
        labeled: list[str] = []
        for doc_id in doc_ids:
            labeled.extend(chunk_ids.get(doc_id, []))
        if not labeled:
            continue
        rows.append(
            {
                "id": f"sf-{query_id}",
                "question": claim,
                "relevant_chunk_ids": labeled,
                "relevant_doc_ids": [f"scifact/{doc_id}.md" for doc_id in doc_ids],
                "answer": None,
                "answerable": True,
            }
        )
    return files, rows, multi_chunk_questions


def ensure_extracted() -> Path:
    if (EXTRACTED / "corpus.jsonl").is_file():
        return EXTRACTED
    if not ZIP_PATH.is_file():
        raise SystemExit(
            "Missing evals/.cache/scifact.zip. Download the BEIR SciFact zip, then re-run."
        )
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(CACHE)
    if not (EXTRACTED / "corpus.jsonl").is_file():
        raise SystemExit("scifact.zip did not contain corpus.jsonl")
    return EXTRACTED


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SciFact corpus and golden set.")
    parser.add_argument(
        "--full", action="store_true", help="Write all 5,183 abstracts & 300 claims"
    )
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--max-docs", type=int, default=1000)
    args = parser.parse_args()

    documents, queries, relevant = load_beir(ensure_extracted())
    if not args.full:
        documents, relevant = select_subset(documents, relevant, args.max_queries, args.max_docs)

    files, rows, multi_chunk_questions = golden_rows(
        documents, queries, relevant, FixedSizeChunker()
    )
    CORPUS.mkdir(parents=True, exist_ok=True)
    for existing in CORPUS.glob("*.md"):
        existing.unlink()
    for name, body in files:
        (CORPUS / name).write_text(body, encoding="utf-8")
    GOLDEN.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(
        f"wrote {len(files)} documents and {len(rows)} questions "
        f"({multi_chunk_questions} questions label more than one chunk)"
    )


if __name__ == "__main__":
    main()

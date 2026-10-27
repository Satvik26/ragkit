"""Write the local handbook and a golden set whose chunk ids match FixedSizeChunker.

Run from the repo root: uv run python evals/build_local_set.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ragkit.adapters.chunk.fixed_size import FixedSizeChunker

ROOT = Path(__file__).resolve().parents[1]
FACTS = Path(__file__).with_name("handbook_facts.json")
CORPUS = ROOT / "corpus" / "handbook"
GOLDEN = ROOT / "evals" / "golden_set.jsonl"


def main() -> None:
    spec = json.loads(FACTS.read_text(encoding="utf-8"))
    CORPUS.mkdir(parents=True, exist_ok=True)
    chunker = FixedSizeChunker()
    rows: list[dict[str, object]] = []
    number = 1
    for item in spec["documents"]:
        text = (
            f"# {item['title']}\n\n{item['fact']}\n\n"
            "This rule applies to all Lumen Field Services employees.\n"
        )
        (CORPUS / item["file"]).write_text(text, encoding="utf-8")
        doc_id = f"handbook/{item['file']}"
        chunks = chunker.chunk(doc_id, text, frozenset({"eng"}))
        if len(chunks) != 1:
            raise SystemExit(f"{doc_id} produced {len(chunks)} chunks")
        for question in (item["question"], item["paraphrase"]):
            rows.append(
                {
                    "id": f"q{number:03d}",
                    "question": question,
                    "relevant_chunk_ids": [chunks[0].id],
                    "answer": item["fact"],
                    "answerable": True,
                }
            )
            number += 1
    for question in spec["unanswerable"]:
        rows.append(
            {
                "id": f"q{number:03d}",
                "question": question,
                "relevant_chunk_ids": [],
                "answer": None,
                "answerable": False,
            }
        )
        number += 1
    GOLDEN.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(spec['documents'])} documents and {len(rows)} questions")


if __name__ == "__main__":
    main()

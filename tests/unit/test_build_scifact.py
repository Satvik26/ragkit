from __future__ import annotations

from evals.build_scifact import golden_rows, render_abstract
from ragkit.adapters.chunk.fixed_size import FixedSizeChunker


def test_render_abstract_collapses_the_title() -> None:
    assert render_abstract("A  title\n", "Body.") == "# A title\n\nBody.\n"


def test_golden_rows_label_every_chunk_of_a_relevant_abstract() -> None:
    long_body = " ".join(f"w{i}" for i in range(600))
    documents = {
        "1": ("Short", "Only a few words."),
        "2": ("Long", long_body),
    }
    queries = {"10": "Which abstract supports this claim?"}
    relevant = {"10": {"1", "2"}}
    _files, rows, multi = golden_rows(documents, queries, relevant, FixedSizeChunker())
    assert multi == 1
    assert rows[0]["id"] == "sf-10"
    assert rows[0]["answerable"] is True
    labeled = rows[0]["relevant_chunk_ids"]
    assert isinstance(labeled, list)
    assert len(labeled) >= 2
    chunker = FixedSizeChunker()
    short = chunker.chunk(
        "scifact/1.md",
        render_abstract("Short", "Only a few words."),
        frozenset({"eng"}),
    )
    assert short[0].id in labeled

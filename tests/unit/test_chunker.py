from __future__ import annotations

import pytest

from ragkit.adapters.chunk.fixed_size import FixedSizeChunker

ENG = frozenset({"eng"})


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_empty_text_returns_no_chunks() -> None:
    assert FixedSizeChunker(size=5, overlap=1).chunk("doc", "  \n  ", ENG) == []


def test_short_document_is_one_chunk() -> None:
    chunker = FixedSizeChunker(size=500, overlap=100)
    chunks = chunker.chunk("doc", "Refunds take fourteen days.", ENG)
    assert len(chunks) == 1
    assert chunks[0].text == "Refunds take fourteen days."
    assert chunks[0].metadata["start"] == "0"
    assert chunks[0].metadata["end"] == "4"


def test_windows_overlap_and_stay_in_order() -> None:
    chunker = FixedSizeChunker(size=5, overlap=2)
    chunks = chunker.chunk("doc", words(10), ENG)
    assert [c.metadata["start"] for c in chunks] == ["0", "3", "6"]
    assert [c.metadata["end"] for c in chunks] == ["5", "8", "10"]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[1].text.split()[-2:] == chunks[2].text.split()[:2]


def test_same_input_returns_the_same_ids() -> None:
    chunker = FixedSizeChunker(size=4, overlap=1)
    text = words(9)
    assert [c.id for c in chunker.chunk("doc", text, ENG)] == [
        c.id for c in chunker.chunk("doc", text, ENG)
    ]


def test_one_word_edit_changes_only_the_touched_chunk() -> None:
    chunker = FixedSizeChunker(size=5, overlap=1)
    original = chunker.chunk("doc", words(12), ENG)
    edited = words(12).replace("w6", "changed")
    updated = chunker.chunk("doc", edited, ENG)
    changed = [
        index
        for index, (old, new) in enumerate(zip(original, updated, strict=True))
        if old.id != new.id
    ]
    assert changed == [1]
    assert updated[1].text.split()[2] == "changed"


def test_same_text_in_two_documents_does_not_share_an_id() -> None:
    chunker = FixedSizeChunker(size=5, overlap=1)
    left = chunker.chunk("doc-a", "same words here", ENG)[0]
    right = chunker.chunk("doc-b", "same words here", ENG)[0]
    assert left.id != right.id
    assert left.metadata["content_hash"] == right.metadata["content_hash"]


def test_acl_tags_are_copied_onto_every_chunk() -> None:
    tags = frozenset({"eng", "hr"})
    chunks = FixedSizeChunker(size=3, overlap=1).chunk("doc", words(7), tags)
    assert chunks
    assert all(c.acl_tags == tags for c in chunks)
    hidden = FixedSizeChunker(size=3, overlap=0).chunk("doc", "secret text", frozenset())
    assert hidden[0].acl_tags == frozenset()


def test_metadata_values_are_strings() -> None:
    chunk = FixedSizeChunker(size=5, overlap=1).chunk("doc", words(3), ENG)[0]
    assert set(chunk.metadata) == {"start", "end", "content_hash"}
    assert all(isinstance(value, str) for value in chunk.metadata.values())


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (5, 5), (5, 6), (4, -1)],
)
def test_window_that_cannot_advance_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(size=size, overlap=overlap)

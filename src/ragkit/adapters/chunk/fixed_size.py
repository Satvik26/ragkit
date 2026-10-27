"""Word windows for the embedder. A tokenizer arrives with the real embedder."""

from __future__ import annotations

import hashlib

from ragkit.domain.models import Chunk
from ragkit.ports.chunker import Chunker


def _digest(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        raw = part.encode()
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
    return hasher.hexdigest()


class FixedSizeChunker(Chunker):
    """About `size` words per chunk, each window overlapping the previous by `overlap` words.

    The id hashes doc id plus chunk text, not the window number, so a re-run overwrites
    the same points. `start` and `end` are word indexes; `end` is exclusive.
    """

    def __init__(self, size: int = 500, overlap: int = 100) -> None:
        if size < 1:
            raise ValueError("size must be at least 1")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be >= 0 and smaller than size")
        self._size = size
        self._overlap = overlap

    def chunk(self, doc_id: str, text: str, acl_tags: frozenset[str]) -> list[Chunk]:
        words = text.split()
        if not words:
            return []
        step = self._size - self._overlap
        chunks: list[Chunk] = []
        start = 0
        while start < len(words):
            end = min(start + self._size, len(words))
            window = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    id=_digest(doc_id, window),
                    doc_id=doc_id,
                    text=window,
                    acl_tags=acl_tags,
                    metadata={
                        "start": str(start),
                        "end": str(end),
                        "content_hash": _digest(window),
                    },
                )
            )
            if end == len(words):
                break
            start += step
        return chunks

"""Pure domain entities. No framework or I/O imports allowed here."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    # A chunk is visible to a caller only if the sets intersect. Empty means visible to nobody.
    acl_tags: frozenset[str] = frozenset()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float  # higher is better, for every adapter


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    doc_id: str


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    abstained: bool
    reason: str | None = None

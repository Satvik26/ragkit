"""Records of an ingested document. No database imports."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from ragkit.domain.models import Chunk


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredDocument:
    doc_id: str
    content_hash: str
    acl_tags: frozenset[str]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class StoredChunk:
    id: str
    doc_id: str
    text: str
    acl_tags: frozenset[str]
    start: int
    end: int
    text_hash: str


def chunk_record(chunk: Chunk) -> StoredChunk:
    try:
        start = int(chunk.metadata["start"])
        end = int(chunk.metadata["end"])
    except KeyError as exc:
        raise ValueError("chunk metadata must include start and end") from exc
    return StoredChunk(
        id=chunk.id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        acl_tags=frozenset(chunk.acl_tags),
        start=start,
        end=end,
        text_hash=content_hash(chunk.text),
    )


def document_record(
    doc_id: str,
    text: str,
    acl_tags: frozenset[str],
    created_at: datetime,
    updated_at: datetime,
) -> StoredDocument:
    return StoredDocument(
        doc_id=doc_id,
        content_hash=content_hash(text),
        acl_tags=frozenset(acl_tags),
        created_at=created_at,
        updated_at=updated_at,
    )


Clock = Callable[[], datetime]

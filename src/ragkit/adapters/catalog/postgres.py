"""Postgres catalog. One connection per call, closed before the method returns."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import psycopg

from ragkit.domain.catalog import StoredChunk, StoredDocument, chunk_record, content_hash
from ragkit.domain.models import Chunk
from ragkit.ports.catalog import DocumentCatalog

_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    acl_tags TEXT[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents (doc_id),
    text TEXT NOT NULL,
    acl_tags TEXT[] NOT NULL,
    start_word INTEGER NOT NULL,
    end_word INTEGER NOT NULL,
    text_hash TEXT NOT NULL
)
"""

_CHUNKS_BY_DOC = "CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id)"

_UPSERT_DOCUMENT = """
INSERT INTO documents (doc_id, content_hash, acl_tags)
VALUES (%s, %s, %s)
ON CONFLICT (doc_id) DO UPDATE
SET content_hash = EXCLUDED.content_hash,
    acl_tags = EXCLUDED.acl_tags,
    updated_at = now()
"""

_UPSERT_CHUNK = """
INSERT INTO chunks (id, doc_id, text, acl_tags, start_word, end_word, text_hash)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE
SET doc_id = EXCLUDED.doc_id,
    text = EXCLUDED.text,
    acl_tags = EXCLUDED.acl_tags,
    start_word = EXCLUDED.start_word,
    end_word = EXCLUDED.end_word,
    text_hash = EXCLUDED.text_hash
"""


class PostgresDocumentCatalog(DocumentCatalog):
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._schema_ready = False

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._database_url, connect_timeout=2)

    def ready(self) -> bool:
        """A readiness probe. Any failure means the process should not take traffic."""
        try:
            conn = self._connect()
        except Exception:
            return False
        try:
            conn.execute("SELECT 1")
        except Exception:
            return False
        finally:
            conn.close()
        return True

    def save(
        self, doc_id: str, text: str, acl_tags: frozenset[str], chunks: Sequence[Chunk]
    ) -> None:
        records = [chunk_record(chunk) for chunk in chunks]
        conn = self._connect()
        try:
            self._ensure_schema(conn)
            conn.execute(
                _UPSERT_DOCUMENT,
                (doc_id, content_hash(text), sorted(acl_tags)),
            )
            for record in records:
                conn.execute(
                    _UPSERT_CHUNK,
                    (
                        record.id,
                        record.doc_id,
                        record.text,
                        sorted(record.acl_tags),
                        record.start,
                        record.end,
                        record.text_hash,
                    ),
                )
            conn.commit()
            self._schema_ready = True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def document(self, doc_id: str) -> StoredDocument | None:
        conn = self._connect()
        try:
            self._ensure_schema(conn)
            row = conn.execute(
                """
                SELECT doc_id, content_hash, acl_tags, created_at, updated_at
                FROM documents WHERE doc_id = %s
                """,
                (doc_id,),
            ).fetchone()
            conn.commit()
            self._schema_ready = True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if row is None:
            return None
        return StoredDocument(
            doc_id=str(row[0]),
            content_hash=str(row[1]),
            acl_tags=frozenset(str(tag) for tag in row[2]),
            created_at=_as_datetime(row[3]),
            updated_at=_as_datetime(row[4]),
        )

    def chunks_for(self, doc_id: str) -> list[StoredChunk]:
        conn = self._connect()
        try:
            self._ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT id, doc_id, text, acl_tags, start_word, end_word, text_hash
                FROM chunks WHERE doc_id = %s
                ORDER BY start_word
                """,
                (doc_id,),
            ).fetchall()
            conn.commit()
            self._schema_ready = True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return [
            StoredChunk(
                id=str(row[0]),
                doc_id=str(row[1]),
                text=str(row[2]),
                acl_tags=frozenset(str(tag) for tag in row[3]),
                start=int(row[4]),
                end=int(row[5]),
                text_hash=str(row[6]),
            )
            for row in rows
        ]

    def _ensure_schema(self, conn: psycopg.Connection) -> None:
        if self._schema_ready:
            return
        conn.execute(_DOCUMENTS)
        conn.execute(_CHUNKS)
        conn.execute(_CHUNKS_BY_DOC)


def _as_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("timestamp column did not return a datetime")
    return value

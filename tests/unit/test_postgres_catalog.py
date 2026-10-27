from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ragkit.adapters.catalog.postgres import PostgresDocumentCatalog
from ragkit.domain.models import Chunk

ENG = frozenset({"eng"})
WHEN = datetime(2026, 1, 1, tzinfo=UTC)


class FakeConn:
    def __init__(self, row: tuple | None = None, rows: list[tuple] | None = None) -> None:
        self.statements: list[str] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self._row = row
        self._rows = rows or []

    def execute(self, sql: str, params: object = None) -> FakeConn:
        self.statements.append(sql)
        return self

    def fetchone(self) -> tuple | None:
        return self._row

    def fetchall(self) -> list[tuple]:
        return self._rows

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


class BoomConn(FakeConn):
    def execute(self, sql: str, params: object = None) -> FakeConn:
        if "INSERT INTO documents" in sql:
            raise RuntimeError("postgres down")
        return super().execute(sql, params)


def _catalog(conn: FakeConn) -> PostgresDocumentCatalog:
    catalog = PostgresDocumentCatalog("postgresql://unused")

    def connect() -> FakeConn:
        return conn

    catalog._connect = connect  # type: ignore[method-assign]
    return catalog


def _chunk() -> Chunk:
    return Chunk("c1", "policy", "Refunds take fourteen days.", ENG, {"start": "0", "end": "4"})


def test_save_upserts_document_and_chunk() -> None:
    conn = FakeConn()
    catalog = _catalog(conn)
    catalog.save("policy", "Refunds take fourteen days.", ENG, [_chunk()])
    joined = "\n".join(conn.statements)
    assert "CREATE TABLE IF NOT EXISTS documents" in joined
    assert "INSERT INTO documents" in joined
    assert "INSERT INTO chunks" in joined
    assert conn.committed and conn.closed
    catalog.save("policy", "Refunds take fourteen days.", ENG, [_chunk()])
    assert sum("CREATE TABLE IF NOT EXISTS documents" in sql for sql in conn.statements) == 1


def test_save_rolls_back_when_the_insert_fails() -> None:
    conn = BoomConn()
    catalog = _catalog(conn)
    with pytest.raises(RuntimeError, match="postgres down"):
        catalog.save("policy", "Refunds take fourteen days.", ENG, [_chunk()])
    assert conn.rolled_back and conn.closed
    assert catalog._schema_ready is False


def test_ready_is_false_when_connect_fails() -> None:
    catalog = PostgresDocumentCatalog("postgresql://unused")

    def connect() -> FakeConn:
        raise ConnectionError("down")

    catalog._connect = connect  # type: ignore[method-assign]
    assert catalog.ready() is False


def test_ready_is_true_when_the_probe_succeeds() -> None:
    conn = FakeConn()
    catalog = _catalog(conn)
    assert catalog.ready() is True
    assert conn.closed


def test_ready_is_false_when_the_probe_fails() -> None:
    class Probe(FakeConn):
        def execute(self, sql: str, params: object = None) -> FakeConn:
            raise ConnectionError("down")

    conn = Probe()
    catalog = _catalog(conn)
    assert catalog.ready() is False
    assert conn.closed


def test_reads_map_rows() -> None:
    row = ("policy", "abc", ["eng"], WHEN, WHEN)
    chunk_row = ("c1", "policy", "Refunds take fourteen days.", ["eng"], 0, 4, "def")
    catalog = _catalog(FakeConn(row=row, rows=[chunk_row]))
    stored = catalog.document("policy")
    assert stored is not None
    assert stored.doc_id == "policy" and stored.acl_tags == ENG
    assert stored.created_at == WHEN
    chunks = catalog.chunks_for("policy")
    assert [(chunk.id, chunk.start, chunk.end) for chunk in chunks] == [("c1", 0, 4)]


def test_bad_timestamp_is_rejected() -> None:
    catalog = _catalog(FakeConn(row=("policy", "abc", ["eng"], "nope", WHEN)))
    with pytest.raises(TypeError):
        catalog.document("policy")


def test_missing_document_returns_none() -> None:
    catalog = _catalog(FakeConn())
    assert catalog.document("policy") is None

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ragkit.adapters.catalog.memory import InMemoryDocumentCatalog
from ragkit.domain.catalog import chunk_record
from ragkit.domain.models import Chunk

ENG = frozenset({"eng"})
T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = T0 + timedelta(seconds=5)


def _chunk() -> Chunk:
    return Chunk(
        "c1",
        "policy",
        "Refunds take fourteen days.",
        ENG,
        {"start": "0", "end": "4"},
    )


def test_save_keeps_created_at_and_moves_updated_at() -> None:
    times = iter((T0, T1))
    catalog = InMemoryDocumentCatalog(clock=lambda: next(times))
    catalog.save("policy", "Refunds take fourteen days.", ENG, [_chunk()])
    catalog.save("policy", "Refunds take fourteen days.", ENG, [_chunk()])
    stored = catalog.document("policy")
    assert stored is not None
    assert stored.created_at == T0
    assert stored.updated_at == T1
    assert len(catalog.chunks_for("policy")) == 1


def test_chunk_without_window_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="start and end"):
        chunk_record(Chunk("c1", "policy", "text", ENG))

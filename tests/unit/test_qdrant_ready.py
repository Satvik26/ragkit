from __future__ import annotations

from ragkit.adapters.vector.qdrant_store import qdrant_is_ready


class _Up:
    def get_collections(self) -> list[str]:
        return []


class _Down:
    def get_collections(self) -> list[str]:
        raise ConnectionError("qdrant down")


def test_ready_when_collections_can_be_listed() -> None:
    assert qdrant_is_ready(_Up()) is True


def test_not_ready_when_the_call_fails() -> None:
    assert qdrant_is_ready(_Down()) is False

"""Every VectorStore adapter must pass this suite (Liskov substitution made executable).
Subclass it, provide a `store` fixture yielding an object that is both writer and searcher."""

from __future__ import annotations

from typing import Any

from ragkit.domain.models import Chunk

ALL = frozenset({"eng", "hr"})


def chunk(cid: str, tags: set[str], text: str = "t") -> Chunk:
    return Chunk(id=cid, doc_id=f"doc-{cid}", text=text, acl_tags=frozenset(tags))


class VectorStoreContract:
    def _load(self, store: Any) -> None:
        store.upsert(
            [chunk("a", {"eng"}), chunk("b", {"hr"}), chunk("c", {"eng"})],
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        )

    def test_returns_nearest_first(self, store: Any) -> None:
        self._load(store)
        ids = [s.chunk.id for s in store.search([1.0, 0.0], 3, ALL)]
        assert ids == ["a", "c", "b"]

    def test_scores_are_higher_is_better(self, store: Any) -> None:
        self._load(store)
        scores = [s.score for s in store.search([1.0, 0.0], 3, ALL)]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_results(self, store: Any) -> None:
        self._load(store)
        assert len(store.search([1.0, 0.0], 2, ALL)) == 2

    def test_acl_filter_hides_other_groups(self, store: Any) -> None:
        self._load(store)
        ids = {s.chunk.id for s in store.search([1.0, 0.0], 10, frozenset({"eng"}))}
        assert ids == {"a", "c"}

    def test_no_tags_returns_nothing(self, store: Any) -> None:
        self._load(store)
        assert store.search([1.0, 0.0], 10, frozenset()) == []

    def test_untagged_chunk_is_invisible(self, store: Any) -> None:
        store.upsert([chunk("x", set())], [[1.0, 0.0]])
        assert store.search([1.0, 0.0], 10, ALL) == []

    def test_upsert_is_idempotent_and_overwrites(self, store: Any) -> None:
        store.upsert([chunk("a", {"eng"}, "old")], [[1.0, 0.0]])
        store.upsert([chunk("a", {"eng"}, "new")], [[1.0, 0.0]])
        hits = store.search([1.0, 0.0], 10, ALL)
        assert [h.chunk.text for h in hits] == ["new"]

    def test_payload_round_trips(self, store: Any) -> None:
        original = Chunk("m", "doc-9", "hello", frozenset({"eng"}), {"page": "3"})
        store.upsert([original], [[1.0, 0.0]])
        [hit] = store.search([1.0, 0.0], 1, ALL)
        assert hit.chunk == original

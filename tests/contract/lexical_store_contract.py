"""Every LexicalStore adapter must pass this suite (Liskov substitution made executable).
Subclass it, provide a `store` fixture yielding an object that is both writer and searcher."""

from __future__ import annotations

from typing import Any

from ragkit.domain.models import Chunk

ALL = frozenset({"eng", "hr"})


def chunk(cid: str, tags: set[str], text: str) -> Chunk:
    return Chunk(id=cid, doc_id=f"doc-{cid}", text=text, acl_tags=frozenset(tags))


class LexicalStoreContract:
    def _load(self, store: Any) -> None:
        store.upsert(
            [
                chunk("a", {"eng"}, "foxp3 expression inhibits il6 signaling in t cells"),
                chunk("b", {"hr"}, "parental leave policy grants sixteen paid weeks"),
                chunk("c", {"eng"}, "foxp3 transcription factor regulates foxp3 target genes"),
            ]
        )

    def test_returns_best_lexical_match_first(self, store: Any) -> None:
        self._load(store)
        ids = [s.chunk.id for s in store.search("foxp3 target genes", 3, ALL)]
        assert ids == ["c", "a"]

    def test_scores_are_higher_is_better(self, store: Any) -> None:
        self._load(store)
        scores = [s.score for s in store.search("foxp3", 3, ALL)]
        assert len(scores) == 2
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_results(self, store: Any) -> None:
        self._load(store)
        assert len(store.search("foxp3", 1, ALL)) == 1

    def test_acl_filter_hides_other_groups(self, store: Any) -> None:
        self._load(store)
        ids = {s.chunk.id for s in store.search("parental leave foxp3", 10, frozenset({"eng"}))}
        assert ids == {"a", "c"}

    def test_no_tags_returns_nothing(self, store: Any) -> None:
        self._load(store)
        assert store.search("foxp3", 10, frozenset()) == []

    def test_untagged_chunk_is_invisible(self, store: Any) -> None:
        store.upsert([chunk("x", set(), "secret biomarker foxp3")])
        assert store.search("foxp3", 10, ALL) == []

    def test_upsert_is_idempotent_and_overwrites(self, store: Any) -> None:
        store.upsert([chunk("a", {"eng"}, "old refund rule")])
        store.upsert([chunk("a", {"eng"}, "new postmortem deadline")])
        assert store.search("refund", 10, ALL) == []
        hits = store.search("postmortem", 10, ALL)
        assert [h.chunk.text for h in hits] == ["new postmortem deadline"]

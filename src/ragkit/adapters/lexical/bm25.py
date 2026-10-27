"""Okapi BM25 lexical search adapters (in-memory and Postgres-backed)."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import psycopg

from ragkit.domain.models import Chunk, ScoredChunk
from ragkit.ports.lexical_store import LexicalSearcher, LexicalWriter

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    filtered = [token for token in tokens if token not in _STOPWORDS]
    return filtered if filtered else tokens


def score_bm25(
    query: str,
    chunks: Iterable[Chunk],
    k: int,
    allowed_tags: frozenset[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[ScoredChunk]:
    """Okapi BM25 over visible chunks only (ACL filter applied before statistics and scoring)."""
    if k <= 0 or not allowed_tags:
        return []
    query_terms = tokenize(query)
    if not query_terms:
        return []

    visible = [chunk for chunk in chunks if chunk.acl_tags & allowed_tags]
    if not visible:
        return []

    tokenized_docs: list[tuple[Chunk, Counter[str], int]] = []
    doc_freq: Counter[str] = Counter()
    total_len = 0
    unique_query_terms = set(query_terms)

    for chunk in visible:
        tokens = tokenize(chunk.text)
        counts = Counter(tokens)
        length = len(tokens)
        total_len += length
        tokenized_docs.append((chunk, counts, length))
        for term in unique_query_terms:
            if counts.get(term, 0) > 0:
                doc_freq[term] += 1

    n_docs = len(tokenized_docs)
    avgdl = total_len / n_docs if total_len > 0 else 1.0

    scored: list[ScoredChunk] = []
    for chunk, counts, length in tokenized_docs:
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq[term]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (length / avgdl))
            score += idf * ((tf * (k1 + 1.0)) / denom)
        if score > 0.0:
            scored.append(ScoredChunk(chunk, score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:k]


class InMemoryBM25Store(LexicalWriter, LexicalSearcher):
    """Reference BM25 implementation: used in unit tests and the dev profile."""

    def __init__(self) -> None:
        self._items: dict[str, Chunk] = {}

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._items[chunk.id] = chunk

    def search(self, query: str, k: int, allowed_tags: frozenset[str]) -> list[ScoredChunk]:
        return score_bm25(query, self._items.values(), k, allowed_tags)


_SELECT_VISIBLE_CHUNKS = """
SELECT id, doc_id, text, acl_tags, start_word, end_word, text_hash
FROM chunks
WHERE acl_tags && %s::text[]
"""


class PostgresBM25Store(LexicalWriter, LexicalSearcher):
    """BM25 over the durable Postgres `chunks` table, filtered by ACL inside SQL."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._database_url, connect_timeout=2)

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        # DocumentCatalog.save already writes the durable `chunks` rows in Postgres.
        del chunks

    def search(self, query: str, k: int, allowed_tags: frozenset[str]) -> list[ScoredChunk]:
        if k <= 0 or not allowed_tags or not tokenize(query):
            return []
        conn = self._connect()
        try:
            rows = conn.execute(_SELECT_VISIBLE_CHUNKS, (sorted(allowed_tags),)).fetchall()
        except psycopg.errors.UndefinedTable:
            conn.rollback()
            return []
        finally:
            conn.close()
        visible = [
            Chunk(
                id=str(row[0]),
                doc_id=str(row[1]),
                text=str(row[2]),
                acl_tags=frozenset(str(tag) for tag in row[3]),
                metadata={
                    "start": str(row[4]),
                    "end": str(row[5]),
                    "content_hash": str(row[6]),
                },
            )
            for row in rows
        ]
        return score_bm25(query, visible, k, allowed_tags)

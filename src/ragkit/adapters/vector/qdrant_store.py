from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from ragkit.domain.models import Chunk, ScoredChunk
from ragkit.ports.vector_store import VectorSearcher, VectorWriter


def _point_id(chunk_id: str) -> str:
    # Qdrant ids must be UUIDs or ints; derive a stable UUID so upserts are idempotent.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class _QdrantProbe(Protocol):
    def get_collections(self) -> object: ...


def qdrant_is_ready(client: _QdrantProbe) -> bool:
    """A readiness probe. Any failure means the process should not take traffic."""
    try:
        client.get_collections()
    except Exception:
        return False
    return True


class QdrantVectorStore(VectorWriter, VectorSearcher):
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim
        self._collection_ready = False

    def _ensure_collection(self) -> None:
        # Created on first use so a down Qdrant does not stop the process from starting.
        if self._collection_ready:
            return
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )
        self._collection_ready = True

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._ensure_collection()
        points = [
            qm.PointStruct(
                id=_point_id(c.id),
                vector=list(v),
                payload={
                    "chunk_id": c.id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "acl_tags": sorted(c.acl_tags),
                    "metadata": c.metadata,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self, vector: Sequence[float], k: int, allowed_tags: frozenset[str]
    ) -> list[ScoredChunk]:
        if k <= 0 or not allowed_tags:
            return []
        self._ensure_collection()
        acl = qm.Filter(
            must=[qm.FieldCondition(key="acl_tags", match=qm.MatchAny(any=sorted(allowed_tags)))]
        )
        hits = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=k,
            query_filter=acl,
            with_payload=True,
        ).points
        results: list[ScoredChunk] = []
        for hit in hits:
            p = hit.payload or {}
            chunk = Chunk(
                id=str(p["chunk_id"]),
                doc_id=str(p["doc_id"]),
                text=str(p["text"]),
                acl_tags=frozenset(p["acl_tags"]),
                metadata=dict(p.get("metadata", {})),
            )
            results.append(ScoredChunk(chunk, float(hit.score)))
        return results

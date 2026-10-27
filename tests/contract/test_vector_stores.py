from __future__ import annotations

from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient

from ragkit.adapters.vector.in_memory import InMemoryVectorStore
from ragkit.adapters.vector.qdrant_store import QdrantVectorStore
from tests.contract.vector_store_contract import VectorStoreContract


class TestInMemoryStore(VectorStoreContract):
    @pytest.fixture
    def store(self) -> InMemoryVectorStore:
        return InMemoryVectorStore()


class TestQdrantStore(VectorStoreContract):
    @pytest.fixture
    def store(self) -> Iterator[QdrantVectorStore]:
        client = QdrantClient(":memory:")  # real Qdrant engine, no container needed
        yield QdrantVectorStore(client, "test", dim=2)
        client.close()


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore().upsert([], [[1.0]])

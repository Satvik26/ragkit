from __future__ import annotations

import pytest

from ragkit.adapters.lexical.bm25 import InMemoryBM25Store
from tests.contract.lexical_store_contract import LexicalStoreContract


class TestInMemoryBM25Store(LexicalStoreContract):
    @pytest.fixture
    def store(self) -> InMemoryBM25Store:
        return InMemoryBM25Store()

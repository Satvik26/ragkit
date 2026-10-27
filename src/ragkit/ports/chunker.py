from __future__ import annotations

from abc import ABC, abstractmethod

from ragkit.domain.models import Chunk


class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc_id: str, text: str, acl_tags: frozenset[str]) -> list[Chunk]:
        """Split one document into ordered chunks. Same text and doc id yield the same ids."""

"""Retrieval router port: classifies a query and decomposes multi-hop queries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

RouteKind = Literal["single", "multi_hop"]


@dataclass(frozen=True, slots=True)
class RoutePlan:
    kind: RouteKind
    sub_queries: tuple[str, ...]


class RetrievalRouter(ABC):
    @abstractmethod
    def route(self, query: str) -> RoutePlan:
        """Choose single-pass vs multi-hop retrieval and return sub-queries to execute."""

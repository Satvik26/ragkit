from __future__ import annotations

from collections.abc import Sequence

from ragkit.ports.embedder import Embedder
from ragkit.ports.llm import LLMClient


class KeywordEmbedder(Embedder):
    """Maps exact texts to fixed 2-d vectors so tests control similarity precisely."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._table[t] for t in texts]


class ScriptedLLM(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply

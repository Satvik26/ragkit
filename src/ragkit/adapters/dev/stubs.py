"""Deterministic stand-ins so the API runs locally with zero external services.
Not for production: replace with real embedding and LLM adapters."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from ragkit.ports.embedder import Embedder
from ragkit.ports.llm import LLMClient


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                idx = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16) % self._dim
                vec[idx] += 1.0
            out.append(vec)
        return out


class ExtractiveLLM(LLMClient):
    """Echoes the first evidence passage with its citation. Exercises the pipeline only."""

    def generate(self, prompt: str) -> str:
        match = re.search(r"^\[1\] (.+)$", prompt, flags=re.MULTILINE)
        return f"{match.group(1)} [1]" if match else "NOT_FOUND"

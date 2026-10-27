from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yield response chunks as they arrive. Default splits `generate()` by word."""
        text = self.generate(prompt)
        words = text.split(" ")
        for idx, word in enumerate(words):
            yield word if idx == 0 else " " + word

"""Factory for the LLM strategy. Callers depend on LLMClient, not on a vendor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ragkit.adapters.dev.stubs import ExtractiveLLM
from ragkit.adapters.llm.ollama import OllamaClient
from ragkit.adapters.llm.openai_chat import OpenAIChatClient
from ragkit.config.settings import Settings
from ragkit.ports.llm import LLMClient


@dataclass(frozen=True, slots=True)
class LLMBinding:
    """The selected strategy, the model name written into traces, and an optional health check."""

    client: LLMClient
    model_name: str
    ready: Callable[[], bool] | None = None


def build_llm(settings: Settings) -> LLMBinding:
    """Select an LLMClient from RAG_LLM_PROVIDER.

    `ollama` without RAG_OLLAMA_URL uses the local extractive stub so unit tests need no model.
    `openai` fails at startup when the API key is missing.
    """
    if settings.llm_provider == "openai":
        key = settings.openai_api_key.strip()
        if not key:
            raise ValueError("RAG_OPENAI_API_KEY is required when RAG_LLM_PROVIDER=openai")
        client = OpenAIChatClient(key, settings.openai_model, settings.openai_base_url)
        return LLMBinding(client, settings.openai_model, client.ready)
    if settings.ollama_url:
        ollama = OllamaClient(settings.ollama_url, settings.ollama_model)
        return LLMBinding(ollama, settings.ollama_model, ollama.ready)
    return LLMBinding(ExtractiveLLM(), settings.ollama_model, None)

"""Provider registry: infer provider from model name and resolve backend class."""

from __future__ import annotations

import re
from typing import Any, Optional, Type

from llm_blanket.base import BaseLLM

# Model prefix -> provider (first match wins if we iterate in order)
MODEL_PREFIX_TO_PROVIDER: list[tuple[str, str]] = [
    (r"^gpt-", "openai"),
    (r"^o1-", "openai"),
    (r"^o3-", "openai"),
    (r"^claude-", "anthropic"),
    (r"^gemini-", "gemini"),
    (r"^grok-", "xai"),
    (r"^grok\b", "xai"),  # "grok" or "grok-2" etc.
    # Groq: no single prefix; they have llama-*, mixtral-*, etc. So we use explicit provider.
]

# Known Groq model name prefixes (Groq-specific)
GROQ_MODEL_PREFIXES = ("llama-", "mixtral-", "whisper-")

# Backend class registry: provider -> class
_backends: dict[str, Type[BaseLLM]] = {}


def register_backend(provider: str, backend_class: Type[BaseLLM]) -> None:
    _backends[provider] = backend_class


def get_backend_class(provider: str) -> Optional[Type[BaseLLM]]:
    return _backends.get(provider)


def infer_provider(model: str, explicit_provider: Optional[str] = None) -> str:
    """
    Infer provider from model name, or use explicit_provider if given.
    For Groq models (e.g. llama-3-70b-8192), we cannot infer uniquely; use explicit provider=groq.
    """
    if explicit_provider is not None:
        return explicit_provider.strip().lower()
    model_lower = (model or "").strip().lower()
    for pattern, provider in MODEL_PREFIX_TO_PROVIDER:
        if re.match(pattern, model_lower):
            return provider
    # Fallback: could be custom or OpenAI-compatible
    return "openai"


def is_likely_groq_model(model: str) -> bool:
    return any(model.lower().startswith(p) for p in GROQ_MODEL_PREFIXES)

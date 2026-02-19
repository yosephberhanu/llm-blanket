"""Configuration for LLM clients: API keys, base URLs, and overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

# Standard env var names (LangChain / AutoGen style)
DEFAULT_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",  # also used by Gemini in many tools
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "custom": "OPENAI_API_KEY",  # custom endpoints often reuse OpenAI key name
}

# Default base URLs per provider (OpenAI-compatible where applicable)
DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
    "anthropic": "https://api.anthropic.com",  # SDK uses its own base
    "gemini": "https://generativelanguage.googleapis.com",  # SDK uses its own
    "custom": "https://api.openai.com/v1",  # placeholder; user must set
}


@dataclass
class LLMConfig:
    """
    Configuration for LLM clients.

    - API keys: pass explicitly or rely on env (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY).
    - base_url: override for the current model's provider (single override).
    - base_urls: map provider or model name -> base URL for overrides (e.g. for custom endpoints).
    """

    api_key: Optional[str] = None
    """Explicit API key. If None, resolved from provider's env var."""

    base_url: Optional[str] = None
    """Override base URL for this client (takes precedence over base_urls)."""

    base_urls: dict[str, str] = field(default_factory=dict)
    """
    Map of provider name or model name -> base URL.
    E.g. {"openai": "https://my-proxy.com/v1", "gpt-4": "https://custom.com/v1"}.
    Used when base_url is not set; exact model match wins, then provider match.
    """

    provider: Optional[str] = None
    """Force provider (openai, anthropic, gemini, groq, xai, custom). If None, inferred from model."""

    # Optional provider-specific options (extensible)
    extra: dict[str, Any] = field(default_factory=dict)

    def get_api_key(self, provider: str) -> Optional[str]:
        """Resolve API key: explicit first, then env for that provider."""
        if self.api_key is not None:
            return self.api_key
        env_key = DEFAULT_ENV_KEYS.get(provider, "OPENAI_API_KEY")
        return os.environ.get(env_key)

    def get_base_url(self, provider: str, model: str) -> Optional[str]:
        """
        Resolve base URL: self.base_url > base_urls[model] > base_urls[provider] > None.
        None means use SDK default (or DEFAULT_BASE_URLS in our code).
        """
        if self.base_url is not None:
            return self.base_url
        if model and model in self.base_urls:
            return self.base_urls[model]
        if provider in self.base_urls:
            return self.base_urls[provider]
        return None

    def get_default_base_url(self, provider: str) -> str:
        """Default base URL for a provider when no override is set."""
        return DEFAULT_BASE_URLS.get(provider, DEFAULT_BASE_URLS["openai"])

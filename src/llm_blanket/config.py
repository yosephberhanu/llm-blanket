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

    Resolution model:
    - **Per-model provider**: `model_provider` maps model name -> provider. If a model
      is in the map, that provider is used; otherwise the provider is inferred from the
      model name (e.g. gpt-* -> openai, claude-* -> anthropic). You can also force
      a single provider for the current call via `provider`.
    - **Per-provider overrides**: For each provider you can override:
      - **URL**: `base_urls[provider]` (e.g. base_urls["openai"]). If not set, the
        default base URL for that provider is used.
      - **API key**: `api_keys[provider]`. If not set, the key is read from the
        environment (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY — see DEFAULT_ENV_KEYS).
    - **Single-call overrides**: `api_key`, `base_url`, and `provider` override for
      the current client/call when set.
    """

    model_provider: dict[str, str] = field(default_factory=dict)
    """
    Map of model name -> provider (e.g. {"llama-3-70b-8192": "groq", "my-model": "custom"}).
    If the model is in this map, that provider is used; otherwise provider is inferred
    from the model name.
    """

    provider: Optional[str] = None
    """Force provider for this config/call (openai, anthropic, gemini, groq, xai, custom). Overrides model_provider and inference."""

    api_key: Optional[str] = None
    """Single API key override for this client. If set, overrides api_keys[provider] and env."""

    api_keys: dict[str, str] = field(default_factory=dict)
    """
    Per-provider API key overrides: provider name -> API key.
    E.g. {"openai": "sk-...", "anthropic": "sk-ant-..."}. If not set for a provider,
    the key is read from the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
    """

    base_url: Optional[str] = None
    """Single base URL override for this client. Overrides base_urls for this call."""

    base_urls: dict[str, str] = field(default_factory=dict)
    """
    Per-provider (or per-model) base URL overrides: provider or model name -> base URL.
    E.g. {"openai": "https://my-proxy.com/v1", "groq": "https://api.groq.com/openai/v1"}.
    If not set for a provider, the default base URL for that provider is used.
    """

    # Optional provider-specific options (extensible)
    extra: dict[str, Any] = field(default_factory=dict)

    def get_api_key(self, provider: str) -> Optional[str]:
        """Resolve API key: explicit first, then api_keys[provider], then env."""
        if self.api_key is not None:
            return self.api_key
        if provider in self.api_keys:
            return self.api_keys[provider]
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

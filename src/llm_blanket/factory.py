"""Factory: create an LLM instance from model name and optional config."""

from __future__ import annotations

from typing import Optional

from llm_blanket.base import BaseLLM
from llm_blanket.config import LLMConfig
from llm_blanket.registry import infer_provider

# Lazy import to avoid loading all provider SDKs at import time
def _get_openai_compatible(model: str, config: LLMConfig, provider: str) -> BaseLLM:
    from llm_blanket.providers.openai_compatible import OpenAICompatibleLLM
    return OpenAICompatibleLLM(model, config, provider=provider)


def _get_anthropic(model: str, config: LLMConfig) -> BaseLLM:
    from llm_blanket.providers.anthropic_ import AnthropicLLM
    return AnthropicLLM(model, config)


def _get_gemini(model: str, config: LLMConfig) -> BaseLLM:
    from llm_blanket.providers.gemini_ import GeminiLLM
    return GeminiLLM(model, config)


def get_llm(
    model: str,
    config: Optional[LLMConfig] = None,
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    base_urls: Optional[dict[str, str]] = None,
) -> BaseLLM:
    """
    Create an LLM instance for the given model.

    - model: Model name (e.g. "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro", "grok-2", "llama-3-70b-8192").
    - config: Optional LLMConfig. If not provided, one is built from the other kwargs.
    - provider: Force provider ("openai", "anthropic", "gemini", "groq", "xai", "custom"). If None, inferred from model.
    - api_key: Override API key (otherwise from config or env).
    - base_url: Override base URL for this client (for custom/OpenAI-compatible endpoints).
    - base_urls: Map provider or model name -> base URL (e.g. {"custom": "https://my-gateway.com/v1"}).

    For Groq models (e.g. llama-3-70b-8192), pass provider="groq" if not using a config that sets provider.
    """
    cfg = config or LLMConfig()
    cfg = LLMConfig(
        api_key=api_key if api_key is not None else cfg.api_key,
        base_url=base_url if base_url is not None else cfg.base_url,
        base_urls={**(cfg.base_urls or {}), **(base_urls or {})},
        provider=provider if provider is not None else cfg.provider,
        extra=cfg.extra,
    )

    resolved_provider = infer_provider(model, cfg.provider)

    if resolved_provider == "anthropic":
        return _get_anthropic(model, cfg)
    if resolved_provider == "gemini":
        return _get_gemini(model, cfg)
    # openai, groq, xai, custom
    return _get_openai_compatible(model, cfg, resolved_provider)

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
    model_provider: Optional[dict[str, str]] = None,
    api_key: Optional[str] = None,
    api_keys: Optional[dict[str, str]] = None,
    base_url: Optional[str] = None,
    base_urls: Optional[dict[str, str]] = None,
) -> BaseLLM:
    """
    Create an LLM instance for the given model.

    Provider resolution (first wins): explicit provider > config.model_provider[model] > infer from model name.
    Per-provider URL and API key come from config (base_urls, api_keys); if not set, defaults are used
    (default base URL per provider, API keys from .env: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).

    - model: Model name (e.g. "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro", "llama-3-70b-8192").
    - config: Optional LLMConfig (model_provider, api_keys, base_urls, etc.). If not provided, one is built from kwargs.
    - provider: Force provider for this call. Overrides model_provider and inference.
    - model_provider: Map model name -> provider (e.g. {"llama-3-70b-8192": "groq"}). Merged with config.model_provider.
    - api_key / api_keys: API key overrides (single or per-provider). Unset keys fall back to .env.
    - base_url / base_urls: Base URL overrides (single or per-provider). Unset URLs use default per provider.
    """
    cfg = config or LLMConfig()
    _model_provider = getattr(cfg, "model_provider", None) or {}
    cfg = LLMConfig(
        model_provider={**_model_provider, **(model_provider or {})},
        provider=provider if provider is not None else cfg.provider,
        api_key=api_key if api_key is not None else cfg.api_key,
        api_keys={**(cfg.api_keys or {}), **(api_keys or {})},
        base_url=base_url if base_url is not None else cfg.base_url,
        base_urls={**(cfg.base_urls or {}), **(base_urls or {})},
        extra=cfg.extra,
    )

    # Resolve provider: explicit > model_provider[model] > infer from model name
    resolved_provider = cfg.provider
    if resolved_provider is None and cfg.model_provider and model in cfg.model_provider:
        resolved_provider = cfg.model_provider[model]
    if resolved_provider is None:
        resolved_provider = infer_provider(model, None)

    if resolved_provider == "anthropic":
        return _get_anthropic(model, cfg)
    if resolved_provider == "gemini":
        return _get_gemini(model, cfg)
    # openai, groq, xai, custom
    return _get_openai_compatible(model, cfg, resolved_provider)

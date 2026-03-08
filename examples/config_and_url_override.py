"""
Configuration: per-model provider, per-provider URL and API key.

Shows how to:
- Set which provider backs each model via model_provider (e.g. llama-* -> groq).
- Override base URL and API key per provider in config; unset values use defaults
  (default URLs per provider, API keys from .env: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).

Set API keys in .env or environment for the providers you use:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, XAI_API_KEY
"""

from llm_blanket import get_llm, LLMConfig, Message


def main() -> None:
    # Per-model provider: which provider backs each model
    # Per-provider: optional URL and API key overrides (otherwise default URL + .env keys)
    config = LLMConfig(
        model_provider={
            "llama-3-70b-8192": "groq",
            "mixtral-8x7b-32768": "groq",
        },
        base_urls={
            "openai": "https://api.openai.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "xai": "https://api.x.ai/v1",
        },
        # api_keys: set per-provider if you don't want to use .env
        # api_keys={"openai": "sk-...", "groq": "gsk_..."},
    )

    # Provider resolved from model_provider or inferred from model name
    openai_llm = get_llm("gpt-4o-mini", config=config)
    print(f"OpenAI: model={openai_llm.model}, provider={openai_llm.provider}")

    groq_llm = get_llm("llama-3-70b-8192", config=config)
    print(f"Groq: model={groq_llm.model}, provider={groq_llm.provider}")

    # Optional: single-call overrides (e.g. custom endpoint for this client only)
    custom_llm = get_llm(
        "gpt-4o-mini",
        config=config,
        base_url="https://your-custom-endpoint.com/v1",
    )
    print("Custom client: base_url override applied")

    # Optional: actually call (requires API keys in .env or in config.api_keys)
    # r = openai_llm.invoke([Message("user", "Hi")])
    # print(r.content)


if __name__ == "__main__":
    main()

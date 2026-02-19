"""
Configuration and URL overrides: LLMConfig, base_url, base_urls, and explicit provider.

Shows how to:
- Pass a shared config with base_urls (e.g. for a proxy or custom endpoint).
- Override base_url for a single client.
- Force provider for models that don't infer (e.g. Groq's llama-*).

Set API keys in the environment for the providers you want to try:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, XAI_API_KEY
"""

from llm_blanket import get_llm, LLMConfig, Message


def main() -> None:
    # Shared config: URL mapping so all OpenAI-style calls go through a proxy
    # (Replace with your real proxy URL or use defaults by omitting base_urls.)
    config = LLMConfig(
        base_urls={
            "openai": "https://api.openai.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "xai": "https://api.x.ai/v1",
        }
    )

    # Client using config (provider inferred from model)
    openai_llm = get_llm("gpt-4o-mini", config=config)
    print(f"OpenAI client: model={openai_llm.model}, provider={openai_llm.provider}")

    # Override base_url for this one client (e.g. custom endpoint)
    custom_llm = get_llm(
        "gpt-4o-mini",
        config=config,
        base_url="https://your-custom-endpoint.com/v1",
    )
    print(f"Custom client: base_url override applied")

    # Groq: model name doesn't imply provider, so pass provider="groq"
    groq_llm = get_llm("llama-3-70b-8192", config=config, provider="groq")
    print(f"Groq client: model={groq_llm.model}, provider={groq_llm.provider}")

    # Optional: actually call one of them (requires API key)
    # response = openai_llm.invoke([Message("user", "Hi")])
    # print(response.content)


if __name__ == "__main__":
    main()

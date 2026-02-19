"""
Quick start: create an LLM and call it with a single user message.

Run from repo root with the appropriate API key set:
  OPENAI_API_KEY=sk-... python examples/quickstart.py

Or install and run from anywhere:
  pip install "llm-blanket[openai]"
  OPENAI_API_KEY=sk-... python -c "from examples.quickstart import main; main()"
"""

from llm_blanket import get_llm, Message


def main() -> None:
    # Provider is inferred from model name (gpt-* -> OpenAI)
    llm = get_llm("gpt-4o-mini")

    # Option 1: system and user as named arguments (temperature, max_tokens, etc. passed through)
    response = llm.invoke(
        system="You are a concise assistant.",
        user="Say hello in one sentence.",
        temperature=0.7,
        max_tokens=256,
    )

    # Option 2: messages list (Message objects or OpenAI-style dicts)
    # response = llm.invoke([Message("user", "Say hello in one sentence.")])

    print(f"Model: {response.model}")
    print(f"Content: {response.content}")
    if response.usage:
        print(f"Usage: {response.usage}")


if __name__ == "__main__":
    main()

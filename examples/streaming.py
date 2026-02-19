"""
Streaming: use invoke_stream() to receive response chunks as they arrive.

Same arguments as invoke() (system=, user=, messages, temperature, etc.).
Yields StreamChunk with .content (text delta) and .finish_reason (set on final chunk).

Run from repo root with API key set:
  OPENAI_API_KEY=sk-... python examples/streaming.py
"""

from llm_blanket import get_llm


def main() -> None:
    llm = get_llm("gpt-4o-mini")

    print("Streaming response:\n")
    for chunk in llm.invoke_stream(
        system="You are a concise assistant.",
        user="Say exactly three short sentences about the weather.",
        temperature=0.5,
        max_tokens=150,
    ):
        print(chunk.content, end="", flush=True)
        if chunk.finish_reason:
            print(f"\n[Finished: {chunk.finish_reason}]")
    print()


if __name__ == "__main__":
    main()

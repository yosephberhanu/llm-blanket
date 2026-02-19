# Examples

Runnable example scripts for **llm-blanket**. Set the right API key in the environment, then run from the repo root:

```bash
# From repo root (after pip install -e ".[openai]")
OPENAI_API_KEY=sk-... python examples/quickstart.py
python examples/config_and_url_override.py
```

| Script | Description |
|--------|-------------|
| `quickstart.py` | Create an LLM, call `invoke()` with a user message. |
| `streaming.py` | Stream response tokens with `invoke_stream()`. |
| `config_and_url_override.py` | Use `LLMConfig`, `base_urls`, `base_url`, and `provider="groq"`. |

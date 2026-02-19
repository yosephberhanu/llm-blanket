# llm-blanket

Unified Python library for LLM APIs: **OpenAI**, **Anthropic**, **Gemini**, **xAI (Grok)**, **Groq**, and **custom OpenAI-compatible** endpoints.

- Single interface: specify a model, get an LLM instance, call `invoke(messages)`.
- Provider inferred from model name (e.g. `gpt-4o` → OpenAI, `claude-3-5-sonnet` → Anthropic) or set explicitly.
- Base URL overrides via config or `base_url` / `base_urls` for custom or proxy endpoints.
- API keys from environment (LangChain/AutoGen-style) or passed in config.

## Install

```bash
pip install llm-blanket
```

Optional provider dependencies (install only what you use):

```bash
pip install "llm-blanket[openai]"       # OpenAI + Groq + xAI + custom (OpenAI-compatible)
pip install "llm-blanket[anthropic]"    # Anthropic Claude
pip install "llm-blanket[gemini]"       # Google Gemini
pip install "llm-blanket[all]"          # All providers
```

## Examples

Runnable scripts are in the [examples/](examples/) directory:

- **[examples/quickstart.py](examples/quickstart.py)** – create an LLM and call `invoke()` with a user message.
- **[examples/streaming.py](examples/streaming.py)** – stream tokens with `invoke_stream()`.
- **[examples/config_and_url_override.py](examples/config_and_url_override.py)** – `LLMConfig`, `base_urls`, `base_url`, and explicit `provider`.

Run from the repo root (set the appropriate API key first):

```bash
OPENAI_API_KEY=sk-... python examples/quickstart.py
```

## Quick start

```python
from llm_blanket import get_llm, Message

# Provider inferred from model name
llm = get_llm("gpt-4o")

# Option 1: system and user as named arguments
resp = llm.invoke(system="You are helpful.", user="Hello!")
print(resp.content)

# Option 2: messages list (Message objects or OpenAI-style dicts)
resp = llm.invoke([Message("user", "Hi")])
resp = llm([{"role": "user", "content": "Hi"}])

# Option 3: common parameters (temperature, max_tokens, etc.) are passed through to the provider
resp = llm.invoke(user="Hello!", temperature=0.7, max_tokens=256)

# Streaming: same signature as invoke(), yields StreamChunk (content delta, optional finish_reason)
for chunk in llm.invoke_stream(user="Hello!", temperature=0.7):
    print(chunk.content, end="", flush=True)
print()
```

## Streaming

Use `invoke_stream()` with the same arguments as `invoke()`. It yields `StreamChunk` objects (`.content` is the text delta; `.finish_reason` is set on the final chunk when the provider supplies it):

```python
from llm_blanket import get_llm

llm = get_llm("gpt-4o-mini")
for chunk in llm.invoke_stream(system="You are concise.", user="Count to 5."):
    print(chunk.content, end="", flush=True)
    if chunk.finish_reason:
        print(f"\n[Done: {chunk.finish_reason}]")
```

Streaming is supported for OpenAI (and OpenAI-compatible), Anthropic, and Gemini.

## Configuration

### API keys

By default, API keys are read from the environment. Use standard names so you can reuse `.env` or shell exports:

| Provider | Environment variable |
|----------|----------------------|
| OpenAI   | `OPENAI_API_KEY`     |
| Anthropic| `ANTHROPIC_API_KEY`  |
| Gemini   | `GOOGLE_API_KEY`     |
| xAI      | `XAI_API_KEY`       |
| Groq     | `GROQ_API_KEY`      |
| Custom   | `OPENAI_API_KEY` (or pass explicitly) |

Override in code:

```python
from llm_blanket import get_llm, LLMConfig

config = LLMConfig(api_key="sk-...")
llm = get_llm("gpt-4o", config=config)

# Or one-off
llm = get_llm("gpt-4o", api_key="sk-...")
```

### Base URL and URL mapping

Override the base URL for a given client (e.g. custom or proxy):

```python
# Single override for this client
llm = get_llm("gpt-4o", base_url="https://my-gateway.com/v1")

# Or via config with a mapping (e.g. per provider or per model)
config = LLMConfig(
    base_urls={
        "openai": "https://my-openai-proxy.com/v1",
        "gpt-4o": "https://special-endpoint.com/v1",
    }
)
llm = get_llm("gpt-4o", config=config)
```

Resolution order: `base_url` (direct) > `base_urls[model]` > `base_urls[provider]` > default URL for that provider.

### Forcing provider

Use when the model name doesn’t indicate the provider (e.g. Groq’s `llama-3-70b-8192`):

```python
llm = get_llm("llama-3-70b-8192", provider="groq")
```

## Supported models / providers

| Provider   | Inferred from      | Notes                    |
|-----------|--------------------|--------------------------|
| OpenAI    | `gpt-*`, `o1-*`, `o3-*` | Default base: `https://api.openai.com/v1` |
| Anthropic | `claude-*`         | Uses Anthropic Messages API |
| Gemini    | `gemini-*`         | Uses Google GenAI SDK   |
| xAI       | `grok*`, `grok-*`  | OpenAI-compatible       |
| Groq      | Set `provider="groq"` | Models like `llama-3-70b-8192`; OpenAI-compatible |
| Custom    | Set `provider="custom"` and `base_url` | Any OpenAI-compatible endpoint |

## Extensibility

- **Unified response**: `invoke()` returns an `LLMResponse` with `content`, `model`, `usage`, `finish_reason`, and optional `raw` (provider-specific object) and `tool_calls`.
- **Provider-specific options**: Pass extra kwargs to `invoke()` (e.g. `temperature`, `max_tokens`); they are forwarded to the underlying API. Use `LLMConfig(extra={...})` for client-level options.
- **Custom backends**: Implement `BaseLLM` (see `llm_blanket.base`) and register or construct your backend explicitly; the factory is focused on the built-in providers.

## Example: multiple providers and URL overrides

```python
from llm_blanket import get_llm, LLMConfig, Message

# Shared URL mapping (e.g. from app config)
config = LLMConfig(
    base_urls={
        "openai": "https://my-proxy.com/openai/v1",
        "groq": "https://api.groq.com/openai/v1",
    }
)

openai_llm = get_llm("gpt-4o-mini", config=config)
groq_llm = get_llm("llama-3-70b-8192", config=config, provider="groq")

for llm in [openai_llm, groq_llm]:
    r = llm.invoke([Message("user", "Say hi in one word.")])
    print(f"{llm.provider}: {r.content}")
```

## License

MIT

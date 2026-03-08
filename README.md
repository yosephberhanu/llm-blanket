# llm-blanket

Unified Python library for LLM APIs: **OpenAI**, **Anthropic**, **Gemini**, **xAI (Grok)**, **Groq**, and **custom OpenAI-compatible** endpoints.

- Single interface: specify a model, get an LLM instance, call `invoke(messages)`.
- **Per-model provider**: specify which provider backs each model via `model_provider`; if not set, provider is inferred from model name (e.g. `gpt-4o` → OpenAI, `claude-*` → Anthropic).
- **Per-provider overrides**: override base URL and API key per provider in config; if not set, default URLs and API keys from `.env` are used.

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

### 1. Per-model provider

Specify which provider backs each model with `model_provider` (model name → provider). If a model is not in the map, the provider is inferred from the model name (e.g. `gpt-*` → openai, `claude-*` → anthropic). Groq models like `llama-3-70b-8192` have no unique prefix, so put them in the map or pass `provider="groq"` for that call.

```python
from llm_blanket import get_llm, LLMConfig

config = LLMConfig(
    model_provider={
        "llama-3-70b-8192": "groq",
        "mixtral-8x7b-32768": "groq",
        "my-custom-model": "custom",
    }
)
llm = get_llm("llama-3-70b-8192", config=config)  # uses groq
llm = get_llm("gpt-4o", config=config)            # inferred openai
```

### 2. Per-provider URL and API key

For each provider you can override the base URL and API key. If you don't, the library uses its default base URL for that provider and the API key from the environment.

**Environment variables (default API keys):**

| Provider | Environment variable |
|----------|----------------------|
| OpenAI   | `OPENAI_API_KEY`     |
| Anthropic| `ANTHROPIC_API_KEY`  |
| Gemini   | `GOOGLE_API_KEY`     |
| xAI      | `XAI_API_KEY`        |
| Groq     | `GROQ_API_KEY`       |
| Custom   | `OPENAI_API_KEY` (or set in config) |

**Override in config:**

```python
config = LLMConfig(
    model_provider={"llama-3-70b-8192": "groq"},
    base_urls={
        "openai": "https://my-openai-proxy.com/v1",
        "groq": "https://api.groq.com/openai/v1",
    },
    api_keys={
        "openai": "sk-openai-...",
        "anthropic": "sk-ant-...",
    },
)
openai_llm = get_llm("gpt-4o", config=config)
groq_llm = get_llm("llama-3-70b-8192", config=config)
anthropic_llm = get_llm("claude-3-5-sonnet-20241022", config=config)
```

Single-call overrides: `provider`, `api_key`, and `base_url` still override for that call only.


## Supported models / providers

| Provider   | Inferred from      | Notes                    |
|-----------|--------------------|--------------------------|
| OpenAI    | `gpt-*`, `o1-*`, `o3-*` | Default base: `https://api.openai.com/v1` |
| Anthropic | `claude-*`         | Uses Anthropic Messages API |
| Gemini    | `gemini-*`         | Uses Google GenAI SDK   |
| xAI       | `grok*`, `grok-*`  | OpenAI-compatible       |
| Groq      | Set in `model_provider` or `provider="groq"` | Models like `llama-3-70b-8192`; OpenAI-compatible |
| Custom    | Set in `model_provider` or `provider="custom"` and `base_url` | Any OpenAI-compatible endpoint |

## Extensibility

- **Unified response**: `invoke()` returns an `LLMResponse` with `content`, `model`, `usage`, `finish_reason`, and optional `raw` (provider-specific object) and `tool_calls`.
- **Provider-specific options**: Pass extra kwargs to `invoke()` (e.g. `temperature`, `max_tokens`); they are forwarded to the underlying API. Use `LLMConfig(extra={...})` for client-level options.
- **Custom backends**: Implement `BaseLLM` (see `llm_blanket.base`) and register or construct your backend explicitly; the factory is focused on the built-in providers.

## Example: multiple providers with per-model and per-provider config

```python
from llm_blanket import get_llm, LLMConfig, Message

# Per-model provider + per-provider URL (and optionally api_keys; else from .env)
config = LLMConfig(
    model_provider={"llama-3-70b-8192": "groq", "mixtral-8x7b-32768": "groq"},
    base_urls={
        "openai": "https://my-proxy.com/openai/v1",
        "groq": "https://api.groq.com/openai/v1",
    },
)

openai_llm = get_llm("gpt-4o-mini", config=config)
groq_llm = get_llm("llama-3-70b-8192", config=config)

for llm in [openai_llm, groq_llm]:
    r = llm.invoke([Message("user", "Say hi in one word.")])
    print(f"{llm.provider}: {r.content}")
```

## License

MIT

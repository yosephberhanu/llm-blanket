"""OpenAI and OpenAI-compatible providers (OpenAI, Groq, xAI, custom)."""

from __future__ import annotations

from typing import Any, Iterator, Optional

from llm_blanket.base import BaseLLM, LLMResponse, Message, StreamChunk
from llm_blanket.config import LLMConfig


def _serialize_tool_calls(tool_calls: Any) -> Optional[list[dict[str, Any]]]:
    if not tool_calls:
        return None
    out = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        out.append({
            "id": getattr(tc, "id", None),
            "function": {"name": getattr(fn, "name", ""), "arguments": getattr(fn, "arguments", "")} if fn else {},
        })
    return out


def _normalize_messages(
    messages: list[Message] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not messages:
        return []
    if isinstance(messages[0], Message):
        return [m.to_openai_format() for m in messages]
    return list(messages)


class OpenAICompatibleLLM(BaseLLM):
    """
    LLM backend for OpenAI and any OpenAI-compatible API (Groq, xAI, custom).
    Uses base_url and api_key from config; supports URL overrides via config.base_urls.
    """

    def __init__(
        self,
        model: str,
        config: Optional[LLMConfig] = None,
        *,
        provider: str = "openai",
    ) -> None:
        super().__init__(model, config)
        self._provider = provider
        self._client: Any = None

    @property
    def provider(self) -> str:
        return self._provider

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires the openai package. Install with: pip install llm-blanket[openai]"
            ) from e

        cfg = self.config
        api_key = cfg.get_api_key(self._provider)
        base_url = cfg.get_base_url(self._provider, self.model)
        if base_url is None:
            base_url = cfg.get_default_base_url(self._provider)

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            **cfg.extra,
        )
        return self._client

    def _invoke_impl(
        self,
        messages: list[Message] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_messages(messages),
            **kwargs,
        }
        resp = client.chat.completions.create(**payload)

        choice = resp.choices[0] if resp.choices else None
        if choice is None:
            return LLMResponse(
                content="",
                model=resp.model or self.model,
                usage=getattr(resp, "usage", None) and {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                },
                raw=resp,
            )

        content = getattr(choice.message, "content", None) or ""
        if isinstance(content, list):
            content = " ".join(
                getattr(block, "text", block) if hasattr(block, "text") else str(block)
                for block in content
            )

        usage = None
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens"),
                "completion_tokens": getattr(resp.usage, "completion_tokens"),
                "total_tokens": getattr(resp.usage, "total_tokens"),
            }

        return LLMResponse(
            content=content,
            model=resp.model or self.model,
            usage=usage,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=resp,
            tool_calls=_serialize_tool_calls(getattr(choice.message, "tool_calls", None)),
            id=getattr(resp, "id", None),
        )

    def _invoke_stream_impl(
        self,
        messages: list[Message] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        kwargs.pop("stream", None)  # we set stream=True
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_messages(messages),
            "stream": True,
            **kwargs,
        }
        stream = client.chat.completions.create(**payload)
        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta = getattr(choice, "delta", None)
            content = (getattr(delta, "content", None) or "") if delta else ""
            finish_reason = getattr(choice, "finish_reason", None)
            yield StreamChunk(content=content, finish_reason=finish_reason)

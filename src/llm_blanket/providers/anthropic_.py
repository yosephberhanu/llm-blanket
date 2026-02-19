"""Anthropic Claude provider."""

from __future__ import annotations

from typing import Any, Iterator, Optional

from llm_blanket.base import BaseLLM, LLMResponse, Message, StreamChunk
from llm_blanket.config import LLMConfig


def _to_anthropic_messages(
    messages: list[Message] | list[dict[str, Any]],
) -> tuple[Optional[str], list[dict[str, Any]]]:
    """Convert to Anthropic format: system (optional) + messages (user/assistant only)."""
    system: Optional[str] = None
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, Message):
            role = m.role
            content = m.content
        else:
            role = m.get("role", "user")
            content = m.get("content", "")
        if role == "system":
            system = content if isinstance(content, str) else str(content)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        out.append({"role": role, "content": content})
    return system, out


class AnthropicLLM(BaseLLM):
    """LLM backend for Anthropic Claude. Uses ANTHROPIC_API_KEY from env if api_key not set."""

    @property
    def provider(self) -> str:
        return "anthropic"

    def _get_client(self) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires the anthropic package. Install with: pip install llm-blanket[anthropic]"
            ) from e
        api_key = self.config.get_api_key("anthropic")
        return Anthropic(api_key=api_key, **self.config.extra)

    def _invoke_impl(
        self,
        messages: list[Message] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        system, anthropic_messages = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": anthropic_messages,
            **kwargs,
        }
        if system:
            payload["system"] = system

        resp = client.messages.create(**payload)

        content = ""
        if resp.content:
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    content += getattr(block, "text", "") or ""

        usage = None
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": getattr(resp.usage, "input_tokens"),
                "completion_tokens": getattr(resp.usage, "output_tokens"),
                "total_tokens": getattr(resp.usage, "input_tokens", 0)
                + getattr(resp.usage, "output_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=resp.model or self.model,
            usage=usage,
            finish_reason=getattr(resp, "stop_reason", None),
            raw=resp,
            id=getattr(resp, "id", None),
        )

    def _invoke_stream_impl(
        self,
        messages: list[Message] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        system, anthropic_messages = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": anthropic_messages,
            **kwargs,
        }
        if system:
            payload["system"] = system

        with client.messages.stream(**payload) as stream:
            for text in stream.text_stream:
                yield StreamChunk(content=text, finish_reason=None)
        yield StreamChunk(content="", finish_reason="end_turn")

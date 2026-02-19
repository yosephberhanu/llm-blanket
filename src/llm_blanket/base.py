"""Base types and protocol for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from llm_blanket.config import LLMConfig

# --- Message and response types (provider-agnostic) ---


@dataclass
class Message:
    """A single chat message."""

    role: str  # "system", "user", "assistant"
    content: str | list[dict[str, Any]]  # str or list of content blocks (e.g. for vision)

    def to_openai_format(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Unified response from any provider."""

    content: str
    model: str
    usage: Optional[dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: Optional[str] = None
    raw: Optional[Any] = None  # provider-specific raw response for extension
    tool_calls: Optional[list[dict[str, Any]]] = None
    id: Optional[str] = None

    def __str__(self) -> str:
        return self.content


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    content: str  # text delta for this chunk
    finish_reason: Optional[str] = None  # set on the final chunk when available


# --- Base LLM interface ---


def _build_messages(
    messages: Optional[list[Message] | list[dict[str, Any]]] = None,
    *,
    system: Optional[str] = None,
    user: Optional[str] = None,
) -> list[Message] | list[dict[str, Any]]:
    """Build a messages list from optional messages, system, and user. Used by invoke()."""
    out: list[Message] | list[dict[str, Any]] = []
    if system is not None:
        out.append(Message("system", system))
    if messages:
        out.extend(messages)
    if user is not None:
        out.append(Message("user", user))
    if not out:
        raise ValueError("Provide at least one of: messages, system, or user")
    return out


class BaseLLM(ABC):
    """Abstract base for all LLM backends. Subclass to add provider-specific capabilities."""

    def __init__(self, model: str, config: Optional["LLMConfig"] = None) -> None:
        self.model = model
        self.config = config or LLMConfig()

    def invoke(
        self,
        messages: Optional[list[Message] | list[dict[str, Any]]] = None,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call the LLM. You can pass:
        - messages: list of Message or role/content dicts (OpenAI-style), or
        - system: system prompt string, and/or user: user prompt string.
        System is prepended, user is appended to any messages list.
        """
        built = _build_messages(messages, system=system, user=user)
        return self._invoke_impl(built, **kwargs)

    @abstractmethod
    def _invoke_impl(
        self,
        messages: list[Message] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Provider-specific implementation. Receives normalized messages list."""
        ...

    def invoke_stream(
        self,
        messages: Optional[list[Message] | list[dict[str, Any]]] = None,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """
        Call the LLM with streaming. Yields StreamChunk (content delta, optional finish_reason).
        Same signature as invoke(); pass stream=True to provider via kwargs if needed (handled internally).
        """
        built = _build_messages(messages, system=system, user=user)
        yield from self._invoke_stream_impl(built, **kwargs)

    def _invoke_stream_impl(
        self,
        messages: list[Message] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Override in providers that support streaming. Default: raise NotImplementedError."""
        raise NotImplementedError(
            f"Streaming is not implemented for provider {self.provider!r}. Use invoke() for non-streaming."
        )

    def __call__(
        self,
        messages: Optional[list[Message] | list[dict[str, Any]]] = None,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Convenience: llm(...) -> response. Accepts messages, system=, user=."""
        return self.invoke(messages, system=system, user=user, **kwargs)

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name (e.g. 'openai', 'anthropic', 'gemini')."""
        ...

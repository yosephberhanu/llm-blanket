"""Google Gemini provider."""

from __future__ import annotations

from typing import Any, Iterator, Optional

from llm_blanket.base import BaseLLM, LLMResponse, Message, StreamChunk
from llm_blanket.config import LLMConfig


def _to_gemini_contents(
    messages: list[Message] | list[dict[str, Any]],
) -> Any:
    """Convert messages to Gemini generate_content format (list of Content)."""
    from google.genai import types

    contents: list[Any] = []
    for m in messages:
        if isinstance(m, Message):
            role = m.role
            content = m.content
        else:
            role = m.get("role", "user")
            content = m.get("content", "")
        if role == "system":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(content if isinstance(content, str) else str(content))]))
            contents.append(types.Content(role="model", parts=[types.Part.from_text("Understood.")]))
            continue
        gemini_role = "user" if role == "user" else "model"
        text = content if isinstance(content, str) else str(content)
        contents.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text)]))
    return contents


class GeminiLLM(BaseLLM):
    """LLM backend for Google Gemini. Uses GOOGLE_API_KEY from env if api_key not set."""

    @property
    def provider(self) -> str:
        return "gemini"

    def _get_client(self) -> Any:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "Gemini provider requires the google-genai package. Install with: pip install llm-blanket[gemini]"
            ) from e
        api_key = self.config.get_api_key("gemini")
        return genai.Client(api_key=api_key, **self.config.extra)

    def _invoke_impl(
        self,
        messages: list[Message] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        contents = _to_gemini_contents(messages)
        # google-genai generate_content expects contents= to be the conversation
        resp = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=kwargs.pop("config", None),
            **kwargs,
        )

        text = ""
        if hasattr(resp, "text") and resp.text:
            text = resp.text
        elif getattr(resp, "candidates", None):
            c = resp.candidates[0] if resp.candidates else None
            if c and getattr(c, "content", None) and getattr(c.content, "parts", None):
                for p in c.content.parts:
                    text += getattr(p, "text", "") or ""

        usage = None
        if getattr(resp, "usage_metadata", None):
            um = resp.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }

        return LLMResponse(
            content=text,
            model=self.model,
            usage=usage,
            raw=resp,
        )

    def _invoke_stream_impl(
        self,
        messages: list[Message] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        contents = _to_gemini_contents(messages)
        config = kwargs.pop("config", None)
        stream = client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
            **kwargs,
        )
        for chunk in stream:
            text = ""
            if hasattr(chunk, "text") and chunk.text:
                text = chunk.text
            elif getattr(chunk, "candidates", None):
                c = chunk.candidates[0] if chunk.candidates else None
                if c and getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        text += getattr(p, "text", "") or ""
            yield StreamChunk(content=text, finish_reason=None)
        yield StreamChunk(content="", finish_reason="stop")

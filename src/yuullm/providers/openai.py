"""OpenAI Chat Completion API provider implementation.

This provider uses the ``/v1/chat/completions`` endpoint, which is also
the wire protocol used by many third-party vendors (DeepSeek, OpenRouter,
Together, Groq, etc.).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import openai

from ..cache_config import CacheConfig
from ..types import (
    Message,
    RawChunkHook,
    Reasoning,
    Response,
    StreamItem,
    StreamResult,
    Tick,
    ToolCall,
    Usage,
)


class OpenAIChatCompletionProvider:
    """Provider for the OpenAI Chat Completion API (``/v1/chat/completions``).

    Also works with any vendor that exposes an OpenAI-compatible
    chat-completion endpoint (DeepSeek, OpenRouter, Together, Groq, …).

    Parameters
    ----------
    api_key : str | None
        API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url : str | None
        Override the base URL for third-party vendors.
    organization : str | None
        OpenAI organization header.
    provider_name : str
        Vendor / supplier identifier used in :class:`Usage` and pricing
        lookups.  Defaults to ``"openai"``; set to ``"deepseek"``,
        ``"openrouter"``, etc. when using a compatible endpoint.
    cache_config : CacheConfig | None
        When set on OpenAI direct, controls ``prompt_cache_retention``.
        OpenAI uses automatic prompt caching -- no content annotation
        needed.  Ignored for non-OpenAI vendors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        *,
        provider_name: str = "openai",
        cache_config: CacheConfig | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._provider_name = provider_name
        self._cache_config = cache_config

    @property
    def api_type(self) -> str:
        return "openai-chat-completion"

    @property
    def provider(self) -> str:
        return self._provider_name

    # ------------------------------------------------------------------
    # Message conversion: (role, items) tuples -> OpenAI API format
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        """Convert (role, items) tuples to OpenAI chat format.

        All items are dicts with a ``type`` key. Dispatch by type:
        - text → simple content or multimodal content array
        - tool_call → assistant tool_calls
        - tool_result → tool result messages
        - image_url, input_audio, file → multimodal content blocks
        """
        result: list[dict] = []
        for role, items in messages:
            if role == "system":
                text = "".join(
                    it["text"] for it in items if it.get("type") == "text"
                )
                result.append({"role": "system", "content": text})

            elif role == "user":
                # If all items are text, use simple string content
                if all(it.get("type") == "text" for it in items):
                    result.append(
                        {
                            "role": "user",
                            "content": "".join(
                                it["text"] for it in items
                            ),
                        }
                    )
                else:
                    # Multimodal: pass content blocks as-is (already dict)
                    result.append({"role": "user", "content": list(items)})

            elif role == "assistant":
                entry: dict[str, Any] = {"role": "assistant"}
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for it in items:
                    if it.get("type") == "text":
                        text_parts.append(it["text"])
                    elif it.get("type") == "tool_call":
                        tool_calls.append(
                            {
                                "id": it["id"],
                                "type": "function",
                                "function": {
                                    "name": it["name"],
                                    "arguments": it.get("arguments", "{}"),
                                },
                            }
                        )
                if text_parts:
                    entry["content"] = "".join(text_parts)
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)

            elif role == "tool":
                for it in items:
                    if it.get("type") == "tool_result":
                        content = it.get("content", "")
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": it["tool_call_id"],
                                "content": content,
                            }
                        )

        return result

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict]:
        """Convert json_schema tool dicts to OpenAI format.

        Accepts dicts already in OpenAI format (``{"type": "function", ...}``),
        or bare function dicts (``{"name": ..., "description": ..., "parameters": ...}``).
        """
        result: list[dict] = []
        for t in tools:
            if t.get("type") == "function":
                # Already in OpenAI format (e.g. from yuutools.ToolManager.specs())
                result.append(t)
            else:
                # Bare dict: wrap in OpenAI function format
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                        },
                    }
                )
        return result

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        on_raw_chunk: RawChunkHook | None = None,
        **kwargs,
    ) -> StreamResult:
        store: dict = {}

        api_messages = self._convert_messages(messages)

        create_kwargs: dict = {
            "model": model,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = self._convert_tools(tools)

        # OpenAI automatic prompt caching: set retention policy
        if (
            self._cache_config is not None
            and self._provider_name == "openai"
            and "prompt_cache_retention" not in create_kwargs
        ):
            # If refresh_interval > 5 min, request 24h retention
            if self._cache_config.refresh_interval > 300:
                create_kwargs["prompt_cache_retention"] = "24h"

        response = await self._client.chat.completions.create(**create_kwargs)

        iterator = self._iterate(response, model, store, on_raw_chunk)
        return iterator, store

    async def _iterate(
        self,
        response,
        model: str,
        store: dict,
        on_raw_chunk: RawChunkHook | None = None,
    ) -> AsyncIterator[StreamItem]:
        # Accumulate tool calls by index
        tool_calls_acc: dict[int, dict] = {}
        request_id: str | None = None

        async for chunk in response:
            if on_raw_chunk is not None:
                on_raw_chunk(chunk)
            # Capture request id from first chunk
            if request_id is None and chunk.id:
                request_id = chunk.id

            # Usage comes in the final chunk (stream_options.include_usage)
            if chunk.usage is not None:
                store["usage"] = Usage(
                    provider=self._provider_name,
                    model=chunk.model or model,
                    request_id=request_id,
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                    cache_read_tokens=self._extract_cache_read(chunk.usage),
                    total_tokens=chunk.usage.total_tokens,
                )
                # Extract provider-reported cost (e.g. OpenRouter)
                provider_cost = getattr(chunk.usage, "cost", None)
                if provider_cost is not None:
                    store["provider_cost"] = float(provider_cost)

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Reasoning / thinking content
            # Different providers use different field names:
            #   - reasoning_content: DeepSeek
            #   - reasoning: OpenRouter
            # The openai SDK may not expose these as typed attributes,
            # so we fall back to model_dump() to catch extra fields.
            reasoning_text = getattr(delta, "reasoning_content", None)
            if not reasoning_text:
                try:
                    delta_dict = delta.model_dump(exclude_none=True)
                except Exception:
                    delta_dict = {}
                reasoning_text = delta_dict.get("reasoning_content") or delta_dict.get(
                    "reasoning"
                )
            if reasoning_text:
                yield Reasoning(item=reasoning_text)

            # Regular content
            if delta.content:
                yield Response(item={"type": "text", "text": delta.content})

            # Tool calls (streamed incrementally)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    acc = tool_calls_acc[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["arguments"] += tc_delta.function.arguments
                # Keep the consumer loop spinning so on_raw_chunk
                # side-effects (e.g. pending_sse) can be flushed.
                if on_raw_chunk is not None:
                    yield Tick()

        # Yield fully assembled tool calls
        for _idx in sorted(tool_calls_acc):
            acc = tool_calls_acc[_idx]
            yield ToolCall(
                id=acc["id"],
                name=acc["name"],
                arguments=acc["arguments"],
            )

        # Ensure usage is set even if the API didn't include it
        store.setdefault(
            "usage",
            Usage(provider=self._provider_name, model=model, request_id=request_id),
        )

        # Extract provider cost if available (e.g. OpenRouter)
        store.setdefault("provider_cost", None)

    @staticmethod
    def _extract_cache_read(usage_obj) -> int:
        """Extract cached token count from OpenAI usage object."""
        details = getattr(usage_obj, "prompt_tokens_details", None)
        if details is not None:
            return getattr(details, "cached_tokens", 0) or 0
        return 0


# Backward-compatible alias (deprecated)
OpenAIProvider = OpenAIChatCompletionProvider

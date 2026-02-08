"""OpenAI / OpenAI-compatible provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import openai

from ..types import (
    AssistantMessage,
    Message,
    Reasoning,
    Response,
    StreamItem,
    StreamResult,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
)


class OpenAIProvider:
    """Provider for OpenAI and OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        *,
        provider_name: str = "openai",
    ) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return self._provider_name

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        result: list[dict] = []
        for msg in messages:
            match msg:
                case SystemMessage(content=c):
                    result.append({"role": "system", "content": c})
                case UserMessage(content=c):
                    result.append({"role": "user", "content": c})
                case AssistantMessage(content=c, tool_calls=tcs):
                    entry: dict = {"role": "assistant"}
                    if c is not None:
                        entry["content"] = c
                    if tcs:
                        entry["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                },
                            }
                            for tc in tcs
                        ]
                    result.append(entry)
                case ToolResultMessage(tool_call_id=tid, content=c):
                    result.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": c,
                    })
        return result

    @staticmethod
    def _convert_tools(tools: list[ToolSpec]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
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

        response = await self._client.chat.completions.create(**create_kwargs)

        iterator = self._iterate(response, model, store)
        return iterator, store

    async def _iterate(
        self,
        response,
        model: str,
        store: dict,
    ) -> AsyncIterator[StreamItem]:
        # Accumulate tool calls by index
        tool_calls_acc: dict[int, dict] = {}
        request_id: str | None = None

        async for chunk in response:
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
                reasoning_text = delta_dict.get("reasoning_content") or delta_dict.get("reasoning")
            if reasoning_text:
                yield Reasoning(text=reasoning_text)

            # Regular content
            if delta.content:
                yield Response(text=delta.content)

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

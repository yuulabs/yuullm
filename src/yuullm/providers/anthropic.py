"""Anthropic provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import anthropic

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


class AnthropicProvider:
    """Provider for the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        provider_name: str = "anthropic",
    ) -> None:
        kwargs: dict = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return self._provider_name

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """Separate the system message (Anthropic uses a top-level param)."""
        system_text: str | None = None
        rest: list[Message] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_text = msg.content
            else:
                rest.append(msg)
        return system_text, rest

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        result: list[dict] = []
        for msg in messages:
            match msg:
                case UserMessage(content=c):
                    result.append({"role": "user", "content": c})
                case AssistantMessage(content=c, tool_calls=tcs):
                    content_blocks: list[dict] = []
                    if c is not None:
                        content_blocks.append({"type": "text", "text": c})
                    if tcs:
                        for tc in tcs:
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": json.loads(tc.arguments) if tc.arguments else {},
                            })
                    result.append({"role": "assistant", "content": content_blocks})
                case ToolResultMessage(tool_call_id=tid, content=c):
                    result.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tid,
                                "content": c,
                            }
                        ],
                    })
        return result

    @staticmethod
    def _convert_tools(tools: list[ToolSpec]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
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

        system_text, rest = self._extract_system(messages)
        api_messages = self._convert_messages(rest)

        create_kwargs: dict = {
            "model": model,
            "messages": api_messages,
            "max_tokens": kwargs.pop("max_tokens", 8192),
            **kwargs,
        }
        if system_text is not None:
            create_kwargs["system"] = system_text
        if tools:
            create_kwargs["tools"] = self._convert_tools(tools)

        iterator = self._iterate(create_kwargs, model, store)
        return iterator, store

    async def _iterate(
        self,
        create_kwargs: dict,
        model: str,
        store: dict,
    ) -> AsyncIterator[StreamItem]:
        # Accumulate tool calls by block index
        tool_calls_acc: dict[int, dict] = {}
        current_block_index: int = -1
        current_block_type: str = ""
        request_id: str | None = None

        async with self._client.messages.stream(**create_kwargs) as stream:
            async for event in stream:
                match event.type:
                    case "message_start":
                        msg = event.message
                        request_id = msg.id
                    case "content_block_start":
                        current_block_index = event.index
                        block = event.content_block
                        current_block_type = block.type
                        if block.type == "tool_use":
                            tool_calls_acc[current_block_index] = {
                                "id": block.id,
                                "name": block.name,
                                "arguments": "",
                            }
                    case "content_block_delta":
                        delta = event.delta
                        match delta.type:
                            case "thinking_delta":
                                yield Reasoning(text=delta.thinking)
                            case "text_delta":
                                yield Response(text=delta.text)
                            case "input_json_delta":
                                if current_block_index in tool_calls_acc:
                                    tool_calls_acc[current_block_index]["arguments"] += delta.partial_json
                    case "content_block_stop":
                        if current_block_index in tool_calls_acc:
                            acc = tool_calls_acc.pop(current_block_index)
                            yield ToolCall(
                                id=acc["id"],
                                name=acc["name"],
                                arguments=acc["arguments"],
                            )
                    case "message_delta":
                        pass  # stop_reason, etc.

            # Extract usage from the final message
            final_message = stream.get_final_message()

        store["usage"] = Usage(
            provider=self._provider_name,
            model=final_message.model or model,
            request_id=request_id,
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
            cache_read_tokens=getattr(final_message.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(final_message.usage, "cache_creation_input_tokens", 0) or 0,
        )
        store.setdefault("provider_cost", None)

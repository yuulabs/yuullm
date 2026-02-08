"""Anthropic provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from ..types import (
    Message,
    Reasoning,
    Response,
    StreamItem,
    StreamResult,
    ToolCall,
    Usage,
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
    # Message conversion: (role, items) tuples -> Anthropic API format
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """Separate system messages (Anthropic uses a top-level param)."""
        system_text: str | None = None
        rest: list[Message] = []
        for role, items in messages:
            if role == "system":
                # Concatenate text items from system messages
                text = " ".join(it for it in items if isinstance(it, str))
                system_text = text
            else:
                rest.append((role, items))
        return system_text, rest

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        """Convert (role, items) tuples to Anthropic messages format.

        Handles:
        - ("user", ["text"]) -> {"role": "user", "content": "text"}
        - ("user", ["text", {"type": "image", ...}]) -> multimodal content blocks
        - ("assistant", ["text", {"type": "tool_call", ...}]) -> content blocks
        - ("tool", [{"type": "tool_result", ...}]) -> tool_result content blocks
        """
        result: list[dict] = []
        for role, items in messages:
            if role == "user":
                # If all items are plain strings, use simple content
                if all(isinstance(it, str) for it in items):
                    result.append(
                        {
                            "role": "user",
                            "content": " ".join(
                                it for it in items if isinstance(it, str)
                            ),
                        }
                    )
                else:
                    content: list[dict] = []
                    for it in items:
                        if isinstance(it, str):
                            content.append({"type": "text", "text": it})
                        else:
                            content.append(it)
                    result.append({"role": "user", "content": content})

            elif role == "assistant":
                content_blocks: list[dict] = []
                for it in items:
                    if isinstance(it, str):
                        content_blocks.append({"type": "text", "text": it})
                    elif isinstance(it, dict) and it.get("type") == "tool_call":
                        args = it.get("arguments", "{}")
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": it["id"],
                                "name": it["name"],
                                "input": json.loads(args)
                                if isinstance(args, str)
                                else args,
                            }
                        )
                result.append({"role": "assistant", "content": content_blocks})

            elif role == "tool":
                # Anthropic expects tool results as user messages with tool_result blocks
                tool_results: list[dict] = []
                for it in items:
                    if isinstance(it, dict) and it.get("type") == "tool_result":
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": it["tool_call_id"],
                                "content": it.get("content", ""),
                            }
                        )
                if tool_results:
                    result.append({"role": "user", "content": tool_results})

        return result

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict]:
        """Convert json_schema tool dicts to Anthropic format.

        Accepts dicts in OpenAI format (``{"type": "function", "function": {...}}``)
        or bare function dicts (``{"name": ..., ...}``).
        """
        result: list[dict] = []
        for t in tools:
            if t.get("type") == "function":
                # OpenAI format (e.g. from yuutools.ToolManager.specs())
                fn = t["function"]
                result.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {}),
                    }
                )
            else:
                # Bare dict
                result.append(
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get("parameters", {}),
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
                                yield Reasoning(item=delta.thinking)
                            case "text_delta":
                                yield Response(item=delta.text)
                            case "input_json_delta":
                                if current_block_index in tool_calls_acc:
                                    tool_calls_acc[current_block_index][
                                        "arguments"
                                    ] += delta.partial_json
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
            cache_read_tokens=getattr(final_message.usage, "cache_read_input_tokens", 0)
            or 0,
            cache_write_tokens=getattr(
                final_message.usage, "cache_creation_input_tokens", 0
            )
            or 0,
        )
        store.setdefault("provider_cost", None)

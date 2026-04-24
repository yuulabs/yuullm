"""OpenAI-compatible chat-completion message conversion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..types import Content, Message, ToolCallItem
from ._content import (
    content_blocks,
    content_items,
    split_assistant_items,
    tool_result_items,
)


def convert_openai_chat_messages(
    messages: Sequence[Message], *, preserve_cache_control: bool = False
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role in {"system", "user"}:
            items = content_items(message.content, role=message.role)
            result.append(
                {
                    "role": message.role,
                    "content": content_blocks(
                        items, preserve_cache_control=preserve_cache_control
                    ),
                    **message.provider_extra,
                }
            )
        elif message.role == "assistant":
            content, tool_calls = split_assistant_items(message.content)
            entry: dict[str, Any] = {
                "role": "assistant",
                **message.provider_extra,
            }
            if content:
                entry["content"] = content_blocks(
                    content, preserve_cache_control=preserve_cache_control
                )
            if tool_calls:
                entry["tool_calls"] = [
                    openai_tool_call(item) for item in tool_calls
                ]
            result.append(entry)
        else:
            for item in tool_result_items(message.content):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["tool_call_id"],
                        "content": openai_tool_result_content(
                            item["content"],
                            preserve_cache_control=preserve_cache_control,
                        ),
                        **message.provider_extra,
                    }
                )
    return result


def openai_tool_call(item: ToolCallItem) -> dict[str, Any]:
    return {
        "id": item["id"],
        "type": "function",
        "function": {
            "name": item["name"],
            "arguments": item["arguments"],
        },
    }


def openai_tool_result_content(
    content: str | Content, *, preserve_cache_control: bool
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return content_blocks(content, preserve_cache_control=preserve_cache_control)

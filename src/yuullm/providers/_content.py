"""Shared provider payload helpers."""

from __future__ import annotations

from typing import Any, cast

from ..types import (
    ContentItem,
    MessageContent,
    MessageItem,
    ToolCallItem,
    ToolResultItem,
    is_audio_item,
    is_file_item,
    is_image_item,
    is_text_item,
    is_tool_call_item,
    is_tool_result_item,
    to_plain_dict,
)


def has_cache_control(items: MessageContent) -> bool:
    return any("cache_control" in item for item in items)


def is_content_item(item: MessageItem) -> bool:
    return (
        is_text_item(item)
        or is_image_item(item)
        or is_audio_item(item)
        or is_file_item(item)
    )


def content_items(items: MessageContent, *, role: str) -> list[ContentItem]:
    for item in items:
        if not is_content_item(item):
            raise TypeError(f"{role} messages only accept content items")
    return cast(list[ContentItem], items)


def split_assistant_items(
    items: MessageContent,
) -> tuple[list[ContentItem], list[ToolCallItem]]:
    content: list[ContentItem] = []
    tool_calls: list[ToolCallItem] = []
    for item in items:
        if is_tool_call_item(item):
            tool_calls.append(item)
        elif is_content_item(item):
            content.append(cast(ContentItem, item))
        else:
            raise TypeError("assistant messages only accept content and tool-call items")
    return content, tool_calls


def content_blocks(
    items: list[ContentItem], *, preserve_cache_control: bool = True
) -> list[dict[str, Any]]:
    if preserve_cache_control or not has_cache_control(cast(MessageContent, items)):
        return cast(list[dict[str, Any]], items)
    blocks = []
    for item in items:
        block = to_plain_dict(item)
        block.pop("cache_control", None)
        blocks.append(block)
    return blocks


def tool_result_items(items: MessageContent) -> list[ToolResultItem]:
    result: list[ToolResultItem] = []
    for item in items:
        if not is_tool_result_item(item):
            raise TypeError("tool messages only accept tool-result items")
        result.append(item)
    return result

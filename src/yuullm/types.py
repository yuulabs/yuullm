"""Core types for yuullm.

Lightweight, minimal abstractions. Messages are plain tuples, tools are
plain dicts.  Only output stream types and usage/cost use msgspec structs.
Content items use TypedDict for type safety, with structures aligned to
the OpenAI API format.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Required, TypedDict

import msgspec


# ---------------------------------------------------------------------------
# Core content types -- TypedDict definitions aligned with OpenAI API
# ---------------------------------------------------------------------------


class ToolCallItem(TypedDict):
    """A tool invocation embedded in an assistant message."""

    type: Literal["tool_call"]
    id: str
    name: str
    arguments: str  # raw JSON string


class ToolResultItem(TypedDict):
    """A tool execution result embedded in a tool message."""

    type: Literal["tool_result"]
    tool_call_id: str
    content: str


class TextItem(TypedDict):
    """A text content block."""

    type: Literal["text"]
    text: str


class _ImageURL(TypedDict, total=False):
    url: Required[str]
    detail: Literal["auto", "low", "high"]


class ImageItem(TypedDict):
    """An image content block (URL or base64)."""

    type: Literal["image_url"]
    image_url: _ImageURL


class _InputAudio(TypedDict, total=False):
    data: Required[str]  # base64 encoded
    format: Required[Literal["wav", "mp3"]]


class AudioItem(TypedDict):
    """An audio input content block."""

    type: Literal["input_audio"]
    input_audio: _InputAudio


class _FileData(TypedDict, total=False):
    file_data: str  # base64 encoded
    file_id: str
    filename: str


class FileItem(TypedDict):
    """A file content block."""

    type: Literal["file"]
    file: _FileData


# The union of all structured content items.
DictItem = ToolCallItem | ToolResultItem | TextItem | ImageItem | AudioItem | FileItem

# A content item: plain text string or a typed structured dict.
Item = str | DictItem


# ---------------------------------------------------------------------------
# StreamItem variants (output types -- what the model produces)
# ---------------------------------------------------------------------------


class Reasoning(msgspec.Struct, frozen=True):
    """A fragment of the model's chain-of-thought / extended thinking.

    The content can be plain text (str) or multimodal content (dict),
    depending on the model's output format.
    """

    item: Item


class ToolCall(msgspec.Struct, frozen=True):
    """A tool invocation request emitted by the model."""

    id: str
    name: str
    arguments: str  # raw JSON string


class Response(msgspec.Struct, frozen=True):
    """A fragment of the model's final reply.

    The content can be plain text (str) or multimodal content (dict),
    allowing models to output structured or non-text content.
    """

    item: Item


StreamItem = Reasoning | ToolCall | Response

# Message = (role, items)
# role: "system" | "user" | "assistant" | "tool"
# items: list of content items
Message = tuple[str, list[Item]]

# History is just a list of messages
History = list[Message]


# ---------------------------------------------------------------------------
# Helper functions for constructing messages
# ---------------------------------------------------------------------------


def system(content: str) -> Message:
    """Create a system message."""
    return ("system", [content])


def user(*items: Item) -> Message:
    """Create a user message with one or more content items.

    Examples::

        user("Hello!")
        user("What is this?", ImageItem(type="image_url", image_url={"url": "..."}))
    """
    return ("user", list(items))


def assistant(*items: Item) -> Message:
    """Create an assistant message.

    Examples::

        assistant("Here is the answer.")
        assistant("Let me search.", ToolCallItem(
            type="tool_call",
            id="tc_1",
            name="search",
            arguments='{"q": "test"}',
        ))
    """
    return ("assistant", list(items))


def tool(tool_call_id: str, content: str) -> Message:
    """Create a tool result message.

    Example::

        tool("tc_1", "Search returned 5 results.")
    """
    result: ToolResultItem = {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "content": content,
    }
    return ("tool", [result])


# ---------------------------------------------------------------------------
# Usage & Cost
# ---------------------------------------------------------------------------


class Usage(msgspec.Struct, frozen=True):
    """Token usage reported by the API."""

    provider: str
    model: str
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int | None = None


class Cost(msgspec.Struct, frozen=True):
    """Cost breakdown for a single request.  All amounts in USD."""

    input_cost: float
    output_cost: float
    total_cost: float
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    source: str = ""  # "provider" | "yaml" | "genai-prices"


# ---------------------------------------------------------------------------
# Type alias for the stream return
# ---------------------------------------------------------------------------

Store = dict
"""Mutable dict populated after the stream is exhausted.

Expected keys (set by the framework):
    ``"usage"``  – :class:`Usage`
    ``"cost"``   – :class:`Cost` | ``None``
"""

StreamResult = tuple[AsyncIterator[StreamItem], Store]
"""Return type of ``Provider.stream()`` and ``YLLMClient.stream()``."""

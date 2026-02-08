"""Core types for yuullm.

Lightweight, minimal abstractions. Messages are plain tuples, tools are
plain dicts.  Only output stream types and usage/cost use msgspec structs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import msgspec


# ---------------------------------------------------------------------------
# StreamItem variants (output types -- what the model produces)
# ---------------------------------------------------------------------------


class Reasoning(msgspec.Struct, frozen=True):
    """A fragment of the model's chain-of-thought / extended thinking."""

    text: str


class ToolCall(msgspec.Struct, frozen=True):
    """A tool invocation request emitted by the model."""

    id: str
    name: str
    arguments: str  # raw JSON string


class Response(msgspec.Struct, frozen=True):
    """A fragment of the model's final text reply."""

    text: str


StreamItem = Reasoning | ToolCall | Response


# ---------------------------------------------------------------------------
# Message types -- lightweight tuple-based
# ---------------------------------------------------------------------------

# A content item in a message: plain text or structured dict
# (images, audio, tool_calls, tool_results, etc.)
Item = str | dict[str, Any]

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
        user("What is this?", {"type": "image_url", "url": "..."})
    """
    return ("user", list(items))


def assistant(*items: Item) -> Message:
    """Create an assistant message.

    Examples::

        assistant("Here is the answer.")
        assistant("Let me search.", {
            "type": "tool_call",
            "id": "tc_1",
            "name": "search",
            "arguments": '{"q": "test"}'
        })
    """
    return ("assistant", list(items))


def tool(tool_call_id: str, content: str) -> Message:
    """Create a tool result message.

    Example::

        tool("tc_1", "Search returned 5 results.")
    """
    return (
        "tool",
        [{"type": "tool_result", "tool_call_id": tool_call_id, "content": content}],
    )


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

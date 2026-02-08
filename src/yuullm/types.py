"""Core types for yuullm.

All data types use ``msgspec.Struct`` for zero-copy serialization.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import msgspec


# ---------------------------------------------------------------------------
# StreamItem variants
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
# Message types
# ---------------------------------------------------------------------------

class SystemMessage(msgspec.Struct, frozen=True):
    """System-level instruction."""

    content: str
    role: Literal["system"] = "system"


class UserMessage(msgspec.Struct, frozen=True):
    """User turn."""

    content: str
    role: Literal["user"] = "user"


class AssistantMessage(msgspec.Struct, frozen=True):
    """Assistant turn (may include tool calls)."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    role: Literal["assistant"] = "assistant"


class ToolResultMessage(msgspec.Struct, frozen=True):
    """Result returned from a tool execution."""

    tool_call_id: str
    content: str
    role: Literal["tool"] = "tool"


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------

class ToolSpec(msgspec.Struct, frozen=True):
    """Describes a tool the model may call.

    ``parameters`` should be a JSON-Schema dict describing the function
    parameters.
    """

    name: str
    description: str
    parameters: dict  # JSON Schema object


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

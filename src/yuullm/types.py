"""Core types for yuullm.

Lightweight, minimal abstractions. Messages are plain tuples, tools are
plain dicts.  Only output stream types and usage/cost use msgspec structs.
Content items use TypedDict for type safety, with structures aligned to
the OpenAI API format.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, Required, TypeGuard, TypedDict, overload

import msgspec


# ---------------------------------------------------------------------------
# Core content types -- TypedDict definitions aligned with OpenAI API
# ---------------------------------------------------------------------------


class CacheControl(TypedDict, total=False):
    """Provider-specific prompt-cache metadata attached to a content block."""

    type: Required[Literal["ephemeral"]]
    ttl: int


class _CacheAnnotated(TypedDict, total=False):
    cache_control: CacheControl


class ToolCallItem(_CacheAnnotated):
    """A tool invocation embedded in an assistant message."""

    type: Literal["tool_call"]
    id: str
    name: str
    arguments: str  # raw JSON string


class TextItem(_CacheAnnotated):
    """A text content block."""

    type: Literal["text"]
    text: str


class _ImageURL(TypedDict, total=False):
    url: Required[str]
    detail: Literal["auto", "low", "high"]


class ImageItem(_CacheAnnotated):
    """An image content block (URL or base64)."""

    type: Literal["image_url"]
    image_url: _ImageURL


ToolResultContentItem = TextItem | ImageItem


class ToolResultItem(_CacheAnnotated):
    """A tool execution result embedded in a tool message."""

    type: Literal["tool_result"]
    tool_call_id: str
    content: str | list[ToolResultContentItem]


class _InputAudio(TypedDict, total=False):
    data: Required[str]  # base64 encoded
    format: Required[Literal["wav", "mp3"]]


class AudioItem(_CacheAnnotated):
    """An audio input content block."""

    type: Literal["input_audio"]
    input_audio: _InputAudio


class _FileData(TypedDict, total=False):
    file_data: str  # base64 encoded
    file_id: str
    filename: str


class FileItem(_CacheAnnotated):
    """A file content block."""

    type: Literal["file"]
    file: _FileData


# The union of all structured content items.
DictItem = ToolCallItem | ToolResultItem | TextItem | ImageItem | AudioItem | FileItem
UserItem = TextItem | ImageItem | AudioItem | FileItem
AssistantItem = TextItem | ToolCallItem

# A content item is always a structured dict.
# str was previously allowed but is now removed -- use TextItem instead.
Item = DictItem


# ---------------------------------------------------------------------------
# StreamItem variants (output types -- what the model produces)
# ---------------------------------------------------------------------------


class Reasoning(msgspec.Struct, frozen=True):
    """A fragment of the model's chain-of-thought / extended thinking.

    The content is always a content dict — typically a ``TextItem``
    (``{"type": "text", "text": "..."}``).  This mirrors :class:`Response`,
    which also carries an ``Item``.
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


class Tick(msgspec.Struct, frozen=True):
    """Lightweight heartbeat yielded during tool-call argument streaming.

    While the provider accumulates tool-call deltas it normally yields
    nothing, which starves consumers that rely on the iteration loop to
    flush side-channel data (e.g. ``pending_sse`` populated by
    ``on_raw_chunk`` hooks).  ``Tick`` keeps the async-for loop spinning
    so those consumers can act promptly.

    Consumers that only care about ``Reasoning | ToolCall | Response``
    can safely ignore ``Tick`` — it carries no payload.
    """


StreamItem = Reasoning | ToolCall | Response | Tick


class ProviderModel(msgspec.Struct, frozen=True):
    """A model surfaced by a provider's model-list API."""

    id: str
    display_name: str | None = None
    supports_vision: bool | None = None


Role = Literal["system", "user", "assistant", "tool"]
SystemMessage = tuple[Literal["system"], list[TextItem]]
UserMessage = tuple[Literal["user"], list[UserItem]]
AssistantMessage = tuple[Literal["assistant"], list[AssistantItem]]
ToolMessage = tuple[Literal["tool"], list[ToolResultItem]]
Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage

# History is just a list of messages
History = list[Message]


# ---------------------------------------------------------------------------
# Helper functions for constructing messages
# ---------------------------------------------------------------------------


@overload
def _to_item(it: str) -> TextItem: ...


@overload
def _to_item(it: DictItem) -> DictItem: ...


def _to_item(it: str | DictItem) -> Item:
    """Convert a str to TextItem; pass dicts through."""
    if isinstance(it, str):
        return {"type": "text", "text": it}
    return it


def _to_user_item(it: str | UserItem) -> UserItem:
    item = _to_item(it)
    if (
        is_text_item(item)
        or is_image_item(item)
        or is_audio_item(item)
        or is_file_item(item)
    ):
        return item
    raise TypeError("user() only accepts text, image, audio, and file items")


def _to_assistant_item(it: str | AssistantItem) -> AssistantItem:
    item = _to_item(it)
    if is_text_item(item) or is_tool_call_item(item):
        return item
    raise TypeError("assistant() only accepts text and tool-call items")


def system(content: str) -> SystemMessage:
    """Create a system message."""
    return ("system", [{"type": "text", "text": content}])


def user(*items: str | UserItem) -> UserMessage:
    """Create a user message with one or more content items.

    Examples::

        user("Hello!")
        user("What is this?", ImageItem(type="image_url", image_url={"url": "..."}))
    """
    return ("user", [_to_user_item(it) for it in items])


def assistant(*items: str | AssistantItem) -> AssistantMessage:
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
    return ("assistant", [_to_assistant_item(it) for it in items])


def tool(tool_call_id: str, content: str | list[TextItem | ImageItem]) -> ToolMessage:
    """Create a tool result message.

    Content can be a plain string or a list of content blocks (for multimodal
    tool results, e.g. images). The list format follows OpenAI's convention::

        [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]

    Example::

        tool("tc_1", "Search returned 5 results.")
        tool("tc_1", [{"type": "text", "text": "Here is the image"}, {"type": "image_url", ...}])
    """
    result: ToolResultItem = {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "content": content,
    }
    return ("tool", [result])


def is_text_item(item: DictItem) -> TypeGuard[TextItem]:
    return item["type"] == "text"


def is_tool_call_item(item: DictItem) -> TypeGuard[ToolCallItem]:
    return item["type"] == "tool_call"


def is_tool_result_item(item: DictItem) -> TypeGuard[ToolResultItem]:
    return item["type"] == "tool_result"


def is_image_item(item: DictItem) -> TypeGuard[ImageItem]:
    return item["type"] == "image_url"


def is_audio_item(item: DictItem) -> TypeGuard[AudioItem]:
    return item["type"] == "input_audio"


def is_file_item(item: DictItem) -> TypeGuard[FileItem]:
    return item["type"] == "file"


@overload
def with_cache_control(
    item: ToolCallItem, cache_control: CacheControl
) -> ToolCallItem: ...


@overload
def with_cache_control(
    item: ToolResultItem, cache_control: CacheControl
) -> ToolResultItem: ...


@overload
def with_cache_control(item: TextItem, cache_control: CacheControl) -> TextItem: ...


@overload
def with_cache_control(item: ImageItem, cache_control: CacheControl) -> ImageItem: ...


@overload
def with_cache_control(item: AudioItem, cache_control: CacheControl) -> AudioItem: ...


@overload
def with_cache_control(item: FileItem, cache_control: CacheControl) -> FileItem: ...


def with_cache_control(item: DictItem, cache_control: CacheControl) -> DictItem:
    """Return a copy of *item* with provider cache metadata attached."""
    if is_text_item(item):
        return {
            "type": "text",
            "text": item["text"],
            "cache_control": cache_control,
        }
    if is_tool_call_item(item):
        return {
            "type": "tool_call",
            "id": item["id"],
            "name": item["name"],
            "arguments": item["arguments"],
            "cache_control": cache_control,
        }
    if is_tool_result_item(item):
        return {
            "type": "tool_result",
            "tool_call_id": item["tool_call_id"],
            "content": item["content"],
            "cache_control": cache_control,
        }
    if is_image_item(item):
        return {
            "type": "image_url",
            "image_url": item["image_url"],
            "cache_control": cache_control,
        }
    if is_audio_item(item):
        return {
            "type": "input_audio",
            "input_audio": item["input_audio"],
            "cache_control": cache_control,
        }
    if is_file_item(item):
        return {
            "type": "file",
            "file": item["file"],
            "cache_control": cache_control,
        }
    raise AssertionError(f"Unsupported item type: {item['type']}")


def to_plain_dict(item: DictItem) -> dict[str, Any]:
    """Convert a TypedDict item into a plain dict for provider SDK payloads."""
    result: dict[str, Any]
    if is_text_item(item):
        result = {"type": "text", "text": item["text"]}
    elif is_tool_call_item(item):
        result = {
            "type": "tool_call",
            "id": item["id"],
            "name": item["name"],
            "arguments": item["arguments"],
        }
    elif is_tool_result_item(item):
        result = {
            "type": "tool_result",
            "tool_call_id": item["tool_call_id"],
            "content": item["content"],
        }
    elif is_image_item(item):
        result = {
            "type": "image_url",
            "image_url": dict(item["image_url"]),
        }
    elif is_audio_item(item):
        result = {
            "type": "input_audio",
            "input_audio": dict(item["input_audio"]),
        }
    elif is_file_item(item):
        result = {
            "type": "file",
            "file": dict(item["file"]),
        }
    else:
        raise AssertionError(f"Unsupported item type: {item['type']}")
    if "cache_control" in item:
        result["cache_control"] = dict(item["cache_control"])
    return result


@overload
def with_last_item_cache_control(
    message: SystemMessage, cache_control: CacheControl
) -> SystemMessage: ...


@overload
def with_last_item_cache_control(
    message: UserMessage, cache_control: CacheControl
) -> UserMessage: ...


@overload
def with_last_item_cache_control(
    message: AssistantMessage, cache_control: CacheControl
) -> AssistantMessage: ...


@overload
def with_last_item_cache_control(
    message: ToolMessage, cache_control: CacheControl
) -> ToolMessage: ...


def with_last_item_cache_control(
    message: Message, cache_control: CacheControl
) -> Message:
    """Return a copy of *message* with cache metadata on its final item."""
    if message[0] == "system":
        items = list(message[1])
        items[-1] = with_cache_control(items[-1], cache_control)
        return ("system", items)
    if message[0] == "user":
        items = list(message[1])
        items[-1] = with_cache_control(items[-1], cache_control)
        return ("user", items)
    if message[0] == "assistant":
        items = list(message[1])
        items[-1] = with_cache_control(items[-1], cache_control)
        return ("assistant", items)
    items = list(message[1])
    items[-1] = with_cache_control(items[-1], cache_control)
    return ("tool", items)


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


class Store(msgspec.Struct):
    """Mutable metadata populated after the stream is exhausted."""

    usage: Usage | None = None
    cost: Cost | None = None
    provider_cost: float | None = None


StreamResult = tuple[AsyncIterator[StreamItem], Store]
"""Return type of ``Provider.stream()`` and ``YLLMClient.stream()``."""


# ---------------------------------------------------------------------------
# Raw chunk hook
# ---------------------------------------------------------------------------

RawChunkHook = Callable[[Any], None]
"""Callback invoked with every raw provider chunk before yuullm processes it.

The chunk type depends on the provider:
- OpenAI: ``openai.types.chat.ChatCompletionChunk``
- Anthropic: an event object with ``.type`` attribute

This is the escape hatch for consumers who need provider-level visibility
without abandoning yuullm's streaming abstraction.
"""


def on_tool_call_name(name: str, callback: Callable[[int], None]) -> RawChunkHook:
    """Helper hook: fires *callback(index)* when a tool call's name matches.

    Works with both OpenAI and Anthropic raw chunks.  The callback is
    invoked at most once per tool-call index.

    If the LLM emits the name after the arguments (unlikely but possible),
    the notification will simply arrive late -- that's the caller's bad luck.

    Parameters
    ----------
    name : str
        The tool / function name to watch for.
    callback : Callable[[int], None]
        Called with the tool-call index the first time *name* is seen.
    """
    seen: set[int] = set()

    def hook(chunk: Any) -> None:
        # --- OpenAI path ---
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = choices[0].delta if choices else None
            tc_deltas = getattr(delta, "tool_calls", None) if delta else None
            if tc_deltas:
                for tc_delta in tc_deltas:
                    idx = tc_delta.index
                    if idx in seen:
                        continue
                    fn = getattr(tc_delta, "function", None)
                    if fn and getattr(fn, "name", None) == name:
                        seen.add(idx)
                        callback(idx)
            return

        # --- Anthropic path ---
        event_type = getattr(chunk, "type", None)
        if event_type == "content_block_start":
            block = getattr(chunk, "content_block", None)
            if block and getattr(block, "type", None) == "tool_use":
                idx = getattr(chunk, "index", -1)
                if idx not in seen and getattr(block, "name", None) == name:
                    seen.add(idx)
                    callback(idx)

    return hook

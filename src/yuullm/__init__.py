"""yuullm -- Unified streaming LLM interface.

Public API re-exports for convenient access::

    import yuullm

    client = yuullm.YLLMClient(...)
    messages = [
        yuullm.system("You are helpful."),
        yuullm.user("What is 2+2?"),
    ]
    stream, store = await client.stream(messages)
"""

from .cache_config import CacheConfig, ConstantRate, TrafficEstimator
from .client import YLLMClient
from .pricing import PriceCalculator
from .provider import Provider
from .types import (
    AudioItem,
    AssistantItem,
    CacheControl,
    Cost,
    DictItem,
    FileItem,
    History,
    ImageItem,
    Item,
    Message,
    ProviderModel,
    RawChunkHook,
    Reasoning,
    Response,
    ResponseItem,
    StreamItem,
    StreamResult,
    Store,
    TextItem,
    Tick,
    ToolArguments,
    ToolCall,
    ToolCallItem,
    ToolOutput,
    ToolOutputItem,
    ToolResultContent,
    ToolResultContentItem,
    ToolResultItem,
    Usage,
    coerce_tool_output,
    coerce_tool_output_item,
    parse_tool_arguments,
    render_item_text,
    render_message_text,
    # Helper functions for constructing messages
    assistant,
    system,
    tool,
    tool_arguments,
    tool_call_item,
    tool_result,
    user,
    # Hook helpers
    on_tool_call_name,
)

# Lazy import to avoid hard dependency on provider SDKs at import time
from . import providers

__all__ = [
    # Client
    "YLLMClient",
    # Provider protocol
    "Provider",
    # Pricing
    "PriceCalculator",
    # Cache config
    "CacheConfig",
    "ConstantRate",
    "TrafficEstimator",
    # Stream items
    "Reasoning",
    "ToolCall",
    "Response",
    "Tick",
    "StreamItem",
    "StreamResult",
    "Store",
    # Content item types (TypedDict)
    "Item",
    "DictItem",
    "AssistantItem",
    "ResponseItem",
    "ToolCallItem",
    "ToolArguments",
    "ToolOutputItem",
    "ToolOutput",
    "ToolResultContentItem",
    "ToolResultContent",
    "ToolResultItem",
    "TextItem",
    "ImageItem",
    "AudioItem",
    "CacheControl",
    "FileItem",
    # Message types & helpers
    "Message",
    "History",
    "ProviderModel",
    "system",
    "user",
    "assistant",
    "tool",
    "tool_result",
    "tool_call_item",
    "coerce_tool_output_item",
    "coerce_tool_output",
    "parse_tool_arguments",
    "tool_arguments",
    "render_item_text",
    "render_message_text",
    # Usage & Cost
    "Usage",
    "Cost",
    # Hook types & helpers
    "RawChunkHook",
    "on_tool_call_name",
    # Providers sub-package
    "providers",
]

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

from .client import YLLMClient
from .pricing import PriceCalculator
from .provider import Provider
from .types import (
    AudioItem,
    Cost,
    DictItem,
    FileItem,
    History,
    ImageItem,
    Item,
    Message,
    Reasoning,
    Response,
    StreamItem,
    StreamResult,
    Store,
    TextItem,
    ToolCall,
    ToolCallItem,
    ToolResultItem,
    Usage,
    # Helper functions for constructing messages
    assistant,
    system,
    tool,
    user,
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
    # Stream items
    "Reasoning",
    "ToolCall",
    "Response",
    "StreamItem",
    "StreamResult",
    "Store",
    # Content item types (TypedDict)
    "Item",
    "DictItem",
    "ToolCallItem",
    "ToolResultItem",
    "TextItem",
    "ImageItem",
    "AudioItem",
    "FileItem",
    # Message types & helpers
    "Message",
    "History",
    "system",
    "user",
    "assistant",
    "tool",
    # Usage & Cost
    "Usage",
    "Cost",
    # Providers sub-package
    "providers",
]

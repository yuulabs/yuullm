"""yuullm – Unified streaming LLM interface.

Public API re-exports for convenient access::

    import yuullm

    client = yuullm.YLLMClient(...)
    stream, store = await client.stream(messages)
"""

from .client import YLLMClient
from .pricing import PriceCalculator
from .provider import Provider
from .types import (
    AssistantMessage,
    Cost,
    Message,
    Reasoning,
    Response,
    StreamItem,
    StreamResult,
    Store,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
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
    # Messages
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "Message",
    # Tool spec
    "ToolSpec",
    # Usage & Cost
    "Usage",
    "Cost",
    # Providers sub-package
    "providers",
]

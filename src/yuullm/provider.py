"""Provider protocol -- the contract every LLM backend must satisfy."""

from __future__ import annotations

from typing import Any, Protocol

from .types import Message, RawChunkHook, StreamResult


class Provider(Protocol):
    """Unified interface for LLM providers.

    Each provider combines two orthogonal concepts:

    - **api_type**: the wire protocol used (e.g. ``"openai-chat-completion"``,
      ``"openai-responses"``, ``"anthropic-messages"``).
    - **provider**: the vendor / supplier name (e.g. ``"openai"``,
      ``"deepseek"``, ``"openrouter"``, ``"anthropic"``).

    Implementors must supply :meth:`stream` which returns an async iterator
    of :class:`StreamItem` together with a mutable *store* dict.  After the
    iterator is exhausted the store will contain at least ``"usage"``
    (:class:`Usage`).  If the provider can report cost directly (e.g.
    OpenRouter) it should also set ``"provider_cost"`` to a ``float``.

    Tools are passed as ``list[dict]`` -- raw json_schema dicts, exactly
    as produced by ``yuutools.ToolManager.specs()``.  No ToolSpec class
    needed.
    """

    @property
    def api_type(self) -> str:
        """Wire protocol identifier.

        One of ``"openai-chat-completion"``, ``"openai-responses"``,
        ``"anthropic-messages"``.
        """
        ...

    @property
    def provider(self) -> str:
        """Vendor / supplier name (e.g. ``"openai"``, ``"deepseek"``)."""
        ...

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        on_raw_chunk: RawChunkHook | None = None,
        **kwargs,
    ) -> StreamResult:
        """Start a streaming completion.

        Returns
        -------
        iterator : AsyncIterator[StreamItem]
            Yields ``Reasoning``, ``ToolCall``, or ``Response`` fragments.
        store : dict
            Mutable dict that will be populated with:
            - ``"usage"``: :class:`Usage` (after iterator exhausted)
            - ``"provider_cost"``: ``float | None`` (if provider returns cost)
        """
        ...

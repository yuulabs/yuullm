"""Provider protocol – the contract every LLM backend must satisfy."""

from __future__ import annotations

from typing import Protocol

from .types import Message, StreamItem, StreamResult, ToolSpec


class Provider(Protocol):
    """Unified interface for LLM providers.

    Implementors must supply :meth:`stream` which returns an async iterator
    of :class:`StreamItem` together with a mutable *store* dict.  After the
    iterator is exhausted the store will contain at least ``"usage"``
    (:class:`Usage`).  If the provider can report cost directly (e.g.
    OpenRouter) it should also set ``"provider_cost"`` to a ``float``.
    """

    @property
    def name(self) -> str:
        """Short identifier for this provider (e.g. ``"openai"``)."""
        ...

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
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

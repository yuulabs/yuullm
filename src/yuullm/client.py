"""YLLMClient – user-facing entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator

from .pricing import PriceCalculator
from .provider import Provider
from .types import (
    Cost,
    Message,
    StreamItem,
    StreamResult,
    ToolSpec,
    Usage,
)


class YLLMClient:
    """Unified LLM client.

    Wraps a :class:`Provider` and an optional :class:`PriceCalculator`,
    exposing a simple ``stream()`` method that returns standardised
    ``StreamItem`` objects and populates a *store* dict with ``Usage``
    and ``Cost`` after the stream is exhausted.
    """

    def __init__(
        self,
        provider: Provider,
        default_model: str,
        tools: list[ToolSpec] | None = None,
        price_calculator: PriceCalculator | None = None,
    ) -> None:
        self.provider = provider
        self.default_model = default_model
        self.tools = tools
        self.price_calculator = price_calculator

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> StreamResult:
        """Start a streaming completion.

        After the returned async iterator is fully consumed the *store*
        dict will contain:

        - ``"usage"`` – :class:`Usage`
        - ``"cost"``  – :class:`Cost` | ``None``
        """
        effective_model = model or self.default_model
        effective_tools = tools if tools is not None else self.tools

        iterator, store = await self.provider.stream(
            messages,
            model=effective_model,
            tools=effective_tools,
            **kwargs,
        )

        wrapped = self._wrap_iterator(iterator, store)
        return wrapped, store

    async def _wrap_iterator(
        self,
        iterator: AsyncIterator[StreamItem],
        store: dict,
    ) -> AsyncIterator[StreamItem]:
        """Yield items from the provider, then compute cost."""
        async for item in iterator:
            yield item

        # After stream is exhausted, compute cost
        usage: Usage | None = store.get("usage")
        if usage is not None and self.price_calculator is not None:
            provider_cost: float | None = store.get("provider_cost")
            cost: Cost | None = self.price_calculator.calculate(
                usage, provider_cost=provider_cost
            )
            store["cost"] = cost
        else:
            store.setdefault("cost", None)

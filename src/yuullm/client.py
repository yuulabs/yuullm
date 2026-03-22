"""YLLMClient -- user-facing entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from .cache_config import CacheConfig
from .pricing import PriceCalculator
from .provider import Provider
from .types import (
    Cost,
    Message,
    RawChunkHook,
    Store,
    StreamItem,
    StreamResult,
    Usage,
)


class YLLMClient:
    """Unified LLM client.

    Wraps a :class:`Provider` and an optional :class:`PriceCalculator`,
    exposing a simple ``stream()`` method that returns standardised
    ``StreamItem`` objects and populates a *store* dict with ``Usage``
    and ``Cost`` after the stream is exhausted.

    Tools are passed as ``list[dict]`` -- raw json_schema dicts from
    yuutools (or any other source).  No ToolSpec class needed.

    Parameters
    ----------
    auto_prompt_caching : bool
        When ``True`` (default), the provider receives *cache_config*
        and *price_calculator* so it can automatically inject
        vendor-specific cache markers.  Set to ``False`` to disable.
    cache_config : CacheConfig | None
        Business-level caching intent passed to the provider.
        Defaults to ``CacheConfig()`` when *auto_prompt_caching* is
        ``True`` and no explicit config is given.
    """

    def __init__(
        self,
        provider: Provider,
        default_model: str,
        tools: list[dict[str, Any]] | None = None,
        price_calculator: PriceCalculator | None = None,
        *,
        auto_prompt_caching: bool = True,
        cache_config: CacheConfig | None = None,
    ) -> None:
        self.default_model = default_model
        self.tools = tools
        self.price_calculator = price_calculator

        # Wire up cache_config into the provider if it supports it
        if auto_prompt_caching:
            effective_config = cache_config or CacheConfig()
            _inject_cache_config(provider, effective_config, price_calculator)

        self.provider = provider

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_raw_chunk: RawChunkHook | None = None,
        **kwargs,
    ) -> StreamResult:
        """Start a streaming completion.

        After the returned async iterator is fully consumed the *store*
        dict will contain:

        - ``"usage"`` -- :class:`Usage`
        - ``"cost"``  -- :class:`Cost` | ``None``
        """
        effective_model = model or self.default_model
        effective_tools = tools if tools is not None else self.tools

        iterator, store = await self.provider.stream(
            messages,
            model=effective_model,
            tools=effective_tools,
            on_raw_chunk=on_raw_chunk,
            **kwargs,
        )

        wrapped = self._wrap_iterator(iterator, store)
        return wrapped, store

    async def _wrap_iterator(
        self,
        iterator: AsyncIterator[StreamItem],
        store: Store,
    ) -> AsyncIterator[StreamItem]:
        """Yield items from the provider, then compute cost."""
        async for item in iterator:
            yield item

        # After stream is exhausted, compute cost
        usage: Usage | None = store.usage
        if usage is not None and self.price_calculator is not None:
            provider_cost: float | None = store.provider_cost
            cost: Cost | None = self.price_calculator.calculate(
                usage, provider_cost=provider_cost
            )
            store.cost = cost


def _inject_cache_config(
    provider: Provider,
    config: CacheConfig,
    price_calc: PriceCalculator | None,
) -> None:
    """Configure prompt caching on providers that support it.

    Providers opt in by exposing a public ``configure_cache`` method.
    No-op for providers that don't (e.g. DeepSeek via
    OpenAIChatCompletionProvider with provider_name="deepseek").
    """
    configure = getattr(provider, "configure_cache", None)
    if configure is not None:
        configure(config, price_calc)

"""OpenRouter provider -- OpenAI-compatible wire format + sub-vendor cache routing.

OpenRouter proxies to upstream vendors (Anthropic, Google, OpenAI, etc.).
Caching requires manually injecting vendor-specific ``cache_control``
markers; otherwise upstream charges full price.

This provider inherits the OpenAI chat-completion wire format and adds
cache annotation logic driven by :class:`CacheConfig` and
:class:`PriceCalculator`.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from ..cache_config import CacheConfig
from ..pricing import PriceCalculator
from ..types import Item, Message, RawChunkHook, StreamResult
from .openai import OpenAIChatCompletionProvider


class OpenRouterProvider(OpenAIChatCompletionProvider):
    """OpenRouter = OpenAI wire format + sub-vendor cache routing.

    Parameters
    ----------
    api_key : str
        OpenRouter API key.
    cache_config : CacheConfig | None
        Business-level caching intent.  ``None`` disables caching.
    price_calculator : PriceCalculator | None
        Used to decide optimal TTL tier via cost estimation.
    """

    def __init__(
        self,
        api_key: str,
        *,
        cache_config: CacheConfig | None = None,
        price_calculator: PriceCalculator | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            provider_name="openrouter",
        )
        self._cache_config = cache_config
        self._price_calc = price_calculator

    def configure_cache(
        self,
        cache_config: CacheConfig,
        price_calculator: PriceCalculator | None = None,
    ) -> None:
        """Enable sub-vendor cache routing."""
        if self._cache_config is None:
            self._cache_config = cache_config
        if self._price_calc is None and price_calculator is not None:
            self._price_calc = price_calculator

    # ------------------------------------------------------------------
    # Stream override: inject cache markers before calling parent
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        on_raw_chunk: RawChunkHook | None = None,
        **kwargs,
    ) -> StreamResult:
        if self._cache_config is not None:
            messages = self._apply_cache(messages, model, tools)
            tools = self._apply_cache_tools(tools, model)
        return await super().stream(
            messages, model=model, tools=tools, on_raw_chunk=on_raw_chunk, **kwargs
        )

    # ------------------------------------------------------------------
    # Message conversion: preserve cache_control in content arrays
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        """Convert messages, preserving ``cache_control`` on content blocks.

        OpenRouter forwards ``cache_control`` to upstream vendors, but it
        must appear on individual content blocks (array format), not on a
        plain string.  When no ``cache_control`` is present we fall back to
        the compact string format.
        """
        result: list[dict] = []
        for role, items in messages:
            if role == "system":
                has_cc = any("cache_control" in it for it in items)
                if has_cc:
                    result.append({"role": "system", "content": list(items)})
                else:
                    text = "".join(
                        it["text"] for it in items if it.get("type") == "text"
                    )
                    result.append({"role": "system", "content": text})

            elif role == "user":
                has_cc = any("cache_control" in it for it in items)
                if has_cc:
                    result.append({"role": "user", "content": list(items)})
                elif all(it.get("type") == "text" for it in items):
                    result.append(
                        {"role": "user", "content": "".join(it["text"] for it in items)}
                    )
                else:
                    result.append({"role": "user", "content": list(items)})

            elif role == "assistant":
                entry: dict[str, Any] = {"role": "assistant"}
                has_cc = any("cache_control" in it for it in items)
                text_parts: list = []
                tool_calls: list[dict] = []
                for it in items:
                    if it.get("type") == "text":
                        text_parts.append(it)
                    elif it.get("type") == "tool_call":
                        tool_calls.append(
                            {
                                "id": it["id"],
                                "type": "function",
                                "function": {
                                    "name": it["name"],
                                    "arguments": it.get("arguments", "{}"),
                                },
                            }
                        )
                if text_parts:
                    if has_cc:
                        entry["content"] = text_parts
                    else:
                        entry["content"] = "".join(p["text"] for p in text_parts)
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)

            elif role == "tool":
                for it in items:
                    if it.get("type") == "tool_result":
                        content = it.get("content", "")
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": it["tool_call_id"],
                                "content": content,
                            }
                        )

        return result

    # ------------------------------------------------------------------
    # Sub-vendor cache routing (private)
    # ------------------------------------------------------------------

    def _apply_cache(
        self,
        messages: list[Message],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> list[Message]:
        if model.startswith("anthropic/"):
            return self._cache_anthropic(messages, model, tools)
        if model.startswith("google/"):
            # OpenRouter translates cache_control breakpoints to Gemini's
            # cachedContent API transparently.  Multiple breakpoints are
            # safe -- OpenRouter uses only the last one for Gemini.
            cc: dict = {"type": "ephemeral"}
            return self._mark_breakpoints(messages, cc)
        return messages

    # --- Anthropic via OpenRouter -------------------------------------------

    # Anthropic cache-write cost multipliers relative to input_mtok
    _ANTHROPIC_WRITE_MUL: dict[int, float] = {300: 1.25, 3600: 2.0}

    def _cache_anthropic(
        self,
        messages: list[Message],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> list[Message]:
        """Inject ``cache_control`` breakpoints for Anthropic models.

        Strategy:
        - System message last block gets a breakpoint (stable prefix).
        - The last item of the last "prefix" message (messages that
          won't change between calls) gets a breakpoint.
        - TTL is chosen by cost estimation when PriceCalculator is available.
        - At most 4 breakpoints total (Anthropic limit).
        """
        ttl = self._pick_ttl_anthropic(model, messages, tools)
        cc = self._make_cache_control(ttl)
        return self._mark_breakpoints(messages, cc)

    def _pick_ttl_anthropic(
        self,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        """Choose the best TTL tier (300s or 3600s) by cost estimation."""
        if self._price_calc is None or self._cache_config is None:
            return 300  # default: 5-minute ephemeral

        base = self._price_calc.get_base_prices("openrouter", model)
        if base is None:
            return 300

        input_price = base.get("input_mtok", 0)
        if input_price <= 0:
            return 300

        n = self._estimate_prefix_tokens(messages, tools)
        if n <= 0:
            return 300

        now = time.monotonic()
        best_ttl, best_cost = 300, float("inf")

        for ttl, mul in self._ANTHROPIC_WRITE_MUL.items():
            reads = self._cache_config.traffic.expected_requests(now, ttl)
            cost = self._price_calc.estimate(
                {**base, "cache_write_mtok": input_price * mul},
                cache_write_tokens=n,
                cache_read_tokens=int(n * reads),
            )
            if cost.total_cost < best_cost:
                best_ttl, best_cost = ttl, cost.total_cost

        return best_ttl

    @staticmethod
    def _make_cache_control(ttl: int) -> dict:
        """Build the ``cache_control`` dict for a given TTL."""
        if ttl <= 300:
            return {"type": "ephemeral"}
        return {"type": "ephemeral", "ttl": ttl}

    @staticmethod
    def _mark_breakpoints(
        messages: list[Message], cc: dict
    ) -> list[Message]:
        """Add cache_control to system-last-block and prefix boundary.

        We deep-copy only the messages we mutate.
        """
        result: list[Message] = []
        breakpoints_used = 0
        max_breakpoints = 4

        for role, items in messages:
            if role == "system" and items and breakpoints_used < max_breakpoints:
                # Mark last block of system message
                items = [copy.copy(it) for it in items]
                items[-1] = {**items[-1], "cache_control": cc}
                breakpoints_used += 1
                result.append((role, items))
            else:
                result.append((role, items))

        # Mark the last item of the last non-system, non-tool message
        # before the final user turn (the "prefix boundary").
        # Find the second-to-last user message as the prefix boundary.
        if breakpoints_used < max_breakpoints and len(result) >= 2:
            # Walk backwards to find a good prefix boundary:
            # skip the last message (it's the new user turn)
            for i in range(len(result) - 2, -1, -1):
                r, its = result[i]
                if r == "system":
                    continue  # already marked
                if its:
                    its = list(its)
                    its[-1] = {**its[-1], "cache_control": cc}
                    result[i] = (r, its)
                    breakpoints_used += 1
                    break

        return result

    @staticmethod
    def _estimate_prefix_tokens(
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        """Rough token estimate for the cacheable prefix.

        Uses a simple heuristic: 4 chars ≈ 1 token.
        Counts system messages + tools + all messages except the last user turn.
        """
        char_count = 0

        for i, (role, items) in enumerate(messages):
            if role == "system":
                for it in items:
                    char_count += len(it.get("text", ""))
            elif i < len(messages) - 1:
                # All messages before the last one are prefix
                for it in items:
                    if it.get("type") == "text":
                        char_count += len(it.get("text", ""))
                    else:
                        char_count += 100  # rough estimate for non-text blocks

        if tools:
            import json
            char_count += len(json.dumps(tools))

        return char_count // 4

    # --- Tools cache annotation ---------------------------------------------

    def _apply_cache_tools(
        self,
        tools: list[dict[str, Any]] | None,
        model: str,
    ) -> list[dict[str, Any]] | None:
        """Mark the last tool with cache_control for supported vendors."""
        if tools is None or not tools:
            return tools

        if model.startswith("anthropic/"):
            cc = self._make_cache_control(self._pick_ttl_anthropic(model, [], tools))
        elif model.startswith("google/"):
            cc = {"type": "ephemeral"}
        else:
            return tools

        tools = [copy.copy(t) for t in tools]
        tools[-1] = {**tools[-1], "cache_control": cc}
        return tools

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
from collections.abc import Sequence
from typing import Any

from ..cache_config import CacheConfig
from ..pricing import PriceCalculator
from ..types import (
    CacheControl,
    Message,
    RawChunkHook,
    StreamResult,
    is_text_item,
    is_tool_call_item,
    is_tool_result_item,
    to_plain_dict,
    with_last_item_cache_control,
)
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
    default_headers : dict[str, str] | None
        Additional headers to send with every request. Useful for
        application identification (e.g., ``{"User-Agent": "my-app/1.0"}``).
    """

    def __init__(
        self,
        api_key: str,
        *,
        cache_config: CacheConfig | None = None,
        price_calculator: PriceCalculator | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            provider_name="openrouter",
            default_headers=default_headers,
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
    def _convert_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Convert messages, preserving ``cache_control`` on content blocks.

        OpenRouter forwards ``cache_control`` to upstream vendors, but it
        must appear on individual content blocks (array format), not on a
        plain string.  When no ``cache_control`` is present we fall back to
        the compact string format.
        """
        result: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                items = message.content
                has_cc = any("cache_control" in it for it in items)
                if has_cc:
                    result.append(
                        {
                            "role": "system",
                            "content": [to_plain_dict(item) for item in items],
                            **message.provider_extra,
                        }
                    )
                else:
                    text = "".join(it["text"] for it in items if is_text_item(it))
                    result.append(
                        {
                            "role": "system",
                            "content": text,
                            **message.provider_extra,
                        }
                    )

            elif message.role == "user":
                items = message.content
                has_cc = any("cache_control" in it for it in items)
                user_text_parts: list[str] = []
                all_text = True
                for item in items:
                    if is_text_item(item):
                        user_text_parts.append(item["text"])
                    else:
                        all_text = False
                        break
                if has_cc:
                    result.append(
                        {
                            "role": "user",
                            "content": [to_plain_dict(item) for item in items],
                            **message.provider_extra,
                        }
                    )
                elif all_text:
                    result.append(
                        {
                            "role": "user",
                            "content": "".join(user_text_parts),
                            **message.provider_extra,
                        }
                    )
                else:
                    result.append(
                        {
                            "role": "user",
                            "content": [to_plain_dict(item) for item in items],
                            **message.provider_extra,
                        }
                    )

            elif message.role == "assistant":
                items = message.content
                entry: dict[str, Any] = {
                    "role": "assistant",
                    **message.provider_extra,
                }
                has_cc = any("cache_control" in it for it in items)
                text_blocks: list[dict[str, Any]] = []
                assistant_text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for item in items:
                    if is_text_item(item):
                        text_blocks.append(to_plain_dict(item))
                        assistant_text_parts.append(item["text"])
                    elif is_tool_call_item(item):
                        tool_calls.append(
                            {
                                "id": item["id"],
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": item["arguments"],
                                },
                            }
                        )
                if text_blocks:
                    if has_cc:
                        entry["content"] = text_blocks
                    else:
                        entry["content"] = "".join(assistant_text_parts)
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)

            else:
                items = message.content
                for item in items:
                    if not is_tool_result_item(item):
                        raise TypeError("tool messages only accept tool-result items")
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": item["tool_call_id"],
                            "content": item["content"],
                            **message.provider_extra,
                        }
                    )

        return result

    # ------------------------------------------------------------------
    # Sub-vendor cache routing (private)
    # ------------------------------------------------------------------

    def _apply_cache(
        self,
        messages: Sequence[Message],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> list[Message]:
        if model.startswith("anthropic/"):
            return self._cache_anthropic(messages, model, tools)
        if model.startswith("google/"):
            # OpenRouter translates cache_control breakpoints to Gemini's
            # cachedContent API transparently.  Multiple breakpoints are
            # safe -- OpenRouter uses only the last one for Gemini.
            cc: CacheControl = {"type": "ephemeral"}
            return self._mark_breakpoints(messages, cc)
        return list(messages)

    # --- Anthropic via OpenRouter -------------------------------------------

    # Anthropic cache-write cost multipliers relative to input_mtok
    _ANTHROPIC_WRITE_MUL: dict[int, float] = {300: 1.25, 3600: 2.0}

    def _cache_anthropic(
        self,
        messages: Sequence[Message],
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
        messages: Sequence[Message],
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
    def _make_cache_control(ttl: int) -> CacheControl:
        """Build the ``cache_control`` dict for a given TTL."""
        if ttl <= 300:
            return {"type": "ephemeral"}
        return {"type": "ephemeral", "ttl": ttl}

    @staticmethod
    def _mark_breakpoints(
        messages: Sequence[Message], cc: CacheControl
    ) -> list[Message]:
        """Add cache_control to system-last-block and prefix boundary.

        We deep-copy only the messages we mutate.
        """
        result: list[Message] = []
        breakpoints_used = 0
        max_breakpoints = 4

        for message in messages:
            if (
                message.role == "system"
                and message.content
                and breakpoints_used < max_breakpoints
            ):
                breakpoints_used += 1
                result.append(with_last_item_cache_control(message, cc))
            else:
                result.append(message)

        # Mark the last item of the last non-system, non-tool message
        # before the final user turn (the "prefix boundary").
        # Find the second-to-last user message as the prefix boundary.
        if breakpoints_used < max_breakpoints and len(result) >= 2:
            # Walk backwards to find a good prefix boundary:
            # skip the last message (it's the new user turn)
            for i in range(len(result) - 2, -1, -1):
                message = result[i]
                if message.role == "system":
                    continue  # already marked
                if message.content:
                    result[i] = with_last_item_cache_control(message, cc)
                    breakpoints_used += 1
                    break

        return result

    @staticmethod
    def _estimate_prefix_tokens(
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        """Rough token estimate for the cacheable prefix.

        Uses a simple heuristic: 4 chars ≈ 1 token.
        Counts system messages + tools + all messages except the last user turn.
        """
        char_count = 0

        for i, message in enumerate(messages):
            if message.role == "system":
                for it in message.content:
                    if is_text_item(it):
                        char_count += len(it["text"])
            elif i < len(messages) - 1:
                # All messages before the last one are prefix
                for it in message.content:
                    if is_text_item(it):
                        char_count += len(it["text"])
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

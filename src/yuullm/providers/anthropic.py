"""Anthropic Messages API provider implementation.

This provider uses the Anthropic ``/v1/messages`` endpoint with streaming.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from ..cache_config import CacheConfig
from ..pricing import PriceCalculator
from ..types import (
    Message,
    RawChunkHook,
    Reasoning,
    Response,
    StreamItem,
    StreamResult,
    Tick,
    ToolCall,
    Usage,
)


class AnthropicMessagesProvider:
    """Provider for the Anthropic Messages API (``/v1/messages``).

    Parameters
    ----------
    api_key : str | None
        API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    base_url : str | None
        Override the base URL (e.g. for a proxy).
    provider_name : str
        Vendor / supplier identifier used in :class:`Usage` and pricing
        lookups.  Defaults to ``"anthropic"``.
    cache_config : CacheConfig | None
        When set, enables automatic cache breakpoint injection.
    price_calculator : PriceCalculator | None
        Used to decide optimal TTL tier via cost estimation.
    """

    # Anthropic cache-write cost multipliers relative to input_mtok
    _WRITE_MUL: dict[int, float] = {300: 1.25, 3600: 2.0}

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        provider_name: str = "anthropic",
        cache_config: CacheConfig | None = None,
        price_calculator: PriceCalculator | None = None,
    ) -> None:
        kwargs: dict = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._provider_name = provider_name
        self._cache_config = cache_config
        self._price_calc = price_calculator

    @property
    def api_type(self) -> str:
        return "anthropic-messages"

    @property
    def provider(self) -> str:
        return self._provider_name

    # ------------------------------------------------------------------
    # Message conversion: (role, items) tuples -> Anthropic API format
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """Separate system messages (Anthropic uses a top-level param).

        Returns the system content as a list of content blocks (for
        cache_control support) or a plain string, plus the remaining
        messages.
        """
        system_blocks: list[dict] | None = None
        rest: list[Message] = []
        for role, items in messages:
            if role == "system":
                system_blocks = list(items)
            else:
                rest.append((role, items))
        return system_blocks, rest

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        """Convert (role, items) tuples to Anthropic messages format.

        All items are dicts with a ``type`` key. Dispatch by type.
        """
        result: list[dict] = []
        for role, items in messages:
            if role == "user":
                # If all items are text-only, use simple string content
                if all(it.get("type") == "text" for it in items):
                    result.append(
                        {
                            "role": "user",
                            "content": "".join(it["text"] for it in items),
                        }
                    )
                else:
                    result.append({"role": "user", "content": list(items)})

            elif role == "assistant":
                content_blocks: list[dict] = []
                for it in items:
                    if it.get("type") == "text":
                        content_blocks.append(it)
                    elif it.get("type") == "tool_call":
                        args = it.get("arguments", "{}")
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": it["id"],
                                "name": it["name"],
                                "input": json.loads(args)
                                if isinstance(args, str)
                                else args,
                            }
                        )
                result.append({"role": "assistant", "content": content_blocks})

            elif role == "tool":
                tool_results: list[dict] = []
                for it in items:
                    if it.get("type") == "tool_result":
                        raw_content = it.get("content", "")
                        if isinstance(raw_content, list):
                            anthropic_blocks: list[dict] = []
                            for block in raw_content:
                                if block.get("type") == "text":
                                    anthropic_blocks.append(block)
                                elif block.get("type") == "image_url":
                                    url = block.get("image_url", {}).get("url", "")
                                    if url.startswith("data:"):
                                        header, _, b64 = url.partition(",")
                                        media_type = header.split(":")[1].split(";")[0]
                                        anthropic_blocks.append({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": b64,
                                            },
                                        })
                                    else:
                                        anthropic_blocks.append({
                                            "type": "image",
                                            "source": {"type": "url", "url": url},
                                        })
                            tool_content = anthropic_blocks
                        else:
                            tool_content = raw_content
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": it["tool_call_id"],
                                "content": tool_content,
                            }
                        )
                if tool_results:
                    result.append({"role": "user", "content": tool_results})

        return result

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict]:
        """Convert json_schema tool dicts to Anthropic format.

        Accepts dicts in OpenAI format (``{"type": "function", "function": {...}}``)
        or bare function dicts (``{"name": ..., ...}``).
        """
        result: list[dict] = []
        for t in tools:
            if t.get("type") == "function":
                # OpenAI format (e.g. from yuutools.ToolManager.specs())
                fn = t["function"]
                result.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {}),
                    }
                )
            else:
                # Bare dict
                result.append(
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get("parameters", {}),
                    }
                )
        return result

    # ------------------------------------------------------------------
    # Streaming
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
        store: dict = {}

        # Inject cache breakpoints if cache_config is set
        if self._cache_config is not None:
            messages = self._inject_cache(messages, model, tools)
            tools = self._inject_cache_tools(tools, model)

        system_blocks, rest = self._extract_system(messages)
        api_messages = self._convert_messages(rest)

        create_kwargs: dict = {
            "model": model,
            "messages": api_messages,
            "max_tokens": kwargs.pop("max_tokens", 8192),
            **kwargs,
        }
        if system_blocks is not None:
            create_kwargs["system"] = system_blocks
        if tools:
            create_kwargs["tools"] = self._convert_tools(tools)

        iterator = self._iterate(create_kwargs, model, store, on_raw_chunk)
        return iterator, store

    # ------------------------------------------------------------------
    # Cache injection (direct Anthropic)
    # ------------------------------------------------------------------

    def _pick_ttl(self, model: str, messages: list[Message], tools: list[dict] | None) -> int:
        """Choose TTL tier by cost estimation. Fallback: 300s."""
        if self._price_calc is None or self._cache_config is None:
            return 300

        base = self._price_calc.get_base_prices(self._provider_name, model)
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

        for ttl, mul in self._WRITE_MUL.items():
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
        if ttl <= 300:
            return {"type": "ephemeral"}
        return {"type": "ephemeral", "ttl": ttl}

    def _inject_cache(
        self, messages: list[Message], model: str, tools: list[dict] | None
    ) -> list[Message]:
        """Add cache_control breakpoints to system last block and prefix boundary."""
        ttl = self._pick_ttl(model, messages, tools)
        cc = self._make_cache_control(ttl)

        result: list[Message] = []
        breakpoints = 0

        for role, items in messages:
            if role == "system" and items and breakpoints < 4:
                items = [copy.copy(it) for it in items]
                items[-1] = {**items[-1], "cache_control": cc}
                breakpoints += 1
                result.append((role, items))
            else:
                result.append((role, items))

        # Mark prefix boundary (last message before the final user turn)
        if breakpoints < 4 and len(result) >= 2:
            for i in range(len(result) - 2, -1, -1):
                r, its = result[i]
                if r == "system":
                    continue
                if its:
                    its = list(its)
                    its[-1] = {**its[-1], "cache_control": cc}
                    result[i] = (r, its)
                    break

        return result

    def _inject_cache_tools(
        self, tools: list[dict] | None, model: str
    ) -> list[dict] | None:
        """Mark last tool with cache_control."""
        if not tools:
            return tools
        ttl = self._pick_ttl(model, [], tools)
        cc = self._make_cache_control(ttl)
        tools = list(tools)
        tools[-1] = {**tools[-1], "cache_control": cc}
        return tools

    @staticmethod
    def _estimate_prefix_tokens(
        messages: list[Message], tools: list[dict] | None
    ) -> int:
        """Rough token estimate: 4 chars ≈ 1 token."""
        char_count = 0
        for i, (role, items) in enumerate(messages):
            if role == "system":
                for it in items:
                    char_count += len(it.get("text", ""))
            elif i < len(messages) - 1:
                for it in items:
                    if it.get("type") == "text":
                        char_count += len(it.get("text", ""))
                    else:
                        char_count += 100
        if tools:
            import json as _json
            char_count += len(_json.dumps(tools))
        return char_count // 4

    async def _iterate(
        self,
        create_kwargs: dict,
        model: str,
        store: dict,
        on_raw_chunk: RawChunkHook | None = None,
    ) -> AsyncIterator[StreamItem]:
        # Accumulate tool calls by block index
        tool_calls_acc: dict[int, dict] = {}
        current_block_index: int = -1
        current_block_type: str = ""
        request_id: str | None = None

        async with self._client.messages.stream(**create_kwargs) as stream:
            async for event in stream:
                if on_raw_chunk is not None:
                    on_raw_chunk(event)

                match event.type:
                    case "message_start":
                        msg = event.message
                        request_id = msg.id
                    case "content_block_start":
                        current_block_index = event.index
                        block = event.content_block
                        current_block_type = block.type
                        if block.type == "tool_use":
                            tool_calls_acc[current_block_index] = {
                                "id": block.id,
                                "name": block.name,
                                "arguments": "",
                            }
                    case "content_block_delta":
                        delta = event.delta
                        match delta.type:
                            case "thinking_delta":
                                yield Reasoning(item=delta.thinking)
                            case "text_delta":
                                yield Response(item={"type": "text", "text": delta.text})
                            case "input_json_delta":
                                if current_block_index in tool_calls_acc:
                                    tool_calls_acc[current_block_index][
                                        "arguments"
                                    ] += delta.partial_json
                                # Keep the consumer loop spinning so
                                # on_raw_chunk side-effects can be flushed.
                                if on_raw_chunk is not None:
                                    yield Tick()
                    case "content_block_stop":
                        if current_block_index in tool_calls_acc:
                            acc = tool_calls_acc.pop(current_block_index)
                            yield ToolCall(
                                id=acc["id"],
                                name=acc["name"],
                                arguments=acc["arguments"],
                            )
                    case "message_delta":
                        pass  # stop_reason, etc.

            # Extract usage from the final message
            final_message = stream.get_final_message()

        store["usage"] = Usage(
            provider=self._provider_name,
            model=final_message.model or model,
            request_id=request_id,
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
            cache_read_tokens=getattr(final_message.usage, "cache_read_input_tokens", 0)
            or 0,
            cache_write_tokens=getattr(
                final_message.usage, "cache_creation_input_tokens", 0
            )
            or 0,
        )
        store.setdefault("provider_cost", None)


# Backward-compatible alias (deprecated)
AnthropicProvider = AnthropicMessagesProvider

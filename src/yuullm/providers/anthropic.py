"""Anthropic Messages API provider implementation.

This provider uses the Anthropic ``/v1/messages`` endpoint with streaming.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic

from ..cache_config import CacheConfig
from ..pricing import PriceCalculator
from ..types import (
    CacheControl,
    Message,
    ProviderModel,
    RawChunkHook,
    Reasoning,
    Response,
    Store,
    StreamItem,
    StreamResult,
    Tick,
    ToolCall,
    Usage,
    is_image_item,
    is_text_item,
    is_tool_call_item,
    is_tool_result_item,
    to_plain_dict,
    with_last_item_cache_control,
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
    default_headers : dict[str, str] | None
        Additional headers to send with every request. Useful for
        application identification (e.g., ``{"User-Agent": "my-app/1.0"}``).
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
        default_headers: dict[str, str] | None = None,
    ) -> None:
        kwargs: dict = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        if default_headers is not None:
            kwargs["default_headers"] = default_headers
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._provider_name = provider_name
        self._cache_config = cache_config
        self._price_calc = price_calculator

    def configure_cache(
        self,
        cache_config: CacheConfig,
        price_calculator: PriceCalculator | None = None,
    ) -> None:
        """Enable automatic cache breakpoint injection."""
        if self._cache_config is None:
            self._cache_config = cache_config
        if self._price_calc is None and price_calculator is not None:
            self._price_calc = price_calculator

    @property
    def api_type(self) -> str:
        return "anthropic-messages"

    @property
    def provider(self) -> str:
        return self._provider_name

    async def list_models(self) -> list[ProviderModel]:
        """Return model IDs exposed by Anthropic's models API."""
        models: list[ProviderModel] = []
        seen: set[str] = set()
        async for item in self._client.models.list(limit=1000):
            if isinstance(item, dict):
                model_id = str(item.get("id", "") or "").strip()
                display_name = str(item.get("display_name", "") or "").strip() or None
            else:
                model_id = str(getattr(item, "id", "") or "").strip()
                display_name = (
                    str(getattr(item, "display_name", "") or "").strip() or None
                )
            if model_id and model_id not in seen:
                seen.add(model_id)
                models.append(ProviderModel(id=model_id, display_name=display_name))
        return models

    # ------------------------------------------------------------------
    # Message conversion: Message structs -> Anthropic API format
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_system(
        messages: Sequence[Message],
    ) -> tuple[list[dict[str, Any]] | None, list[Message]]:
        """Separate system messages (Anthropic uses a top-level param).

        Returns the system content blocks plus the remaining messages.
        """
        system_blocks: list[dict[str, Any]] | None = None
        rest: list[Message] = []
        for message in messages:
            if message.role == "system":
                system_blocks = [to_plain_dict(item) for item in message.content]
            else:
                rest.append(message)
        return system_blocks, rest

    @staticmethod
    def _convert_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Convert yuullm messages to Anthropic messages format.

        All items are dicts with a ``type`` key. Dispatch by type.
        """
        result: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "user":
                items = message.content
                # If all items are text-only, use simple string content
                text_parts: list[str] = []
                all_text = True
                for item in items:
                    if is_text_item(item):
                        text_parts.append(item["text"])
                    else:
                        all_text = False
                        break
                if all_text:
                    result.append(
                        {
                            "role": "user",
                            "content": "".join(text_parts),
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
                content_blocks: list[dict[str, Any]] = []
                for item in items:
                    if is_text_item(item):
                        content_blocks.append(to_plain_dict(item))
                    elif is_tool_call_item(item):
                        args = item["arguments"]
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": item["id"],
                                "name": item["name"],
                                "input": json.loads(args),
                            }
                        )
                result.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                        **message.provider_extra,
                    }
                )

            elif message.role == "tool":
                items = message.content
                tool_results: list[dict[str, Any]] = []
                for item in items:
                    if not is_tool_result_item(item):
                        raise TypeError("tool messages only accept tool-result items")
                    raw_content = item["content"]
                    if isinstance(raw_content, list):
                        anthropic_blocks: list[dict[str, Any]] = []
                        for block in raw_content:
                            if is_text_item(block):
                                anthropic_blocks.append(to_plain_dict(block))
                            elif is_image_item(block):
                                url = block["image_url"].get("url", "")
                                if url.startswith("data:"):
                                    header, _, b64 = url.partition(",")
                                    media_type = header.split(":")[1].split(";")[0]
                                    anthropic_blocks.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": b64,
                                            },
                                        }
                                    )
                                else:
                                    anthropic_blocks.append(
                                        {
                                            "type": "image",
                                            "source": {"type": "url", "url": url},
                                        }
                                    )
                        tool_content: str | list[dict[str, Any]] = anthropic_blocks
                    else:
                        tool_content = raw_content
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": item["tool_call_id"],
                            "content": tool_content,
                        }
                    )
                if tool_results:
                    result.append(
                        {
                            "role": "user",
                            "content": tool_results,
                            **message.provider_extra,
                        }
                    )

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
        store = Store()

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

    def _pick_ttl(
        self, model: str, messages: Sequence[Message], tools: list[dict] | None
    ) -> int:
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
    def _make_cache_control(ttl: int) -> CacheControl:
        if ttl <= 300:
            return {"type": "ephemeral"}
        return {"type": "ephemeral", "ttl": ttl}

    def _inject_cache(
        self, messages: Sequence[Message], model: str, tools: list[dict] | None
    ) -> list[Message]:
        """Add cache_control breakpoints to system last block and prefix boundary."""
        ttl = self._pick_ttl(model, messages, tools)
        cc = self._make_cache_control(ttl)

        result: list[Message] = []
        breakpoints = 0

        for message in messages:
            if message.role == "system" and message.content and breakpoints < 4:
                breakpoints += 1
                result.append(with_last_item_cache_control(message, cc))
            else:
                result.append(message)

        # Mark prefix boundary (last message before the final user turn)
        if breakpoints < 4 and len(result) >= 2:
            for i in range(len(result) - 2, -1, -1):
                message = result[i]
                if message.role == "system":
                    continue
                if message.content:
                    result[i] = with_last_item_cache_control(message, cc)
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
        messages: Sequence[Message], tools: list[dict] | None
    ) -> int:
        """Rough token estimate: 4 chars ≈ 1 token."""
        char_count = 0
        for i, message in enumerate(messages):
            if message.role == "system":
                for it in message.content:
                    if is_text_item(it):
                        char_count += len(it["text"])
            elif i < len(messages) - 1:
                for it in message.content:
                    if is_text_item(it):
                        char_count += len(it["text"])
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
        store: Store,
        on_raw_chunk: RawChunkHook | None = None,
    ) -> AsyncIterator[StreamItem]:
        # Accumulate tool calls by block index
        tool_calls_acc: dict[int, dict[str, str]] = {}
        current_block_index: int = -1
        request_id: str | None = None

        async with self._client.messages.stream(**create_kwargs) as stream:
            async for event in stream:
                if on_raw_chunk is not None:
                    on_raw_chunk(event)

                event_type = getattr(event, "type", None)
                if event_type == "message_start":
                    msg = getattr(event, "message", None)
                    msg_id = getattr(msg, "id", None)
                    if isinstance(msg_id, str):
                        request_id = msg_id
                elif event_type == "content_block_start":
                    index = getattr(event, "index", None)
                    block = getattr(event, "content_block", None)
                    if not isinstance(index, int) or block is None:
                        continue
                    current_block_index = index
                    if getattr(block, "type", None) == "tool_use":
                        block_id = getattr(block, "id", None)
                        block_name = getattr(block, "name", None)
                        if isinstance(block_id, str) and isinstance(block_name, str):
                            tool_calls_acc[current_block_index] = {
                                "id": block_id,
                                "name": block_name,
                                "arguments": "",
                            }
                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "thinking_delta":
                        thinking = getattr(delta, "thinking", None)
                        if isinstance(thinking, str):
                            yield Reasoning(item={"type": "text", "text": thinking})
                    elif delta_type == "text_delta":
                        text = getattr(delta, "text", None)
                        if isinstance(text, str):
                            yield Response(item={"type": "text", "text": text})
                    elif delta_type == "input_json_delta":
                        partial_json = getattr(delta, "partial_json", None)
                        if current_block_index in tool_calls_acc and isinstance(
                            partial_json, str
                        ):
                            tool_calls_acc[current_block_index]["arguments"] += (
                                partial_json
                            )
                        if on_raw_chunk is not None:
                            yield Tick()
                elif event_type == "content_block_stop":
                    acc = tool_calls_acc.pop(current_block_index, None)
                    if acc is not None:
                        yield ToolCall(
                            id=acc["id"],
                            name=acc["name"],
                            arguments=acc["arguments"],
                        )

            # Extract usage from the final message
            final_message = await stream.get_final_message()

        usage = getattr(final_message, "usage", None)

        store.usage = Usage(
            provider=self._provider_name,
            model=getattr(final_message, "model", None) or model,
            request_id=request_id,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )


# Backward-compatible alias (deprecated)
AnthropicProvider = AnthropicMessagesProvider

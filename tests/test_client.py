"""Tests for yuullm.client and provider message conversion."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from yuullm import (
    AssistantMessage,
    Cost,
    PriceCalculator,
    Reasoning,
    Response,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
    YLLMClient,
)
from yuullm.providers.openai import OpenAIProvider
from yuullm.providers.anthropic import AnthropicProvider


# ---------------------------------------------------------------------------
# Fake provider for testing the client
# ---------------------------------------------------------------------------


class FakeProvider:
    """A provider that yields pre-configured items."""

    def __init__(self, items: list, usage: Usage) -> None:
        self._items = items
        self._usage = usage

    @property
    def name(self) -> str:
        return "fake"

    async def stream(self, messages, *, model, tools=None, **kwargs):
        store: dict = {}

        async def _iter() -> AsyncIterator:
            for item in self._items:
                yield item
            store["usage"] = self._usage
            store["provider_cost"] = None

        return _iter(), store


# ---------------------------------------------------------------------------
# Client tests
# ---------------------------------------------------------------------------


class TestYLLMClient:
    @pytest.mark.asyncio
    async def test_stream_basic(self):
        items = [Response(text="Hello"), Response(text=" world")]
        usage = Usage(provider="fake", model="test-model", input_tokens=10, output_tokens=5)
        provider = FakeProvider(items, usage)
        client = YLLMClient(provider=provider, default_model="test-model")

        stream, store = await client.stream([UserMessage(content="Hi")])
        collected = [item async for item in stream]

        assert len(collected) == 2
        assert collected[0] == Response(text="Hello")
        assert collected[1] == Response(text=" world")
        assert store["usage"] == usage
        assert store["cost"] is None  # no price calculator

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_and_tool_calls(self):
        items = [
            Reasoning(text="Let me think..."),
            ToolCall(id="tc_1", name="search", arguments='{"q": "test"}'),
            Response(text="Here's what I found."),
        ]
        usage = Usage(provider="fake", model="m", input_tokens=20, output_tokens=15)
        provider = FakeProvider(items, usage)
        client = YLLMClient(provider=provider, default_model="m")

        stream, store = await client.stream([UserMessage(content="Search for test")])
        collected = [item async for item in stream]

        assert isinstance(collected[0], Reasoning)
        assert isinstance(collected[1], ToolCall)
        assert isinstance(collected[2], Response)

    @pytest.mark.asyncio
    async def test_stream_with_price_calculator(self):
        items = [Response(text="ok")]
        usage = Usage(provider="fake", model="m", input_tokens=100, output_tokens=50)
        provider = FakeProvider(items, usage)

        # Use provider_cost path
        class FakeProviderWithCost(FakeProvider):
            async def stream(self, messages, *, model, tools=None, **kwargs):
                store: dict = {}

                async def _iter():
                    for item in self._items:
                        yield item
                    store["usage"] = self._usage
                    store["provider_cost"] = 0.005

                return _iter(), store

        provider = FakeProviderWithCost(items, usage)
        calc = PriceCalculator(enable_genai_prices=False)
        client = YLLMClient(provider=provider, default_model="m", price_calculator=calc)

        stream, store = await client.stream([UserMessage(content="Hi")])
        _ = [item async for item in stream]

        assert store["cost"] is not None
        assert store["cost"].total_cost == 0.005
        assert store["cost"].source == "provider"

    @pytest.mark.asyncio
    async def test_default_model_override(self):
        """model kwarg should override default_model."""
        items = [Response(text="ok")]
        usage = Usage(provider="fake", model="override-model", input_tokens=1, output_tokens=1)

        class TrackingProvider(FakeProvider):
            last_model: str | None = None

            async def stream(self, messages, *, model, tools=None, **kwargs):
                TrackingProvider.last_model = model
                return await super().stream(messages, model=model, tools=tools, **kwargs)

        provider = TrackingProvider(items, usage)
        client = YLLMClient(provider=provider, default_model="default-model")

        stream, store = await client.stream([UserMessage(content="Hi")], model="override-model")
        _ = [item async for item in stream]

        assert TrackingProvider.last_model == "override-model"


# ---------------------------------------------------------------------------
# OpenAI message conversion tests
# ---------------------------------------------------------------------------


class TestOpenAIMessageConversion:
    def test_system_message(self):
        msgs = OpenAIProvider._convert_messages([SystemMessage(content="Be helpful")])
        assert msgs == [{"role": "system", "content": "Be helpful"}]

    def test_user_message(self):
        msgs = OpenAIProvider._convert_messages([UserMessage(content="Hi")])
        assert msgs == [{"role": "user", "content": "Hi"}]

    def test_assistant_message_text(self):
        msgs = OpenAIProvider._convert_messages([AssistantMessage(content="Hello")])
        assert msgs == [{"role": "assistant", "content": "Hello"}]

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="tc_1", name="fn", arguments='{"a": 1}')
        msgs = OpenAIProvider._convert_messages([AssistantMessage(content="ok", tool_calls=[tc])])
        assert len(msgs) == 1
        assert msgs[0]["tool_calls"][0]["id"] == "tc_1"
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "fn"

    def test_tool_result_message(self):
        msgs = OpenAIProvider._convert_messages([ToolResultMessage(tool_call_id="tc_1", content="result")])
        assert msgs == [{"role": "tool", "tool_call_id": "tc_1", "content": "result"}]

    def test_tool_spec_conversion(self):
        tools = OpenAIProvider._convert_tools([
            ToolSpec(name="search", description="Search", parameters={"type": "object"}),
        ])
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search"


# ---------------------------------------------------------------------------
# Anthropic message conversion tests
# ---------------------------------------------------------------------------


class TestAnthropicMessageConversion:
    def test_system_extraction(self):
        msgs = [
            SystemMessage(content="Be helpful"),
            UserMessage(content="Hi"),
        ]
        system_text, rest = AnthropicProvider._extract_system(msgs)
        assert system_text == "Be helpful"
        assert len(rest) == 1
        assert isinstance(rest[0], UserMessage)

    def test_user_message(self):
        msgs = AnthropicProvider._convert_messages([UserMessage(content="Hi")])
        assert msgs == [{"role": "user", "content": "Hi"}]

    def test_assistant_with_tool_use(self):
        tc = ToolCall(id="tc_1", name="fn", arguments='{"a": 1}')
        msgs = AnthropicProvider._convert_messages([AssistantMessage(content="ok", tool_calls=[tc])])
        assert len(msgs) == 1
        blocks = msgs[0]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "tc_1"
        assert blocks[1]["input"] == {"a": 1}

    def test_tool_result_message(self):
        msgs = AnthropicProvider._convert_messages([ToolResultMessage(tool_call_id="tc_1", content="result")])
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["type"] == "tool_result"
        assert msgs[0]["content"][0]["tool_use_id"] == "tc_1"

    def test_tool_spec_conversion(self):
        tools = AnthropicProvider._convert_tools([
            ToolSpec(name="search", description="Search", parameters={"type": "object"}),
        ])
        assert tools[0]["name"] == "search"
        assert tools[0]["input_schema"] == {"type": "object"}

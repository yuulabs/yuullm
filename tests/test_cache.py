"""Tests for cache_config, PriceCalculator extensions, and OpenRouterProvider."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from yuullm import (
    CacheControl,
    CacheConfig,
    ConstantRate,
    Cost,
    Message,
    PriceCalculator,
    TextItem,
    TrafficEstimator,
    system,
    user,
    assistant,
)
from yuullm.cache_config import TrafficEstimator
from yuullm.providers.openrouter import OpenRouterProvider


# ---------------------------------------------------------------------------
# CacheConfig + TrafficEstimator
# ---------------------------------------------------------------------------


class TestTrafficEstimator:
    def test_constant_rate(self):
        est = ConstantRate(qps=2.0)
        assert est.expected_requests(0.0, 300) == 600.0

    def test_constant_rate_zero(self):
        est = ConstantRate(qps=0)
        assert est.expected_requests(0.0, 3600) == 0.0

    def test_base_class_raises(self):
        est = TrafficEstimator()
        with pytest.raises(NotImplementedError):
            est.expected_requests(0.0, 1.0)


class TestCacheConfig:
    def test_defaults(self):
        cc = CacheConfig()
        assert cc.refresh_interval == 30.0
        assert isinstance(cc.traffic, ConstantRate)
        assert cc.traffic.expected_requests(0, 1) == 0.0

    def test_custom(self):
        cc = CacheConfig(refresh_interval=60.0, traffic=ConstantRate(qps=5.0))
        assert cc.refresh_interval == 60.0
        assert cc.traffic.expected_requests(0, 10) == 50.0


# ---------------------------------------------------------------------------
# PriceCalculator extensions
# ---------------------------------------------------------------------------

YAML_CONTENT = """\
- provider: openai
  models:
    - id: gpt-4o
      prices:
        input_mtok: 2.5
        output_mtok: 10
        cache_read_mtok: 1.25

- provider: anthropic
  models:
    - id: claude-sonnet-4-20250514
      prices:
        input_mtok: 3
        output_mtok: 15
        cache_read_mtok: 0.3
        cache_write_mtok: 3.75
"""


def _make_yaml() -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(YAML_CONTENT)
    f.close()
    return Path(f.name)


class TestModelSuffixStripping:
    """OpenRouter appends suffixes like :beta, :free, timestamps to model names."""

    def test_strip_beta(self):
        from yuullm.pricing import _strip_suffix
        assert _strip_suffix("claude-sonnet-4-20250514:beta") == "claude-sonnet-4-20250514"

    def test_strip_free(self):
        from yuullm.pricing import _strip_suffix
        assert _strip_suffix("claude-sonnet-4-20250514:free") == "claude-sonnet-4-20250514"

    def test_strip_timestamp(self):
        from yuullm.pricing import _strip_suffix
        assert _strip_suffix("claude-sonnet-4-20250514:1234567890") == "claude-sonnet-4-20250514"

    def test_strip_date(self):
        from yuullm.pricing import _strip_suffix
        assert _strip_suffix("claude-sonnet-4-20250514:2024-11-20") == "claude-sonnet-4-20250514"

    def test_no_suffix(self):
        from yuullm.pricing import _strip_suffix
        assert _strip_suffix("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"

    def test_with_slash_prefix(self):
        from yuullm.pricing import _strip_suffix
        assert _strip_suffix("anthropic/claude-sonnet-4-20250514:beta") == "anthropic/claude-sonnet-4-20250514"

    def test_yaml_lookup_with_suffix(self):
        """YAML lookup should find model even with suffix."""
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=False)
        prices = calc.get_base_prices("anthropic", "claude-sonnet-4-20250514:beta")
        assert prices is not None
        assert prices["input_mtok"] == 3

    def test_calculate_with_suffix(self):
        """calculate() should find pricing even with suffixed model name."""
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=False)
        from yuullm import Usage
        usage = Usage(
            provider="anthropic",
            model="claude-sonnet-4-20250514:beta",
            input_tokens=1_000_000,
        )
        cost = calc.calculate(usage)
        assert cost is not None
        assert cost.source == "yaml"

    def test_genai_relay_with_suffix(self):
        """genai-prices should work with relay model names + suffix."""
        calc = PriceCalculator(enable_genai_prices=True)
        prices = calc.get_base_prices("openrouter", "anthropic/claude-sonnet-4-20250514:beta")
        assert prices is not None
        assert prices["input_mtok"] > 0

    def test_calculate_openrouter_relay(self):
        """calculate() should work for OpenRouter relay model names."""
        calc = PriceCalculator(enable_genai_prices=True)
        from yuullm import Usage
        usage = Usage(
            provider="openrouter",
            model="anthropic/claude-sonnet-4-20250514",
            input_tokens=1_000_000,
            output_tokens=100_000,
        )
        cost = calc.calculate(usage)
        assert cost is not None
        assert cost.source == "genai-prices"
        assert cost.total_cost > 0


class TestGetBasePrices:
    def test_yaml_source(self):
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=False)
        prices = calc.get_base_prices("openai", "gpt-4o")
        assert prices is not None
        assert prices["input_mtok"] == 2.5
        assert prices["output_mtok"] == 10

    def test_yaml_miss(self):
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=False)
        prices = calc.get_base_prices("openai", "unknown-model")
        assert prices is None

    def test_genai_fallback(self):
        calc = PriceCalculator(enable_genai_prices=True)
        prices = calc.get_base_prices("openai", "gpt-4o")
        # genai-prices should know about gpt-4o
        assert prices is not None
        assert prices["input_mtok"] > 0

    def test_genai_unknown_returns_none(self):
        calc = PriceCalculator(enable_genai_prices=True)
        prices = calc.get_base_prices("openai", "totally-fake-model-xyz-999")
        assert prices is None

    def test_relay_model_slash_fallback(self):
        """OpenRouter-style model 'anthropic/claude-...' should parse and find upstream."""
        calc = PriceCalculator(enable_genai_prices=True)
        prices = calc.get_base_prices("openrouter", "anthropic/claude-sonnet-4-20250514")
        # Should find via upstream fallback
        assert prices is not None
        assert prices["input_mtok"] > 0


class TestEstimate:
    def test_basic(self):
        calc = PriceCalculator()
        prices = {"input_mtok": 3.0, "output_mtok": 15.0, "cache_read_mtok": 0.3, "cache_write_mtok": 3.75}
        cost = calc.estimate(
            prices,
            input_tokens=1_000_000,
            output_tokens=100_000,
            cache_read_tokens=200_000,
            cache_write_tokens=50_000,
        )
        assert isinstance(cost, Cost)
        assert abs(cost.input_cost - 3.0) < 1e-9
        assert abs(cost.output_cost - 1.5) < 1e-9
        assert abs(cost.cache_read_cost - 0.06) < 1e-9
        assert abs(cost.cache_write_cost - 0.1875) < 1e-9
        assert cost.source == "estimate"

    def test_zero_tokens(self):
        calc = PriceCalculator()
        cost = calc.estimate({"input_mtok": 10, "output_mtok": 30})
        assert cost.total_cost == 0.0

    def test_overridden_prices(self):
        """Provider can pass modified prices to compare TTL tiers."""
        calc = PriceCalculator()
        base = {"input_mtok": 3.0, "output_mtok": 15.0, "cache_read_mtok": 0.3, "cache_write_mtok": 3.75}
        # Override cache_write for 1h TTL
        cost_1h = calc.estimate(
            {**base, "cache_write_mtok": 3.0 * 2.0},  # 2x input for 1h
            cache_write_tokens=100_000,
            cache_read_tokens=500_000,
        )
        cost_5m = calc.estimate(
            {**base, "cache_write_mtok": 3.0 * 1.25},  # 1.25x input for 5m
            cache_write_tokens=100_000,
            cache_read_tokens=50_000,
        )
        # 1h has more reads so could be cheaper overall
        assert cost_1h.cache_write_cost > cost_5m.cache_write_cost
        assert cost_1h.cache_read_cost > cost_5m.cache_read_cost


# ---------------------------------------------------------------------------
# OpenRouterProvider
# ---------------------------------------------------------------------------


class TestOpenRouterProvider:
    def test_init(self):
        p = OpenRouterProvider(api_key="fake")
        assert p.provider == "openrouter"
        assert p.api_type == "openai-chat-completion"

    def test_anthropic_cache_injection(self):
        """For anthropic/* models, cache_control should be injected."""
        p = OpenRouterProvider(
            api_key="fake",
            cache_config=CacheConfig(),
        )
        messages = [
            system("You are a helpful assistant."),
            user("Hello!"),
        ]
        result = p._apply_cache(messages, "anthropic/claude-sonnet-4-20250514", None)

        # System message should have cache_control on last block
        sys_items = result[0][1]
        assert "cache_control" in sys_items[-1]

    def test_no_cache_for_openai_models(self):
        """For openai/* models, messages should pass through unchanged."""
        p = OpenRouterProvider(
            api_key="fake",
            cache_config=CacheConfig(),
        )
        messages = [
            system("You are helpful."),
            user("Hi"),
        ]
        result = p._apply_cache(messages, "openai/gpt-4o", None)
        # Should be unchanged
        for _, items in result:
            for it in items:
                assert "cache_control" not in it

    def test_ttl_decision_with_pricing(self):
        """With PriceCalculator, should pick TTL based on cost."""
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=True)

        # High traffic: 1h TTL should be preferred (more reads to amortize)
        p = OpenRouterProvider(
            api_key="fake",
            cache_config=CacheConfig(traffic=ConstantRate(qps=1.0)),
            price_calculator=calc,
        )
        # Large system prompt
        messages = [
            system("A" * 10000),  # ~2500 tokens
            user("Hello"),
        ]
        ttl = p._pick_ttl_anthropic("anthropic/claude-sonnet-4-20250514", messages, None)
        # With high traffic, should prefer longer TTL
        assert ttl in (300, 3600)

    def test_ttl_default_without_pricing(self):
        """Without PriceCalculator, default to 300s."""
        p = OpenRouterProvider(
            api_key="fake",
            cache_config=CacheConfig(),
        )
        ttl = p._pick_ttl_anthropic("anthropic/claude-sonnet-4-20250514", [], None)
        assert ttl == 300

    def test_tools_cache_annotation(self):
        """Last tool should get cache_control for anthropic models."""
        p = OpenRouterProvider(
            api_key="fake",
            cache_config=CacheConfig(),
        )
        tools = [
            {"name": "search", "description": "Search", "parameters": {}},
            {"name": "calc", "description": "Calculate", "parameters": {}},
        ]
        result = p._apply_cache_tools(tools, "anthropic/claude-sonnet-4-20250514")
        assert result is not None
        assert "cache_control" not in result[0]
        assert "cache_control" in result[-1]

    def test_tools_no_annotation_for_openai(self):
        """Tools should not be annotated for openai models."""
        p = OpenRouterProvider(
            api_key="fake",
            cache_config=CacheConfig(),
        )
        tools = [{"name": "search", "description": "Search", "parameters": {}}]
        result = p._apply_cache_tools(tools, "openai/gpt-4o")
        assert result is tools  # unchanged, same object

    def test_google_cache_injection(self):
        """For google/* models, cache_control should be injected."""
        p = OpenRouterProvider(
            api_key="fake",
            cache_config=CacheConfig(),
        )
        messages = [
            system("You are a helpful assistant."),
            user("Turn 1"),
            assistant("Response 1"),
            user("Turn 2"),
        ]
        result = p._apply_cache(messages, "google/gemini-3-flash-preview", None)

        # System message should have cache_control
        assert "cache_control" in result[0][1][-1]
        # Prefix boundary should have cache_control
        assert "cache_control" in result[2][1][-1]
        # All cache_control should be plain ephemeral (no TTL for Gemini)
        assert result[0][1][-1]["cache_control"] == {"type": "ephemeral"}

    def test_google_tools_cache_annotation(self):
        """Last tool should get cache_control for google models."""
        p = OpenRouterProvider(
            api_key="fake",
            cache_config=CacheConfig(),
        )
        tools = [
            {"name": "search", "description": "Search", "parameters": {}},
            {"name": "calc", "description": "Calculate", "parameters": {}},
        ]
        result = p._apply_cache_tools(tools, "google/gemini-3-flash-preview")
        assert result is not None
        assert "cache_control" not in result[0]
        assert result[-1]["cache_control"] == {"type": "ephemeral"}

    def test_mark_breakpoints_prefix_boundary(self):
        """Should mark system last block + prefix boundary."""
        messages = [
            system("System prompt"),
            user("Turn 1"),
            assistant("Response 1"),
            user("Turn 2"),
        ]
        cc: CacheControl = {"type": "ephemeral"}
        result = OpenRouterProvider._mark_breakpoints(messages, cc)

        # System last block marked
        assert "cache_control" in result[0][1][-1]

        # Prefix boundary: assistant "Response 1" (second-to-last message before final user)
        assert "cache_control" in result[2][1][-1]

        # Final user turn should NOT be marked
        for it in result[3][1]:
            assert "cache_control" not in it


class TestOpenRouterProviderEstimatePrefixTokens:
    def test_basic(self):
        messages = [
            system("A" * 400),  # ~100 tokens
            user("Hello"),
        ]
        n = OpenRouterProvider._estimate_prefix_tokens(messages, None)
        assert n == 400 // 4  # system only, last user excluded

    def test_with_tools(self):
        messages = [system("Hi"), user("Hello")]
        tools = [{"name": "fn", "description": "d", "parameters": {}}]
        n = OpenRouterProvider._estimate_prefix_tokens(messages, tools)
        assert n > 0  # includes tool JSON


# ---------------------------------------------------------------------------
# YLLMClient auto_prompt_caching integration
# ---------------------------------------------------------------------------


class TestAutoPromptCaching:
    def test_cache_config_injected_into_provider(self):
        """YLLMClient should inject cache_config into provider."""
        from yuullm import YLLMClient

        p = OpenRouterProvider(api_key="fake")
        assert p._cache_config is None

        client = YLLMClient(
            provider=p,
            default_model="anthropic/claude-sonnet-4-20250514",
            auto_prompt_caching=True,
        )
        assert p._cache_config is not None

    def test_auto_caching_disabled(self):
        """When auto_prompt_caching=False, cache_config should not be set."""
        from yuullm import YLLMClient

        p = OpenRouterProvider(api_key="fake")
        client = YLLMClient(
            provider=p,
            default_model="anthropic/claude-sonnet-4-20250514",
            auto_prompt_caching=False,
        )
        assert p._cache_config is None

    def test_explicit_cache_config_preserved(self):
        """Explicit cache_config on provider should not be overwritten."""
        from yuullm import YLLMClient

        explicit = CacheConfig(refresh_interval=120.0)
        p = OpenRouterProvider(api_key="fake", cache_config=explicit)

        client = YLLMClient(
            provider=p,
            default_model="m",
            auto_prompt_caching=True,
        )
        # Should keep the explicit one, not overwrite
        assert p._cache_config is explicit


# ---------------------------------------------------------------------------
# _convert_messages: cache_control preservation
# ---------------------------------------------------------------------------


class TestConvertMessagesWithCacheControl:
    """OpenRouterProvider._convert_messages should preserve cache_control."""

    def test_system_with_cache_control_uses_array(self):
        """System message with cache_control must use content array format."""
        text_item: TextItem = {
            "type": "text",
            "text": "You are helpful.",
            "cache_control": {"type": "ephemeral"},
        }
        messages: list[Message] = [
            ("system", [text_item]),
        ]
        result = OpenRouterProvider._convert_messages(messages)
        assert result[0]["role"] == "system"
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_without_cache_control_uses_string(self):
        """System message without cache_control should use plain string."""
        text_item: TextItem = {"type": "text", "text": "You are helpful."}
        messages: list[Message] = [
            ("system", [text_item]),
        ]
        result = OpenRouterProvider._convert_messages(messages)
        assert result[0]["content"] == "You are helpful."

    def test_user_with_cache_control_uses_array(self):
        """User message with cache_control must use content array format."""
        message = user("Hello!")
        message[1][0]["cache_control"] = {"type": "ephemeral"}
        messages: list[Message] = [message]
        result = OpenRouterProvider._convert_messages(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_user_without_cache_control_uses_string(self):
        """User text-only message without cache_control should use plain string."""
        messages: list[Message] = [user("Hello!")]
        result = OpenRouterProvider._convert_messages(messages)
        assert result[0]["content"] == "Hello!"

    def test_assistant_with_cache_control_uses_array(self):
        """Assistant text with cache_control must use content array format."""
        message = assistant("Sure!")
        message[1][0]["cache_control"] = {"type": "ephemeral"}
        messages: list[Message] = [message]
        result = OpenRouterProvider._convert_messages(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_end_to_end_cache_preserved(self):
        """Full pipeline: _mark_breakpoints -> _convert_messages preserves cache_control."""
        messages = [
            system("System prompt"),
            user("Turn 1"),
            assistant("Response 1"),
            user("Turn 2"),
        ]
        marked = OpenRouterProvider._mark_breakpoints(messages, {"type": "ephemeral"})
        converted = OpenRouterProvider._convert_messages(marked)

        # System message should have cache_control in content array
        sys_msg = converted[0]
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}

        # Assistant (prefix boundary) should have cache_control
        asst_msg = converted[2]
        assert isinstance(asst_msg["content"], list)
        assert asst_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}

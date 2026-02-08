"""Tests for yuullm.pricing."""

import tempfile
from pathlib import Path

from yuullm import Cost, Usage
from yuullm.pricing import PriceCalculator


YAML_CONTENT = """\
- provider: openai
  models:
    - id: gpt-4o
      prices:
        input_mtok: 2.5
        output_mtok: 10
        cache_read_mtok: 1.25

    - id: gpt-4o-mini
      prices:
        input_mtok: 0.15
        output_mtok: 0.6

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


class TestProviderCostSource:
    def test_provider_cost_takes_priority(self):
        calc = PriceCalculator()
        usage = Usage(provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=500)
        cost = calc.calculate(usage, provider_cost=0.042)
        assert cost is not None
        assert cost.total_cost == 0.042
        assert cost.source == "provider"

    def test_provider_cost_zero_is_valid(self):
        calc = PriceCalculator()
        usage = Usage(provider="openai", model="gpt-4o")
        cost = calc.calculate(usage, provider_cost=0.0)
        assert cost is not None
        assert cost.total_cost == 0.0
        assert cost.source == "provider"


class TestYamlSource:
    def test_yaml_basic(self):
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=False)
        usage = Usage(
            provider="openai",
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        cost = calc.calculate(usage)
        assert cost is not None
        assert cost.source == "yaml"
        assert cost.input_cost == 2.5  # 1M * 2.5/M
        assert cost.output_cost == 10.0  # 1M * 10/M
        assert cost.total_cost == 12.5

    def test_yaml_with_cache(self):
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=False)
        usage = Usage(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            input_tokens=500_000,
            output_tokens=100_000,
            cache_read_tokens=200_000,
            cache_write_tokens=50_000,
        )
        cost = calc.calculate(usage)
        assert cost is not None
        assert cost.source == "yaml"
        assert abs(cost.input_cost - 1.5) < 1e-9  # 500k * 3/M
        assert abs(cost.output_cost - 1.5) < 1e-9  # 100k * 15/M
        assert abs(cost.cache_read_cost - 0.06) < 1e-9  # 200k * 0.3/M
        assert abs(cost.cache_write_cost - 0.1875) < 1e-9  # 50k * 3.75/M

    def test_yaml_miss_returns_none(self):
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=False)
        usage = Usage(provider="openai", model="unknown-model")
        cost = calc.calculate(usage)
        assert cost is None

    def test_yaml_provider_mismatch(self):
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=False)
        usage = Usage(provider="anthropic", model="gpt-4o")
        cost = calc.calculate(usage)
        assert cost is None


class TestGenaiPricesFallback:
    def test_known_model(self):
        calc = PriceCalculator(enable_genai_prices=True)
        usage = Usage(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=100,
        )
        cost = calc.calculate(usage)
        assert cost is not None
        assert cost.source == "genai-prices"
        assert cost.total_cost > 0

    def test_unknown_model_returns_none(self):
        calc = PriceCalculator(enable_genai_prices=True)
        usage = Usage(
            provider="openai",
            model="totally-fake-model-xyz-999",
            input_tokens=100,
            output_tokens=50,
        )
        cost = calc.calculate(usage)
        assert cost is None

    def test_disabled(self):
        calc = PriceCalculator(enable_genai_prices=False)
        usage = Usage(provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=100)
        cost = calc.calculate(usage)
        assert cost is None


class TestPriority:
    def test_provider_cost_over_yaml(self):
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path)
        usage = Usage(provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=100)
        cost = calc.calculate(usage, provider_cost=0.001)
        assert cost is not None
        assert cost.source == "provider"

    def test_yaml_over_genai_prices(self):
        yaml_path = _make_yaml()
        calc = PriceCalculator(yaml_path=yaml_path, enable_genai_prices=True)
        usage = Usage(
            provider="openai",
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        cost = calc.calculate(usage)
        assert cost is not None
        assert cost.source == "yaml"

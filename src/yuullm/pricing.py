"""PriceCalculator – three-level cost calculation engine.

Priority (high → low):
1. Provider-supplied cost (e.g. OpenRouter ``total_cost``)
2. User YAML configuration file
3. ``genai-prices`` library fallback
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .types import Cost, Usage


class PriceCalculator:
    """Calculates :class:`Cost` from :class:`Usage` using a three-level
    price source hierarchy."""

    def __init__(
        self,
        yaml_path: str | Path | None = None,
        enable_genai_prices: bool = True,
    ) -> None:
        self._yaml_prices: dict[tuple[str, str], dict[str, float]] = {}
        self._enable_genai_prices = enable_genai_prices

        if yaml_path is not None:
            self._load_yaml(Path(yaml_path))

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------

    def _load_yaml(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            return

        for provider_entry in data:
            provider_name = provider_entry.get("provider", "")
            for model_entry in provider_entry.get("models", []):
                model_id = model_entry.get("id", "")
                prices = model_entry.get("prices", {})
                self._yaml_prices[(provider_name, model_id)] = prices

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        usage: Usage,
        provider_cost: float | None = None,
    ) -> Cost | None:
        """Calculate cost using the three-level hierarchy.

        Returns ``None`` when no price source can determine the cost.
        """
        # Source 1: provider-supplied cost
        if provider_cost is not None:
            return Cost(
                input_cost=0.0,
                output_cost=0.0,
                total_cost=provider_cost,
                source="provider",
            )

        # Source 2: YAML configuration
        cost = self._from_yaml(usage)
        if cost is not None:
            return cost

        # Source 3: genai-prices fallback
        if self._enable_genai_prices:
            cost = self._from_genai_prices(usage)
            if cost is not None:
                return cost

        return None

    # ------------------------------------------------------------------
    # Source 2: YAML
    # ------------------------------------------------------------------

    def _from_yaml(self, usage: Usage) -> Cost | None:
        prices = self._yaml_prices.get((usage.provider, usage.model))
        if prices is None:
            return None

        input_mtok = prices.get("input_mtok", 0.0)
        output_mtok = prices.get("output_mtok", 0.0)
        cache_read_mtok = prices.get("cache_read_mtok", 0.0)
        cache_write_mtok = prices.get("cache_write_mtok", 0.0)

        input_cost = usage.input_tokens * input_mtok / 1_000_000
        output_cost = usage.output_tokens * output_mtok / 1_000_000
        cache_read_cost = usage.cache_read_tokens * cache_read_mtok / 1_000_000
        cache_write_cost = usage.cache_write_tokens * cache_write_mtok / 1_000_000
        total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

        return Cost(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            source="yaml",
        )

    # ------------------------------------------------------------------
    # Source 3: genai-prices
    # ------------------------------------------------------------------

    def _from_genai_prices(self, usage: Usage) -> Cost | None:
        try:
            from genai_prices import Usage as GPUsage
            from genai_prices import calc_price
        except ImportError:
            return None

        try:
            price_data = calc_price(
                GPUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                ),
                model_ref=usage.model,
                provider_id=usage.provider,
            )
        except Exception:
            return None

        if price_data is None or price_data.total_price is None:
            return None

        input_price = getattr(price_data, "input_price", None) or 0.0
        output_price = getattr(price_data, "output_price", None) or 0.0

        return Cost(
            input_cost=float(input_price),
            output_cost=float(output_price),
            total_cost=float(price_data.total_price),
            source="genai-prices",
        )

"""PriceCalculator – three-level cost calculation engine.

Priority (high → low):
1. Provider-supplied cost (e.g. OpenRouter ``total_cost``)
2. User YAML configuration file
3. ``genai-prices`` library fallback
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .types import Cost, Usage

# OpenRouter (and similar relays) may append suffixes to model names:
#   "anthropic/claude-sonnet-4-20250514:beta"
#   "anthropic/claude-sonnet-4-20250514:1234567890"   (unix timestamp)
#   "anthropic/claude-sonnet-4-20250514:2024-11-20"   (date)
# Strip everything from the first colon that is NOT part of the base
# model id (dates like 20250514 are part of the name; suffixes after
# a colon are not).
_SUFFIX_RE = re.compile(r":[\w-]+$")


def _strip_suffix(model: str) -> str:
    """Remove OpenRouter-style colon suffixes (`:beta`, `:free`, timestamps)."""
    return _SUFFIX_RE.sub("", model)


def _coerce_price_value(value) -> float:
    """Normalize genai-prices price values to a scalar per-million-token price.

    Recent genai-prices versions may return ``TieredPrices`` instead of a raw
    Decimal/float for some aliases like ``claude-sonnet-4.6``. For base price
    lookups used by cache TTL estimation, we only need the baseline tier.
    """
    if value is None:
        return 0.0

    base = getattr(value, "base", None)
    if base is not None:
        value = base

    return float(value or 0)


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

    def get_base_prices(
        self,
        provider: str,
        model: str,
    ) -> dict[str, float] | None:
        """Return raw per-million-token prices for *(provider, model)*.

        Priority: YAML → genai-prices.  Returns ``None`` if no source
        can determine prices.

        Keys: ``input_mtok``, ``output_mtok``, ``cache_read_mtok``,
        ``cache_write_mtok``.

        For relay providers like OpenRouter, if ``model`` contains a
        slash (e.g. ``"anthropic/claude-sonnet-4-20250514"``), the method
        first tries the relay provider + full model id, then falls back
        to parsing ``"vendor/model"`` and querying the upstream vendor
        prices from genai-prices.
        """
        # Try YAML first (exact match, then suffix-stripped)
        prices = self._yaml_prices.get((provider, model))
        if prices is None:
            prices = self._yaml_prices.get((provider, _strip_suffix(model)))
        if prices is not None:
            return dict(prices)

        # genai-prices fallback
        if self._enable_genai_prices:
            prices = self._prices_from_genai(provider, model)
            if prices is not None:
                return prices

        return None

    def _prices_from_genai(
        self, provider: str, model: str
    ) -> dict[str, float] | None:
        """Query genai-prices, with suffix stripping and relay fallback."""
        try:
            from genai_prices import Usage as GPUsage, calc_price
        except ImportError:
            return None

        clean = _strip_suffix(model)

        # Try direct lookup (with cleaned model name)
        mp = self._try_genai_model_price(provider, clean)

        # Relay fallback: "anthropic/claude-..." → ("anthropic", "claude-...")
        if mp is None and "/" in clean:
            upstream_provider, _, upstream_model = clean.partition("/")
            mp = self._try_genai_model_price(upstream_provider, upstream_model)

        if mp is None:
            return None

        return {
            "input_mtok": _coerce_price_value(getattr(mp, "input_mtok", 0)),
            "output_mtok": _coerce_price_value(getattr(mp, "output_mtok", 0)),
            "cache_read_mtok": _coerce_price_value(getattr(mp, "cache_read_mtok", 0)),
            "cache_write_mtok": _coerce_price_value(getattr(mp, "cache_write_mtok", 0)),
        }

    @staticmethod
    def _try_genai_model_price(provider: str, model: str):
        """Attempt a genai-prices lookup, return ModelPrice or None."""
        try:
            from genai_prices import Usage as GPUsage, calc_price

            result = calc_price(
                GPUsage(input_tokens=0, output_tokens=0),
                model_ref=model,
                provider_id=provider,
            )
            if result is not None and result.model_price is not None:
                return result.model_price
        except Exception:
            pass
        return None

    def estimate(
        self,
        prices: dict[str, float],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Cost:
        """Pure arithmetic: prices * tokens.

        *prices* is a dict with keys ``input_mtok``, ``output_mtok``,
        ``cache_read_mtok``, ``cache_write_mtok``.  Providers pass in
        overridden dicts (e.g. with different ``cache_write_mtok``) to
        compare TTL options.
        """
        input_cost = input_tokens * prices.get("input_mtok", 0) / 1_000_000
        output_cost = output_tokens * prices.get("output_mtok", 0) / 1_000_000
        cache_read_cost = cache_read_tokens * prices.get("cache_read_mtok", 0) / 1_000_000
        cache_write_cost = cache_write_tokens * prices.get("cache_write_mtok", 0) / 1_000_000
        total = input_cost + output_cost + cache_read_cost + cache_write_cost
        return Cost(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            source="estimate",
        )

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
        # Fallback: try with suffix stripped
        if prices is None:
            prices = self._yaml_prices.get((usage.provider, _strip_suffix(usage.model)))
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

        model = _strip_suffix(usage.model)
        provider = usage.provider
        gp_usage = GPUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )

        price_data = self._try_calc_price(calc_price, gp_usage, provider, model)

        # Relay fallback: "anthropic/claude-..." → ("anthropic", "claude-...")
        if price_data is None and "/" in model:
            upstream_provider, _, upstream_model = model.partition("/")
            price_data = self._try_calc_price(
                calc_price, gp_usage, upstream_provider, upstream_model
            )

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

    @staticmethod
    def _try_calc_price(calc_price, gp_usage, provider: str, model: str):
        """Attempt calc_price, return result or None."""
        try:
            return calc_price(gp_usage, model_ref=model, provider_id=provider)
        except Exception:
            return None

from __future__ import annotations

import pytest

from benchtrace.providers.pricing import calculate_cost, load_pricing_table
from benchtrace.trace import TokenUsage


def test_current_model_aliases_have_prices() -> None:
    table = load_pricing_table()

    assert table.find("openai", "gpt-5.6-sol-2026-08-31") is not None
    assert table.find("anthropic", "claude-fable-5-20260831") is not None
    assert table.find("google", "gemini-3.7-flash-preview") is not None


def test_cost_distinguishes_uncached_reads_and_cache_writes() -> None:
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_input_tokens=200_000,
        cache_write_tokens=100_000,
        cache_write_1h_tokens=50_000,
    )

    result = calculate_cost("anthropic", "claude-fable-5-20260831", usage)

    # 650k uncached input + 200k reads + 100k 5m writes + 50k 1h writes + output.
    assert result.usd == pytest.approx(13.95)
    assert result.priced


def test_long_context_rate_uses_total_input_threshold() -> None:
    usage = TokenUsage(input_tokens=300_000, output_tokens=10_000)

    result = calculate_cost("openai", "gpt-5.6-sol", usage)

    assert result.usd == pytest.approx(2.7)

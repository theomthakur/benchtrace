"""Versioned provider pricing loaded from package data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from benchtrace.trace import TokenUsage

DEFAULT_PRICING_VERSION = "v1"


class ModelPrice(BaseModel):
    """Token prices in USD per one million tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: str
    model: str
    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)
    cached_input: float = Field(default=0.0, ge=0.0)


class PricingTable(BaseModel):
    """A dated, immutable pricing snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    effective_date: str
    currency: str
    unit: str
    models: list[ModelPrice]

    def find(self, provider: str, model: str) -> ModelPrice | None:
        """Find an exact or explicitly wildcarded model price."""

        provider_key = provider.lower()
        for price in self.models:
            if price.provider.lower() != provider_key:
                continue
            if price.model == model:
                return price
            if price.model.endswith("*") and model.startswith(price.model[:-1]):
                return price
        return None


@dataclass(frozen=True, slots=True)
class CostResult:
    """Computed call cost and whether the model was present in the snapshot."""

    usd: float
    priced: bool
    pricing_version: str


def load_pricing_table(version: str = DEFAULT_PRICING_VERSION) -> PricingTable:
    """Load a bundled pricing snapshot by version."""

    resource = files("benchtrace").joinpath("data", f"pricing-{version}.json")
    raw = json.loads(resource.read_text(encoding="utf-8"))
    return PricingTable.model_validate(raw)


def load_pricing_file(path: str | Path) -> PricingTable:
    """Load a user-supplied pricing snapshot with the same validated schema."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return PricingTable.model_validate(raw)


def calculate_cost(
    provider: str,
    model: str,
    usage: TokenUsage,
    table: PricingTable | None = None,
    *,
    pricing_version: str = DEFAULT_PRICING_VERSION,
) -> CostResult:
    """Calculate token cost without embedding prices in provider clients."""

    selected = table or load_pricing_table(pricing_version)
    price = selected.find(provider, model)
    if price is None:
        return CostResult(usd=0.0, priced=False, pricing_version=pricing_version)
    uncached_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
    total = (
        uncached_input * price.input
        + usage.cached_input_tokens * price.cached_input
        + usage.output_tokens * price.output
    ) / 1_000_000
    return CostResult(usd=round(total, 12), priced=True, pricing_version=pricing_version)

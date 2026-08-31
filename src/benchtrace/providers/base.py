"""Provider interface, normalized result, and retry policy."""

from __future__ import annotations

import asyncio
import email.utils
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeAlias

from benchtrace.providers.pricing import PricingTable, calculate_cost
from benchtrace.trace import JsonValue, TokenUsage, TraceRequest, TraceResponse

Sleep: TypeAlias = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Shared execution controls for all provider adapters."""

    max_concurrency: int = 4
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    sleeper: Sleep = asyncio.sleep
    random_value: Callable[[], float] = random.random
    pricing_table: PricingTable | None = None


class ProviderError(RuntimeError):
    """Normalized provider failure with retry and response context."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        headers: Mapping[str, str] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = {key.lower(): value for key, value in (headers or {}).items()}
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class WireResult:
    """Adapter output before latency and cost enrichment."""

    response: TraceResponse
    usage: TokenUsage
    raw: JsonValue


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Normalized provider call ready to be incorporated into a trace."""

    response: TraceResponse
    usage: TokenUsage
    latency_ms: float
    cost_usd: float
    pricing_known: bool
    pricing_version: str
    attempts: int
    raw: JsonValue


def rate_limit_delay(headers: Mapping[str, str], *, now: datetime | None = None) -> float | None:
    """Parse standard and provider-specific reset headers into seconds."""

    normalized = {key.lower(): value.strip() for key, value in headers.items()}
    current = now or datetime.now(UTC)
    retry_after_ms = normalized.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            return max(float(retry_after_ms) / 1000.0, 0.0)
        except ValueError:
            pass
    retry_after = normalized.get("retry-after")
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(retry_after)
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return max((parsed - current).total_seconds(), 0.0)
    for name in (
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset",
        "ratelimit-reset",
    ):
        value = normalized.get(name)
        if value is None:
            continue
        try:
            numeric = float(value.rstrip("s"))
        except ValueError:
            continue
        if numeric > current.timestamp():
            return max(numeric - current.timestamp(), 0.0)
        return max(numeric, 0.0)
    return None


class Provider(ABC):
    """One async provider interface shared by every adapter."""

    name: str
    supports_seed: bool = False

    def __init__(self, config: ProviderConfig | None = None) -> None:
        selected = config or ProviderConfig()
        if selected.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if selected.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.max_retries = selected.max_retries
        self.base_delay = selected.base_delay
        self.max_delay = selected.max_delay
        self._sleep = selected.sleeper
        self._random = selected.random_value
        self._semaphore = asyncio.Semaphore(selected.max_concurrency)
        self._pricing_table = selected.pricing_table

    async def complete(self, request: TraceRequest, model: str) -> ProviderResult:
        """Execute with bounded concurrency, retries, latency, and cost accounting."""

        started = time.perf_counter()
        attempts = 0
        async with self._semaphore:
            while True:
                attempts += 1
                try:
                    wire = await self._send_once(request, model)
                    break
                except ProviderError as error:
                    if not error.retryable or attempts > self.max_retries:
                        raise
                    delay = self._retry_delay(error, attempts)
                    await self._sleep(delay)
        latency_ms = (time.perf_counter() - started) * 1000.0
        cost = calculate_cost(self.name, model, wire.usage, self._pricing_table)
        return ProviderResult(
            response=wire.response,
            usage=wire.usage,
            latency_ms=latency_ms,
            cost_usd=cost.usd,
            pricing_known=cost.priced,
            pricing_version=cost.pricing_version,
            attempts=attempts,
            raw=wire.raw,
        )

    def _retry_delay(self, error: ProviderError, attempt: int) -> float:
        if error.status_code == 429:
            header_delay = rate_limit_delay(error.headers)
            if header_delay is not None:
                return min(header_delay, self.max_delay)
        exponential = min(self.base_delay * (2.0 ** (attempt - 1)), self.max_delay)
        return exponential * (0.5 + self._random() * 0.5)

    @abstractmethod
    async def _send_once(self, request: TraceRequest, model: str) -> WireResult:
        """Perform and normalize one HTTP attempt."""

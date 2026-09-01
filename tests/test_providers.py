from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx
import pytest

from benchtrace.providers.anthropic import AnthropicProvider
from benchtrace.providers.base import (
    Provider,
    ProviderConfig,
    ProviderError,
    WireResult,
    rate_limit_delay,
)
from benchtrace.providers.google import GoogleProvider
from benchtrace.providers.openai import OpenAICompatibleProvider
from benchtrace.trace import Message, TokenUsage, TraceRequest, TraceResponse


class RetryingProvider(Provider):
    name = "test"
    supports_seed = True

    def __init__(self, sleeper: Callable[[float], Awaitable[None]]) -> None:
        super().__init__(
            ProviderConfig(max_retries=2, sleeper=sleeper, random_value=lambda: 0.0)
        )
        self.calls = 0

    async def _send_once(self, request: TraceRequest, model: str) -> WireResult:
        del request, model
        self.calls += 1
        if self.calls == 1:
            raise ProviderError(
                "limited",
                status_code=429,
                headers={"Retry-After": "1.25"},
                retryable=True,
            )
        return WireResult(
            response=TraceResponse(content="ok"),
            usage=TokenUsage(input_tokens=5, output_tokens=3),
            raw={"ok": True},
        )


@pytest.mark.asyncio
async def test_retry_uses_rate_limit_header_and_records_cost() -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    provider = RetryingProvider(sleep)
    result = await provider.complete(
        TraceRequest(messages=[Message(role="user", content="hello")]), "test-model"
    )

    assert provider.calls == 2
    assert delays == [1.25]
    assert result.attempts == 2
    assert result.usage.total_tokens == 8
    assert result.cost_usd == pytest.approx(0.000011)
    assert result.pricing_known
    assert result.latency_ms >= 0


def test_rate_limit_delay_parses_milliseconds_epoch_and_http_date() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)

    assert rate_limit_delay({"retry-after-ms": "250"}, now=now) == 0.25
    assert rate_limit_delay(
        {"x-ratelimit-reset": str(now.timestamp() + 3)}, now=now
    ) == pytest.approx(3.0)
    assert rate_limit_delay({"retry-after": "Thu, 01 Jan 2026 00:00:04 GMT"}, now=now) == 4


@pytest.mark.asyncio
async def test_openai_compatible_adapter_normalizes_response_and_seed() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "weather",
                                        "arguments": "{\"city\":\"Paris\"}",
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {
                        "cached_tokens": 2,
                        "cache_write_tokens": 3,
                    },
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            name="test", base_url="https://example.test/v1", api_key="secret", client=client
        )
        result = await provider.complete(
            TraceRequest(messages=[Message(role="user", content="weather")], seed=7),
            "test-model",
        )

    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert '"seed":7' in str(seen["body"])
    assert result.response.tool_calls[0].name == "weather"
    assert result.response.tool_calls[0].arguments == {"city": "Paris"}
    assert result.usage.cached_input_tokens == 2
    assert result.usage.cache_write_tokens == 3


@pytest.mark.asyncio
async def test_anthropic_adapter_normalizes_text_tool_and_usage() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg-1",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"q": "x"}},
                ],
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 2,
                    "cache_creation_input_tokens": 7,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 3,
                        "ephemeral_1h_input_tokens": 2,
                    },
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider("secret", client=client)
        result = await provider.complete(
            TraceRequest(messages=[Message(role="user", content="look up x")]),
            "claude-sonnet-4-20250514",
        )

    assert result.response.content == "Checking."
    assert result.response.tool_calls[0].arguments == {"q": "x"}
    assert result.usage == TokenUsage(
        input_tokens=17,
        output_tokens=5,
        cached_input_tokens=2,
        cache_write_tokens=5,
        cache_write_1h_tokens=2,
    )


@pytest.mark.asyncio
async def test_google_adapter_normalizes_function_call_and_usage() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "responseId": "google-1",
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"text": "Done."},
                                {"functionCall": {"name": "lookup", "args": {"q": "x"}}},
                            ]
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 6,
                    "candidatesTokenCount": 2,
                    "cachedContentTokenCount": 1,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleProvider("secret", client=client)
        result = await provider.complete(
            TraceRequest(messages=[Message(role="user", content="look up x")]),
            "gemini-2.5-flash",
        )

    assert result.response.content == "Done."
    assert result.response.tool_calls[0].name == "lookup"
    assert result.usage.input_tokens == 6

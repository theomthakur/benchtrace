"""Shared HTTP mechanics for REST provider adapters."""

from __future__ import annotations

from collections.abc import Mapping

import httpx

from benchtrace.providers.base import Provider, ProviderConfig, ProviderError
from benchtrace.trace import JsonValue, json_value


class HTTPProvider(Provider):
    """Provider base that owns an injectable async HTTP client."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        config: ProviderConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self._owns_client = client is None

    async def __aenter__(self) -> HTTPProvider:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close an internally-created HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def _post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: dict[str, JsonValue],
    ) -> tuple[JsonValue, Mapping[str, str]]:
        try:
            response = await self._client.post(url, headers=headers, json=payload)
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            raise ProviderError(str(error), retryable=True) from error
        if response.is_error:
            retryable = response.status_code == 429 or response.status_code >= 500
            detail = response.text[:500]
            raise ProviderError(
                f"provider returned HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
                headers=response.headers,
                retryable=retryable,
            )
        try:
            return json_value(response.json()), response.headers
        except ValueError as error:
            raise ProviderError("provider returned invalid JSON", retryable=False) from error


def mapping(value: object, field: str) -> dict[str, object]:
    """Require a JSON object while parsing a provider response."""

    if not isinstance(value, dict):
        raise ProviderError(f"provider response field {field!r} must be an object")
    return {str(key): item for key, item in value.items()}


def sequence(value: object, field: str) -> list[object]:
    """Require a JSON array while parsing a provider response."""

    if not isinstance(value, list):
        raise ProviderError(f"provider response field {field!r} must be an array")
    return list(value)


def integer(value: object, field: str, default: int = 0) -> int:
    """Parse a non-negative integer usage field."""

    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProviderError(f"provider response field {field!r} must be numeric")
    return max(int(value), 0)


def text(value: object, field: str, default: str = "") -> str:
    """Parse an optional string response field."""

    if value is None:
        return default
    if not isinstance(value, str):
        raise ProviderError(f"provider response field {field!r} must be a string")
    return value

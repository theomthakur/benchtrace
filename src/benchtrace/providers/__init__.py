"""Provider adapters and registry helpers."""

from __future__ import annotations

import os

from benchtrace.providers.anthropic import AnthropicProvider
from benchtrace.providers.base import Provider, ProviderConfig, ProviderError, ProviderResult
from benchtrace.providers.google import GoogleProvider
from benchtrace.providers.openai import OpenAICompatibleProvider, OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "GoogleProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderConfig",
    "ProviderError",
    "ProviderResult",
    "provider_for_model",
]


def provider_for_model(model: str, *, max_concurrency: int = 4) -> Provider:
    """Construct a provider from a model prefix and standard environment keys."""

    config = ProviderConfig(max_concurrency=max_concurrency)
    if model.startswith(("claude-", "anthropic/")):
        return AnthropicProvider(os.environ.get("ANTHROPIC_API_KEY", ""), config=config)
    if model.startswith(("gemini-", "google/")):
        return GoogleProvider(os.environ.get("GOOGLE_API_KEY", ""), config=config)
    if model.startswith("together/"):
        return OpenAICompatibleProvider(
            name="together",
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            config=config,
        )
    if model.startswith("groq/"):
        return OpenAICompatibleProvider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY", ""),
            config=config,
        )
    if model.startswith("openrouter/"):
        return OpenAICompatibleProvider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            config=config,
        )
    return OpenAIProvider(os.environ.get("OPENAI_API_KEY", ""), config=config)

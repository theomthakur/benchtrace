"""OpenAI and OpenAI-compatible chat-completions adapters."""

from __future__ import annotations

import json

import httpx

from benchtrace.providers.base import ProviderConfig, WireResult
from benchtrace.providers.http import HTTPProvider, integer, mapping, sequence, text
from benchtrace.trace import (
    JsonValue,
    TokenUsage,
    ToolCall,
    TraceRequest,
    TraceResponse,
    json_mapping,
)


class OpenAICompatibleProvider(HTTPProvider):
    """Generic adapter used by OpenAI, Together, Groq, and OpenRouter."""

    supports_seed = True

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        config: ProviderConfig | None = None,
    ) -> None:
        super().__init__(client=client, config=config)
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra_headers = extra_headers or {}

    async def _send_once(self, request: TraceRequest, model: str) -> WireResult:
        messages: list[JsonValue] = []
        for message in request.messages:
            normalized: dict[str, JsonValue] = {
                "role": message.role,
                "content": message.content,
            }
            if message.name:
                normalized["name"] = message.name
            if message.tool_call_id:
                normalized["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                normalized["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, separators=(",", ":")),
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(normalized)
        api_model = model.split("/", 1)[1] if model.startswith(f"{self.name}/") else model
        payload: dict[str, JsonValue] = {"model": api_model, "messages": messages}
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        headers = {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}
        raw, _ = await self._post(
            f"{self.base_url}/chat/completions", headers=headers, payload=payload
        )
        root = mapping(raw, "root")
        choices = sequence(root.get("choices"), "choices")
        if not choices:
            raise ValueError("provider returned no choices")
        choice = mapping(choices[0], "choices[0]")
        response_message = mapping(choice.get("message"), "message")
        calls: list[ToolCall] = []
        for index, item in enumerate(
            sequence(response_message.get("tool_calls", []), "tool_calls")
        ):
            call = mapping(item, f"tool_calls[{index}]")
            function = mapping(call.get("function"), "function")

            argument_text = text(function.get("arguments"), "arguments", "{}")
            try:
                arguments_raw = json.loads(argument_text)
            except json.JSONDecodeError as error:
                raise ValueError("tool arguments were not valid JSON") from error
            calls.append(
                ToolCall(
                    id=text(call.get("id"), "id", f"call-{index}"),
                    name=text(function.get("name"), "name"),
                    arguments=json_mapping(mapping(arguments_raw, "arguments")),
                )
            )
        usage_raw = mapping(root.get("usage", {}), "usage")
        details = mapping(usage_raw.get("prompt_tokens_details", {}), "prompt_tokens_details")
        usage = TokenUsage(
            input_tokens=integer(usage_raw.get("prompt_tokens"), "prompt_tokens"),
            output_tokens=integer(usage_raw.get("completion_tokens"), "completion_tokens"),
            cached_input_tokens=integer(details.get("cached_tokens"), "cached_tokens"),
        )
        return WireResult(
            response=TraceResponse(
                content=text(response_message.get("content"), "content"),
                tool_calls=calls,
                finish_reason=text(choice.get("finish_reason"), "finish_reason") or None,
                provider_response_id=text(root.get("id"), "id") or None,
            ),
            usage=usage,
            raw=raw,
        )


class OpenAIProvider(OpenAICompatibleProvider):
    """Official OpenAI API adapter."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        config: ProviderConfig | None = None,
    ) -> None:
        super().__init__(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            client=client,
            config=config,
        )

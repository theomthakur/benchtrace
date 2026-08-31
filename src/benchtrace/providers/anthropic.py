"""Anthropic Messages API adapter."""

from __future__ import annotations

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


class AnthropicProvider(HTTPProvider):
    """Adapter for Anthropic's versioned Messages REST API."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        config: ProviderConfig | None = None,
    ) -> None:
        super().__init__(client=client, config=config)
        self.api_key = api_key

    async def _send_once(self, request: TraceRequest, model: str) -> WireResult:
        system_parts = [message.content for message in request.messages if message.role == "system"]
        messages: list[JsonValue] = []
        for message in request.messages:
            if message.role == "system":
                continue
            if message.role == "assistant":
                blocks: list[JsonValue] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                )
                messages.append({"role": "assistant", "content": blocks})
            elif message.role == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id or "",
                                "content": message.content,
                            }
                        ],
                    }
                )
            else:
                messages.append({"role": "user", "content": message.content})
        api_model = model.split("/", 1)[1] if model.startswith("anthropic/") else model
        payload: dict[str, JsonValue] = {
            "model": api_model,
            "max_tokens": request.max_tokens or 1024,
            "messages": messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in request.tools
            ]
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        raw, _ = await self._post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            payload=payload,
        )
        root = mapping(raw, "root")
        content = sequence(root.get("content"), "content")
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for index, item in enumerate(content):
            block = mapping(item, f"content[{index}]")
            block_type = text(block.get("type"), "type")
            if block_type == "text":
                text_parts.append(text(block.get("text"), "text"))
            elif block_type == "tool_use":
                calls.append(
                    ToolCall(
                        id=text(block.get("id"), "id", f"call-{index}"),
                        name=text(block.get("name"), "name"),
                        arguments=json_mapping(mapping(block.get("input", {}), "input")),
                    )
                )
        usage_raw = mapping(root.get("usage", {}), "usage")
        usage = TokenUsage(
            input_tokens=integer(usage_raw.get("input_tokens"), "input_tokens"),
            output_tokens=integer(usage_raw.get("output_tokens"), "output_tokens"),
            cached_input_tokens=integer(
                usage_raw.get("cache_read_input_tokens"), "cache_read_input_tokens"
            ),
        )
        return WireResult(
            response=TraceResponse(
                content="".join(text_parts),
                tool_calls=calls,
                finish_reason=text(root.get("stop_reason"), "stop_reason") or None,
                provider_response_id=text(root.get("id"), "id") or None,
            ),
            usage=usage,
            raw=raw,
        )

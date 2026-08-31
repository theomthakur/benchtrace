"""Google Gemini generateContent adapter."""

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


class GoogleProvider(HTTPProvider):
    """Adapter for the Google Generative Language REST API."""

    name = "google"
    supports_seed = False

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
        contents: list[JsonValue] = []
        for message in request.messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.name or "tool",
                                    "response": {"result": message.content},
                                }
                            }
                        ],
                    }
                )
                continue
            parts: list[JsonValue] = []
            if message.content:
                parts.append({"text": message.content})
            parts.extend(
                {"functionCall": {"name": call.name, "args": call.arguments}}
                for call in message.tool_calls
            )
            contents.append(
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": parts,
                }
            )
        payload: dict[str, JsonValue] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in request.tools
                    ]
                }
            ]
        generation: dict[str, JsonValue] = {}
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation["maxOutputTokens"] = request.max_tokens
        if generation:
            payload["generationConfig"] = generation
        api_model = model.split("/", 1)[1] if model.startswith("google/") else model
        raw, _ = await self._post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{api_model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            payload=payload,
        )
        root = mapping(raw, "root")
        candidates = sequence(root.get("candidates"), "candidates")
        if not candidates:
            raise ValueError("provider returned no candidates")
        candidate = mapping(candidates[0], "candidates[0]")
        content = mapping(candidate.get("content"), "content")
        response_parts = sequence(content.get("parts"), "parts")
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for index, item in enumerate(response_parts):
            part = mapping(item, f"parts[{index}]")
            if "text" in part:
                text_parts.append(text(part["text"], "text"))
            if "functionCall" in part:
                call = mapping(part["functionCall"], "functionCall")
                calls.append(
                    ToolCall(
                        id=f"call-{index}",
                        name=text(call.get("name"), "name"),
                        arguments=json_mapping(mapping(call.get("args", {}), "args")),
                    )
                )
        usage_raw = mapping(root.get("usageMetadata", {}), "usageMetadata")
        usage = TokenUsage(
            input_tokens=integer(usage_raw.get("promptTokenCount"), "promptTokenCount"),
            output_tokens=integer(
                usage_raw.get("candidatesTokenCount"), "candidatesTokenCount"
            ),
            cached_input_tokens=integer(
                usage_raw.get("cachedContentTokenCount"), "cachedContentTokenCount"
            ),
        )
        return WireResult(
            response=TraceResponse(
                content="".join(text_parts),
                tool_calls=calls,
                finish_reason=text(candidate.get("finishReason"), "finishReason") or None,
                provider_response_id=text(root.get("responseId"), "responseId") or None,
            ),
            usage=usage,
            raw=raw,
        )

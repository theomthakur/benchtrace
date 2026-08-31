from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchtrace.graders import build_grader
from benchtrace.providers.base import Provider, WireResult
from benchtrace.task import (
    ExactMatchSpec,
    JsonSchemaSpec,
    LLMJudgeSpec,
    NumericSpec,
    RegexSpec,
    ToolCallSpec,
)
from benchtrace.trace import (
    Message,
    TokenUsage,
    ToolCall,
    TraceRecord,
    TraceRequest,
    TraceResponse,
)


def trace_with(
    content: str,
    *,
    expected: dict[str, object],
    tool_calls: list[ToolCall] | None = None,
    model: str = "model-under-test",
) -> TraceRecord:
    return TraceRecord.model_validate(
        {
            "run_id": "run",
            "task_id": "task",
            "case_id": "case",
            "provider": "test",
            "model": model,
            "started_at": datetime.now(UTC),
            "request": TraceRequest(
                messages=[Message(role="user", content="question")],
                expected=expected,
            ),
            "response": TraceResponse(content=content, tool_calls=tool_calls or []),
            "status": "completed",
        }
    )


@pytest.mark.asyncio
async def test_exact_match_always_returns_score_and_reason() -> None:
    grader = build_grader(ExactMatchSpec(type="exact_match", case_sensitive=False))

    result = await grader.grade(trace_with(" YES \n", expected={"answer": "yes"}))

    assert result.output.score == 1.0
    assert result.output.reason == "answer matched exactly"


@pytest.mark.asyncio
async def test_regex_flags_and_failure_reason() -> None:
    grader = build_grader(RegexSpec(type="regex", flags=["IGNORECASE"]))

    passed = await grader.grade(trace_with("Status: OK", expected={"pattern": "^status: ok$"}))
    failed = await grader.grade(trace_with("no", expected={"pattern": "^yes$"}))

    assert passed.output.score == 1.0
    assert failed.output.score == 0.0
    assert "did not match" in failed.output.reason


@pytest.mark.asyncio
async def test_numeric_absolute_and_relative_tolerance() -> None:
    grader = build_grader(
        NumericSpec(type="numeric", absolute_tolerance=0.01, relative_tolerance=0.0)
    )

    close = await grader.grade(trace_with("3.145", expected={"answer": 3.14}))
    invalid = await grader.grade(trace_with("about three", expected={"answer": 3.14}))

    assert close.output.score == 1.0
    assert invalid.output.score == 0.0
    assert "not a number" in invalid.output.reason


@pytest.mark.asyncio
async def test_json_schema_conformance_reports_path() -> None:
    grader = build_grader(JsonSchemaSpec(type="json_schema"))
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 1}},
        "required": ["count"],
        "additionalProperties": False,
    }

    valid = await grader.grade(trace_with('{"count": 2}', expected={"schema": schema}))
    invalid = await grader.grade(trace_with('{"count": 0}', expected={"schema": schema}))

    assert valid.output.score == 1.0
    assert invalid.output.score == 0.0
    assert "count" in invalid.output.reason


@pytest.mark.asyncio
async def test_tool_call_grader_checks_tool_and_arguments() -> None:
    grader = build_grader(ToolCallSpec(type="tool_call"))
    expected = {"tool_calls": [{"name": "weather", "arguments": {"city": "Paris"}}]}

    passed = await grader.grade(
        trace_with(
            "",
            expected=expected,
            tool_calls=[ToolCall(name="weather", arguments={"city": "Paris"})],
        )
    )
    wrong = await grader.grade(
        trace_with(
            "",
            expected=expected,
            tool_calls=[ToolCall(name="weather", arguments={"city": "London"})],
        )
    )

    assert passed.output.score == 1.0
    assert wrong.output.score == 0.0
    assert "matched 0" in wrong.output.reason


class JudgeProvider(Provider):
    name = "judge-provider"

    async def _send_once(self, request: TraceRequest, model: str) -> WireResult:
        del request, model
        return WireResult(
            response=TraceResponse(content='{"score": 0.75, "reason": "mostly correct"}'),
            usage=TokenUsage(input_tokens=20, output_tokens=8),
            raw={},
        )


@pytest.mark.asyncio
async def test_llm_judge_can_use_different_provider_and_records_call() -> None:
    grader = build_grader(
        LLMJudgeSpec(type="llm_judge", rubric="Be correct", model="judge-model"),
        judge_provider=JudgeProvider(),
    )

    result = await grader.grade(trace_with("answer", expected={}))

    assert result.output.score == 0.75
    assert result.output.reason == "mostly correct"
    assert result.model_calls[0].purpose == "judge"
    assert result.model_calls[0].usage.total_tokens == 28


@pytest.mark.asyncio
async def test_llm_judge_refuses_same_model_unless_forced() -> None:
    grader = build_grader(
        LLMJudgeSpec(type="llm_judge", rubric="Be correct", model="same-model"),
        judge_provider=JudgeProvider(),
    )

    with pytest.raises(ValueError, match="refuses to grade the same model"):
        await grader.grade(trace_with("answer", expected={}, model="same-model"))

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchtrace.trace import (
    Failure,
    GraderOutput,
    Message,
    TraceRecord,
    TraceRequest,
    TraceResponse,
    TraceValidationError,
    append_trace,
    load_traces,
)


def make_trace(**overrides: object) -> TraceRecord:
    values: dict[str, object] = {
        "run_id": "run-1",
        "task_id": "arithmetic",
        "case_id": "case-1",
        "provider": "openai",
        "model": "test-model",
        "started_at": datetime.now(UTC),
        "completed_at": datetime.now(UTC) + timedelta(milliseconds=10),
        "request": TraceRequest(messages=[Message(role="user", content="2 + 2?")]),
        "response": TraceResponse(content="4"),
        "latency_ms": 10.0,
        "grader_outputs": [GraderOutput(grader="exact_match", score=1.0, reason="matched")],
        "status": "completed",
    }
    values.update(overrides)
    return TraceRecord.model_validate(values)


def test_trace_round_trip_jsonl(tmp_path: Path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    expected = make_trace()

    append_trace(trace_file, expected)

    assert load_traces(trace_file) == [expected]
    assert load_traces(tmp_path) == [expected]


def test_trace_rejects_unknown_fields_and_out_of_range_scores() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        make_trace(unknown="not allowed")

    with pytest.raises(ValidationError, match="less_than_equal"):
        GraderOutput(grader="bad", score=1.1, reason="invalid")


def test_failed_trace_requires_failure_and_completed_trace_forbids_it() -> None:
    with pytest.raises(ValidationError, match="failed traces require failure details"):
        make_trace(status="failed", response=None, grader_outputs=[])

    with pytest.raises(ValidationError, match="completed traces cannot contain failure details"):
        make_trace(failure=Failure(kind="wrong_answer", message="no"))


def test_trace_rejects_naive_and_backwards_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        make_trace(started_at=datetime.now())

    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="cannot precede"):
        make_trace(started_at=now, completed_at=now - timedelta(seconds=1))


def test_trace_validation_error_names_file_and_line(tmp_path: Path) -> None:
    trace_file = tmp_path / "broken.jsonl"
    trace_file.write_text(make_trace().model_dump_json() + "\n{}\n", encoding="utf-8")

    with pytest.raises(TraceValidationError) as captured:
        load_traces(trace_file)

    assert f"{trace_file}:2:" in str(captured.value)


def test_passed_requires_all_graders_to_pass() -> None:
    passing = make_trace()
    partial = make_trace(
        grader_outputs=[GraderOutput(grader="numeric", score=0.5, reason="close")]
    )

    assert passing.passed
    assert not partial.passed
    assert partial.aggregate_score == 0.5

"""Canonical, versioned trace schema and JSONL persistence.

Every benchtrace capability consumes :class:`TraceRecord`. Extend this schema when new
recorded data is needed; do not introduce parallel run-result formats elsewhere.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic import JsonValue as PydanticJsonValue

SCHEMA_VERSION: Literal["1.0"] = "1.0"
JsonValue: TypeAlias = PydanticJsonValue
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class TraceModel(BaseModel):
    """Strict base class shared by every object in the trace contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Message(TraceModel):
    """A normalized chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolDefinition(TraceModel):
    """A tool exposed to an agent, represented with JSON Schema arguments."""

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, JsonValue]


class ToolCall(TraceModel):
    """A normalized tool invocation returned by a provider."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


class TokenUsage(TraceModel):
    """Provider-reported token accounting."""

    # input_tokens is the complete billable input count. Cached reads and cache
    # writes are subsets used to apply the correct rate.
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    cache_write_1h_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Return the total non-cached input and output token count."""

        return self.input_tokens + self.output_tokens


class TraceRequest(TraceModel):
    """Fully rendered request plus benchmark inputs and expectations."""

    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    case_input: dict[str, JsonValue] = Field(default_factory=dict)
    expected: dict[str, JsonValue] = Field(default_factory=dict)
    relevant_context_ids: list[str] = Field(default_factory=list)
    seed: int | None = None
    temperature: float | None = Field(default=None, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)


class TraceResponse(TraceModel):
    """Normalized model response."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    provider_response_id: str | None = None


class ModelCall(TraceModel):
    """One provider call made while producing or grading a case."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    purpose: Literal["agent", "judge"]
    request: TraceRequest
    response: TraceResponse
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=1, ge=1)
    pricing_known: bool = True
    pricing_version: str | None = None


class RetrievedContext(TraceModel):
    """A retrieved unit made available to the model."""

    id: str = Field(min_length=1)
    text: str
    score: float | None = None
    source: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    relevant: bool | None = None


class Citation(TraceModel):
    """A claim-to-context citation emitted in the answer."""

    claim: str = Field(min_length=1)
    context_ids: list[str] = Field(min_length=1)
    supported: bool | None = None
    reason: str | None = None


class GraderOutput(TraceModel):
    """A grader result with a bounded score and an actionable reason."""

    grader: str = Field(min_length=1)
    score: Score
    reason: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TrajectoryStep(TraceModel):
    """One observe/plan/act transition in an agent trajectory."""

    step: int = Field(ge=0)
    observation: JsonValue = None
    plan: str | None = None
    action: ToolCall | None = None
    tool_result: JsonValue = None
    state_hash: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)


class Failure(TraceModel):
    """A normalized case-level failure."""

    kind: Literal[
        "provider_error",
        "grader_error",
        "tool_error",
        "wrong_answer",
        "thrash",
        "budget_exhausted",
        "invalid_response",
    ]
    message: str
    retryable: bool = False


class TraceRecord(TraceModel):
    """Single source of truth for one completed or failed benchmark case."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    trace_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    repeat: int = Field(default=0, ge=0)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    request: TraceRequest
    response: TraceResponse | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = Field(default=0.0, ge=0.0)
    model_calls: list[ModelCall] = Field(default_factory=list)
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    grader_outputs: list[GraderOutput] = Field(default_factory=list)
    status: Literal["completed", "failed"]
    failure: Failure | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("started_at", "completed_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        """Reject ambiguous timestamps."""

        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> TraceRecord:
        """Keep failure state and chronological fields internally consistent."""

        if self.status == "failed" and self.failure is None:
            raise ValueError("failed traces require failure details")
        if self.status == "completed" and self.failure is not None:
            raise ValueError("completed traces cannot contain failure details")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self

    @property
    def key(self) -> tuple[str, str, str, int]:
        """Stable identity used by resumable runs."""

        return (self.task_id, self.case_id, self.model, self.repeat)

    @property
    def aggregate_score(self) -> float | None:
        """Return the mean grader score, or ``None`` when no grader ran."""

        if not self.grader_outputs:
            return None
        return sum(result.score for result in self.grader_outputs) / len(self.grader_outputs)

    @property
    def passed(self) -> bool:
        """A case passes only when completed and all graders score 1."""

        return self.status == "completed" and bool(self.grader_outputs) and all(
            result.score == 1.0 for result in self.grader_outputs
        )


class TraceValidationError(ValueError):
    """Validation error carrying the source JSONL path and line."""

    def __init__(self, path: Path, line: int, detail: str) -> None:
        super().__init__(f"{path}:{line}: {detail}")
        self.path = path
        self.line = line


_append_lock = threading.Lock()


def append_trace(path: str | Path, record: TraceRecord) -> None:
    """Append one validated record as a single durable JSONL write."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (record.model_dump_json() + "\n").encode("utf-8")
    with _append_lock:
        descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(f"short trace write: wrote {written} of {len(payload)} bytes")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def iter_traces(path: str | Path) -> Iterator[TraceRecord]:
    """Stream and validate records from one JSONL file."""

    source = Path(path)
    try:
        stream = source.open(encoding="utf-8")
    except OSError as error:
        raise TraceValidationError(source, 0, str(error)) from error
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                yield TraceRecord.model_validate_json(raw_line)
            except (ValidationError, ValueError, json.JSONDecodeError) as error:
                raise TraceValidationError(source, line_number, str(error)) from error


def load_traces(path: str | Path) -> list[TraceRecord]:
    """Load all records from a JSONL file or all ``*.jsonl`` files in a run directory."""

    source = Path(path)
    if source.is_file():
        return list(iter_traces(source))
    if not source.is_dir():
        raise TraceValidationError(source, 0, "trace path does not exist")
    records: list[TraceRecord] = []
    for trace_file in sorted(source.glob("*.jsonl")):
        records.extend(iter_traces(trace_file))
    return records


def completed_keys(records: Iterable[TraceRecord]) -> set[tuple[str, str, str, int]]:
    """Return stable keys for traces already persisted, including isolated failures."""

    return {record.key for record in records}


def json_value(value: Any) -> JsonValue:
    """Validate and normalize an arbitrary object as a JSON-compatible value."""

    class _JsonAdapter(BaseModel):
        value: JsonValue

    return _JsonAdapter.model_validate({"value": value}).value


def json_mapping(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Validate and normalize a mapping for a trace metadata field."""

    return {str(key): json_value(item) for key, item in value.items()}

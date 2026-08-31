"""Deterministic graders over canonical traces."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from jsonschema import Draft202012Validator

from benchtrace.graders.base import GradingResult
from benchtrace.task import (
    ExactMatchSpec,
    JsonSchemaSpec,
    NumericSpec,
    RegexSpec,
    ToolCallSpec,
)
from benchtrace.trace import GraderOutput, JsonValue, ToolCall, TraceRecord


def _answer(trace: TraceRecord) -> str:
    return trace.response.content if trace.response is not None else ""


def _expected(trace: TraceRecord, key: str) -> JsonValue:
    if key not in trace.request.expected:
        raise ValueError(f"trace expected data has no {key!r} field")
    return trace.request.expected[key]


@dataclass(frozen=True, slots=True)
class ExactMatchGrader:
    spec: ExactMatchSpec

    async def grade(self, trace: TraceRecord) -> GradingResult:
        expected = _expected(trace, self.spec.expected)
        if not isinstance(expected, str):
            raise ValueError("exact-match expected value must be a string")
        actual = _answer(trace)
        if self.spec.strip:
            expected, actual = expected.strip(), actual.strip()
        if not self.spec.case_sensitive:
            expected, actual = expected.casefold(), actual.casefold()
        matched = actual == expected
        reason = "answer matched exactly" if matched else f"expected {expected!r}, got {actual!r}"
        return GradingResult(
            GraderOutput(grader="exact_match", score=float(matched), reason=reason)
        )


@dataclass(frozen=True, slots=True)
class RegexGrader:
    spec: RegexSpec

    async def grade(self, trace: TraceRecord) -> GradingResult:
        expected = _expected(trace, self.spec.expected)
        if not isinstance(expected, str):
            raise ValueError("regex pattern must be a string")
        flags = 0
        for flag in self.spec.flags:
            flags |= getattr(re, flag)
        matched = re.search(expected, _answer(trace), flags) is not None
        reason = f"answer matched /{expected}/" if matched else f"answer did not match /{expected}/"
        return GradingResult(GraderOutput(grader="regex", score=float(matched), reason=reason))


@dataclass(frozen=True, slots=True)
class NumericGrader:
    spec: NumericSpec

    async def grade(self, trace: TraceRecord) -> GradingResult:
        expected_raw = _expected(trace, self.spec.expected)
        if isinstance(expected_raw, bool) or not isinstance(expected_raw, int | float):
            raise ValueError("numeric expected value must be a number")
        try:
            actual = float(_answer(trace).strip())
        except ValueError:
            return GradingResult(
                GraderOutput(
                    grader="numeric",
                    score=0.0,
                    reason=f"answer {_answer(trace)!r} is not a number",
                )
            )
        expected = float(expected_raw)
        matched = math.isclose(
            actual,
            expected,
            rel_tol=self.spec.relative_tolerance,
            abs_tol=self.spec.absolute_tolerance,
        )
        delta = abs(actual - expected)
        reason = (
            f"numeric answer within tolerance (absolute error {delta:g})"
            if matched
            else f"absolute error {delta:g} exceeds tolerance"
        )
        return GradingResult(GraderOutput(grader="numeric", score=float(matched), reason=reason))


@dataclass(frozen=True, slots=True)
class JsonSchemaGrader:
    spec: JsonSchemaSpec

    async def grade(self, trace: TraceRecord) -> GradingResult:
        schema = _expected(trace, self.spec.expected)
        if not isinstance(schema, dict):
            raise ValueError("JSON-schema expected value must be an object")
        try:
            instance: object = json.loads(_answer(trace))
        except json.JSONDecodeError as error:
            return GradingResult(
                GraderOutput(grader="json_schema", score=0.0, reason=f"invalid JSON: {error.msg}")
            )
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda item: list(item.path))
        if not errors:
            return GradingResult(
                GraderOutput(grader="json_schema", score=1.0, reason="valid against JSON schema")
            )
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "$"
        return GradingResult(
            GraderOutput(
                grader="json_schema",
                score=0.0,
                reason=f"schema violation at {location}: {first.message}",
            )
        )


def _all_tool_calls(trace: TraceRecord) -> list[ToolCall]:
    calls = [step.action for step in trace.trajectory if step.action is not None]
    if trace.response is not None:
        calls.extend(trace.response.tool_calls)
    return calls


@dataclass(frozen=True, slots=True)
class ToolCallGrader:
    spec: ToolCallSpec

    async def grade(self, trace: TraceRecord) -> GradingResult:
        expected_raw = _expected(trace, self.spec.expected)
        if not isinstance(expected_raw, list):
            raise ValueError("tool-call expected value must be a list")
        expected: list[tuple[str, dict[str, JsonValue]]] = []
        for index, item in enumerate(expected_raw):
            if not isinstance(item, dict):
                raise ValueError(f"expected tool call {index} must be an object")
            name, arguments = item.get("name"), item.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError(f"expected tool call {index} has invalid name or arguments")
            expected.append((name, arguments))
        actual = _all_tool_calls(trace)
        matched = sum(
            1
            for index, (name, arguments) in enumerate(expected)
            if (
                index < len(actual)
                and actual[index].name == name
                and actual[index].arguments == arguments
            )
        )
        observed_count = len(expected) if self.spec.allow_extra else len(actual)
        denominator = max(len(expected), observed_count, 1)
        score = matched / denominator
        reason = (
            f"all {len(expected)} expected tool calls matched"
            if score == 1.0
            else f"matched {matched} of {len(expected)} expected calls; observed {len(actual)}"
        )
        return GradingResult(GraderOutput(grader="tool_call", score=score, reason=reason))

"""Built-in grader construction."""

from __future__ import annotations

from benchtrace.graders.base import Grader, GradingResult
from benchtrace.graders.builtin import (
    ExactMatchGrader,
    JsonSchemaGrader,
    NumericGrader,
    RegexGrader,
    ToolCallGrader,
)
from benchtrace.graders.llm import LLMJudgeGrader
from benchtrace.providers import Provider
from benchtrace.task import (
    ExactMatchSpec,
    GraderSpec,
    JsonSchemaSpec,
    LLMJudgeSpec,
    NumericSpec,
    RegexSpec,
    ToolCallSpec,
)

__all__ = [
    "Grader",
    "GradingResult",
    "build_grader",
]


def build_grader(spec: GraderSpec, *, judge_provider: Provider | None = None) -> Grader:
    """Build a grader from validated task data."""

    if isinstance(spec, ExactMatchSpec):
        return ExactMatchGrader(spec)
    if isinstance(spec, RegexSpec):
        return RegexGrader(spec)
    if isinstance(spec, NumericSpec):
        return NumericGrader(spec)
    if isinstance(spec, JsonSchemaSpec):
        return JsonSchemaGrader(spec)
    if isinstance(spec, ToolCallSpec):
        return ToolCallGrader(spec)
    if isinstance(spec, LLMJudgeSpec):
        if judge_provider is None:
            raise ValueError("llm_judge requires a judge provider")
        return LLMJudgeGrader(spec, judge_provider)
    raise TypeError(f"unsupported grader spec: {type(spec).__name__}")

"""Grader protocol and normalized execution result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from benchtrace.trace import GraderOutput, ModelCall, TraceRecord


@dataclass(frozen=True, slots=True)
class GradingResult:
    """A grader output plus any judge calls that must be retained in the trace."""

    output: GraderOutput
    model_calls: tuple[ModelCall, ...] = ()


class Grader(Protocol):
    """Every grader consumes only the canonical trace record."""

    async def grade(self, trace: TraceRecord) -> GradingResult:
        """Return a bounded score and human-readable reason."""
        ...

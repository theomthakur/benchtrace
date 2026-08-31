"""Rubric-based LLM judge grader."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from benchtrace.graders.base import GradingResult
from benchtrace.providers import Provider
from benchtrace.task import LLMJudgeSpec
from benchtrace.trace import (
    GraderOutput,
    Message,
    ModelCall,
    TraceRecord,
    TraceRequest,
)


class JudgeResponse(BaseModel):
    """Runtime-validated response contract requested from the judge."""

    model_config = ConfigDict(extra="forbid")
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class LLMJudgeGrader:
    """Judge a trace with an independently selectable provider and model."""

    spec: LLMJudgeSpec
    provider: Provider

    async def grade(self, trace: TraceRecord) -> GradingResult:
        if trace.model == self.spec.model and not self.spec.force_same_model:
            raise ValueError(
                "LLM judge refuses to grade the same model; set force_same_model=true to override"
            )
        response_content = trace.response.content if trace.response else ""
        trajectory_json = json.dumps(
            [step.model_dump(mode="json") for step in trace.trajectory]
        )
        judge_request = TraceRequest(
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are an evaluation judge. Return only JSON with keys score (0 to 1) "
                        "and reason (a concise explanation)."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Rubric:\n{self.spec.rubric}\n\n"
                        f"Request:\n{trace.request.model_dump_json()}\n\n"
                        f"Answer:\n{response_content}\n\n"
                        f"Trajectory:\n{trajectory_json}"
                    ),
                ),
            ],
            temperature=0.0,
        )
        result = await self.provider.complete(judge_request, self.spec.model)
        try:
            judged = JudgeResponse.model_validate_json(result.response.content)
        except ValidationError as error:
            raise ValueError(f"judge returned invalid score JSON: {error}") from error
        call = ModelCall(
            provider=self.provider.name,
            model=self.spec.model,
            purpose="judge",
            request=judge_request,
            response=result.response,
            usage=result.usage,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            attempts=result.attempts,
            pricing_known=result.pricing_known,
            pricing_version=result.pricing_version,
        )
        return GradingResult(
            output=GraderOutput(grader="llm_judge", score=judged.score, reason=judged.reason),
            model_calls=(call,),
        )

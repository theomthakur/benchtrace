"""Async, resumable benchmark runner with agent-loop evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from benchtrace.graders import build_grader
from benchtrace.providers import Provider, ProviderError
from benchtrace.providers.base import ProviderResult
from benchtrace.task import LLMJudgeSpec, TaskCase, TaskDefinition
from benchtrace.trace import (
    Failure,
    GraderOutput,
    JsonValue,
    Message,
    ModelCall,
    TokenUsage,
    ToolCall,
    TraceRecord,
    TraceRequest,
    TraceResponse,
    TrajectoryStep,
    append_trace,
    completed_keys,
    json_value,
    load_traces,
)

ToolFunction = Callable[
    [Mapping[str, JsonValue], Mapping[str, JsonValue]],
    Awaitable[JsonValue],
]


class ToolExecutor(Protocol):
    """Execution boundary for tools exposed to an evaluated agent."""

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        case_input: Mapping[str, JsonValue],
    ) -> JsonValue:
        """Execute a named tool and return JSON-compatible observed state."""
        ...


class ToolRegistry:
    """Tool functions plus a data-defined fixture fallback for committed suites."""

    def __init__(self, tools: Mapping[str, ToolFunction] | None = None) -> None:
        self._tools = dict(tools or {})

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        case_input: Mapping[str, JsonValue],
    ) -> JsonValue:
        function = self._tools.get(name)
        if function is not None:
            return json_value(await function(arguments, case_input))
        fixtures = case_input.get("tool_results")
        if isinstance(fixtures, dict):
            candidates = fixtures.get(name)
            if isinstance(candidates, list):
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    if candidate.get("arguments") == dict(arguments) and "result" in candidate:
                        return json_value(candidate["result"])
        raise KeyError(f"no implementation or matching fixture for tool {name!r}")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Controls for a reproducible run."""

    output_dir: Path
    repeats: int = 1
    seed: int = 0
    force: bool = False

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    response: TraceResponse | None
    trajectory: tuple[TrajectoryStep, ...]
    model_calls: tuple[ModelCall, ...]
    failure: Failure | None


def _stable_seed(base: int, task_id: str, case_id: str, model: str, repeat: int) -> int:
    material = f"{base}\0{task_id}\0{case_id}\0{model}\0{repeat}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _state_hash(value: JsonValue) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _action_signature(call: ToolCall) -> str:
    arguments = json.dumps(call.arguments, sort_keys=True, separators=(",", ":"))
    return f"{call.name}\0{arguments}"


def _tool_message_content(result: JsonValue) -> str:
    return json.dumps(result, sort_keys=True, ensure_ascii=False)


def _model_call(
    provider: Provider,
    model: str,
    request: TraceRequest,
    result: ProviderResult,
    *,
    purpose: str = "agent",
) -> ModelCall:
    return ModelCall.model_validate(
        {
            "provider": provider.name,
            "model": model,
            "purpose": purpose,
            "request": request,
            "response": result.response,
            "usage": result.usage,
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
            "attempts": result.attempts,
            "pricing_known": result.pricing_known,
            "pricing_version": result.pricing_version,
        }
    )


def _sum_usage(calls: Sequence[ModelCall]) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(call.usage.input_tokens for call in calls),
        output_tokens=sum(call.usage.output_tokens for call in calls),
        cached_input_tokens=sum(call.usage.cached_input_tokens for call in calls),
        reasoning_tokens=sum(call.usage.reasoning_tokens for call in calls),
    )


class Runner:
    """Evaluate task cases concurrently while isolating and persisting every outcome."""

    def __init__(
        self,
        providers: Mapping[str, Provider],
        *,
        tools: ToolExecutor | None = None,
        judge_providers: Mapping[str, Provider] | None = None,
    ) -> None:
        self._providers = dict(providers)
        self._tools = tools or ToolRegistry()
        self._judge_providers = dict(judge_providers or {})

    async def run(
        self,
        tasks: Sequence[TaskDefinition],
        models: Sequence[str],
        config: RunConfig,
    ) -> list[TraceRecord]:
        """Run all requested cases and return the complete persisted run."""

        missing = sorted(set(models) - self._providers.keys())
        if missing:
            raise ValueError(f"no provider configured for models: {missing}")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        trace_file = config.output_dir / "traces.jsonl"
        existing = load_traces(trace_file) if trace_file.exists() else []
        skipped = set() if config.force else completed_keys(existing)
        run_id = existing[0].run_id if existing else f"run-{uuid4().hex}"
        work: list[Awaitable[TraceRecord]] = []
        for task in tasks:
            for case in task.cases:
                for model in models:
                    for repeat in range(config.repeats):
                        key = (task.id, case.id, model, repeat)
                        if key in skipped:
                            continue
                        work.append(
                            self._run_and_persist(
                                run_id,
                                task,
                                case,
                                model,
                                repeat,
                                config.seed,
                                trace_file,
                            )
                        )
        if work:
            await asyncio.gather(*work)
        return load_traces(trace_file) if trace_file.exists() else []

    async def _run_and_persist(
        self,
        run_id: str,
        task: TaskDefinition,
        case: TaskCase,
        model: str,
        repeat: int,
        base_seed: int,
        trace_file: Path,
    ) -> TraceRecord:
        provider = self._providers[model]
        seed = _stable_seed(base_seed, task.id, case.id, model, repeat)
        request = task.request_for(case, seed=seed if provider.supports_seed else None)
        try:
            trace = await self._evaluate_case(
                run_id, task, case, model, repeat, provider, request
            )
        except Exception as error:
            failure_kind: Literal["provider_error", "invalid_response"] = (
                "provider_error" if isinstance(error, ProviderError) else "invalid_response"
            )
            trace = TraceRecord(
                run_id=run_id,
                task_id=task.id,
                case_id=case.id,
                repeat=repeat,
                provider=provider.name,
                model=model,
                request=request,
                status="failed",
                failure=Failure(
                    kind=failure_kind,
                    message=str(error),
                    retryable=isinstance(error, ProviderError) and error.retryable,
                ),
                metadata={"unhandled_exception": type(error).__name__},
            )
        append_trace(trace_file, trace)
        return trace

    async def _evaluate_case(
        self,
        run_id: str,
        task: TaskDefinition,
        case: TaskCase,
        model: str,
        repeat: int,
        provider: Provider,
        request: TraceRequest,
    ) -> TraceRecord:
        started_at = datetime.now(UTC)
        started_clock = time.perf_counter()
        outcome = await self._agent_loop(task, case, model, provider, request)
        calls = list(outcome.model_calls)
        outputs: list[GraderOutput] = []
        failure = outcome.failure
        if failure is None:
            provisional = TraceRecord(
                run_id=run_id,
                task_id=task.id,
                case_id=case.id,
                repeat=repeat,
                provider=provider.name,
                model=model,
                started_at=started_at,
                request=request,
                response=outcome.response,
                trajectory=list(outcome.trajectory),
                model_calls=calls,
                usage=_sum_usage(calls),
                cost_usd=sum(call.cost_usd for call in calls),
                status="completed",
                metadata=case.metadata,
            )
            for spec in task.graders:
                judge_provider = None
                if isinstance(spec, LLMJudgeSpec):
                    judge_provider = self._judge_providers.get(spec.model)
                    if judge_provider is None:
                        failure = Failure(
                            kind="grader_error",
                            message=f"no judge provider configured for model {spec.model!r}",
                        )
                        break
                try:
                    grading = await build_grader(
                        spec, judge_provider=judge_provider
                    ).grade(provisional)
                except Exception as error:
                    failure = Failure(kind="grader_error", message=str(error))
                    break
                outputs.append(grading.output)
                calls.extend(grading.model_calls)
                provisional = provisional.model_copy(
                    update={
                        "grader_outputs": outputs,
                        "model_calls": calls,
                        "usage": _sum_usage(calls),
                        "cost_usd": sum(call.cost_usd for call in calls),
                    }
                )
            if failure is None and any(output.score < 1.0 for output in outputs):
                reasons = "; ".join(
                    output.reason for output in outputs if output.score < 1.0
                )
                failure = Failure(kind="wrong_answer", message=reasons)
        completed_at = datetime.now(UTC)
        pricing_missing = sorted(
            {f"{call.provider}/{call.model}" for call in calls if not call.pricing_known}
        )
        metadata: dict[str, JsonValue] = dict(case.metadata)
        if pricing_missing:
            metadata["pricing_missing"] = json_value(pricing_missing)
        return TraceRecord(
            run_id=run_id,
            task_id=task.id,
            case_id=case.id,
            repeat=repeat,
            provider=provider.name,
            model=model,
            started_at=started_at,
            completed_at=completed_at,
            request=request,
            response=outcome.response,
            latency_ms=(time.perf_counter() - started_clock) * 1000.0,
            usage=_sum_usage(calls),
            cost_usd=sum(call.cost_usd for call in calls),
            model_calls=calls,
            trajectory=list(outcome.trajectory),
            grader_outputs=outputs,
            status="failed" if failure else "completed",
            failure=failure,
            metadata=metadata,
        )

    async def _agent_loop(
        self,
        task: TaskDefinition,
        case: TaskCase,
        model: str,
        provider: Provider,
        initial_request: TraceRequest,
    ) -> AgentOutcome:
        messages = list(initial_request.messages)
        calls: list[ModelCall] = []
        trajectory: list[TrajectoryStep] = []
        observation: JsonValue = None
        resulting_states: dict[str, str] = {}
        latest_response: TraceResponse | None = None
        for _turn in range(task.step_budget):
            call_request = initial_request.model_copy(update={"messages": messages})
            result = await provider.complete(call_request, model)
            calls.append(_model_call(provider, model, call_request, result))
            latest_response = result.response
            if not result.response.tool_calls:
                return AgentOutcome(latest_response, tuple(trajectory), tuple(calls), None)
            messages.append(
                Message(
                    role="assistant",
                    content=result.response.content,
                    tool_calls=result.response.tool_calls,
                )
            )
            for tool_call in result.response.tool_calls:
                signature = _action_signature(tool_call)
                current_state = _state_hash(observation)
                if resulting_states.get(signature) == current_state:
                    trajectory.append(
                        TrajectoryStep(
                            step=len(trajectory),
                            observation=observation,
                            plan=result.response.content or None,
                            action=tool_call,
                            state_hash=current_state,
                        )
                    )
                    return AgentOutcome(
                        latest_response,
                        tuple(trajectory),
                        tuple(calls),
                        Failure(
                            kind="thrash",
                            message=f"repeated {tool_call.name!r} with unchanged observed state",
                        ),
                    )
                tool_started = time.perf_counter()
                try:
                    tool_result = await self._tools.execute(
                        tool_call.name, tool_call.arguments, case.input
                    )
                except Exception as error:
                    return AgentOutcome(
                        latest_response,
                        tuple(trajectory),
                        tuple(calls),
                        Failure(
                            kind="tool_error",
                            message=f"tool {tool_call.name!r} failed: {error}",
                        ),
                    )
                result_state = _state_hash(tool_result)
                resulting_states[signature] = result_state
                trajectory.append(
                    TrajectoryStep(
                        step=len(trajectory),
                        observation=observation,
                        plan=result.response.content or None,
                        action=tool_call,
                        tool_result=tool_result,
                        state_hash=result_state,
                        latency_ms=(time.perf_counter() - tool_started) * 1000.0,
                    )
                )
                observation = tool_result
                messages.append(
                    Message(
                        role="tool",
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                        content=_tool_message_content(tool_result),
                    )
                )
        return AgentOutcome(
            latest_response,
            tuple(trajectory),
            tuple(calls),
            Failure(
                kind="budget_exhausted",
                message=f"agent exhausted step budget of {task.step_budget}",
            ),
        )

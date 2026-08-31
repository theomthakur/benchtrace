from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchtrace.providers.base import Provider, ProviderConfig, ProviderError, WireResult
from benchtrace.runner import RunConfig, Runner
from benchtrace.task import ExactMatchSpec, TaskCase, TaskDefinition, ToolCallSpec
from benchtrace.trace import TokenUsage, ToolCall, TraceRequest, TraceResponse


class ScriptedProvider(Provider):
    name = "test"
    supports_seed = True

    def __init__(self, *, max_concurrency: int = 4) -> None:
        super().__init__(ProviderConfig(max_concurrency=max_concurrency, max_retries=0))
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def _send_once(self, request: TraceRequest, model: str) -> WireResult:
        del model
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.001)
        self.active -= 1
        mode = request.case_input.get("mode")
        if mode == "error":
            raise ProviderError("isolated failure")
        if mode == "tool":
            if any(message.role == "tool" for message in request.messages):
                response = TraceResponse(content="sunny")
            else:
                response = TraceResponse(
                    content="I will check.",
                    tool_calls=[ToolCall(name="weather", arguments={"city": "Paris"})],
                )
        elif mode == "thrash":
            response = TraceResponse(
                content="checking again",
                tool_calls=[ToolCall(name="weather", arguments={"city": "Paris"})],
            )
        elif mode == "budget":
            response = TraceResponse(
                content="checking",
                tool_calls=[ToolCall(name="weather", arguments={"city": "Paris"})],
            )
        else:
            answer = request.case_input.get("answer", "ok")
            response = TraceResponse(content=str(answer))
        return WireResult(
            response=response,
            usage=TokenUsage(input_tokens=5, output_tokens=2),
            raw={},
        )


def simple_task(*cases: TaskCase, step_budget: int = 1) -> TaskDefinition:
    return TaskDefinition(
        id="simple",
        prompt_template="{prompt}",
        cases=list(cases),
        graders=[ExactMatchSpec(type="exact_match")],
        step_budget=step_budget,
    )


@pytest.mark.asyncio
async def test_run_persists_cost_usage_seed_and_resumes(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    task = simple_task(
        TaskCase(id="one", input={"prompt": "hello", "answer": "ok"}, expected={"answer": "ok"})
    )
    runner = Runner({"test-model": provider})
    config = RunConfig(output_dir=tmp_path / "run", seed=17)

    first = await runner.run([task], ["test-model"], config)
    second = await runner.run([task], ["test-model"], config)

    assert len(first) == len(second) == 1
    assert provider.calls == 1
    assert first[0].passed
    assert first[0].request.seed is not None
    assert first[0].usage.total_tokens == 7
    assert first[0].cost_usd > 0
    assert first[0].model_calls[0].purpose == "agent"


@pytest.mark.asyncio
async def test_force_reruns_completed_case(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    task = simple_task(
        TaskCase(id="one", input={"prompt": "hello"}, expected={"answer": "ok"})
    )
    runner = Runner({"test-model": provider})
    output = tmp_path / "run"

    await runner.run([task], ["test-model"], RunConfig(output_dir=output))
    traces = await runner.run([task], ["test-model"], RunConfig(output_dir=output, force=True))

    assert provider.calls == 2
    assert len(traces) == 2


@pytest.mark.asyncio
async def test_partial_case_failure_does_not_abort_run(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    task = simple_task(
        TaskCase(id="good", input={"prompt": "good"}, expected={"answer": "ok"}),
        TaskCase(
            id="bad",
            input={"prompt": "bad", "mode": "error"},
            expected={"answer": "ok"},
        ),
    )

    traces = await Runner({"test-model": provider}).run(
        [task], ["test-model"], RunConfig(output_dir=tmp_path / "run")
    )

    assert len(traces) == 2
    assert {trace.status for trace in traces} == {"completed", "failed"}
    failed = next(trace for trace in traces if trace.status == "failed")
    assert failed.failure is not None
    assert failed.failure.kind == "provider_error"


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_and_grades_trajectory(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    case = TaskCase(
        id="weather",
        input={
            "prompt": "weather?",
            "mode": "tool",
            "tool_results": {
                "weather": [
                    {"arguments": {"city": "Paris"}, "result": {"condition": "sunny"}}
                ]
            },
        },
        expected={
            "tool_calls": [{"name": "weather", "arguments": {"city": "Paris"}}]
        },
    )
    task = TaskDefinition(
        id="tools",
        prompt_template="{prompt}",
        cases=[case],
        graders=[ToolCallSpec(type="tool_call")],
        step_budget=3,
    )

    trace = (
        await Runner({"test-model": provider}).run(
            [task], ["test-model"], RunConfig(output_dir=tmp_path / "run")
        )
    )[0]

    assert trace.status == "completed"
    assert trace.grader_outputs[0].score == 1.0
    assert trace.response is not None and trace.response.content == "sunny"
    assert trace.trajectory[0].tool_result == {"condition": "sunny"}
    assert len(trace.model_calls) == 2


@pytest.mark.asyncio
async def test_thrash_and_budget_exhaustion_are_distinct(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    fixtures = {
        "weather": [{"arguments": {"city": "Paris"}, "result": {"condition": "sunny"}}]
    }
    task = simple_task(
        TaskCase(
            id="thrash",
            input={"prompt": "x", "mode": "thrash", "tool_results": fixtures},
            expected={"answer": "sunny"},
        ),
        step_budget=3,
    )
    budget_task = simple_task(
        TaskCase(
            id="budget",
            input={"prompt": "x", "mode": "budget", "tool_results": fixtures},
            expected={"answer": "sunny"},
        ),
        step_budget=1,
    ).model_copy(update={"id": "budget-task"})

    traces = await Runner({"test-model": provider}).run(
        [task, budget_task], ["test-model"], RunConfig(output_dir=tmp_path / "run")
    )
    failure_kinds = {trace.failure.kind for trace in traces if trace.failure is not None}

    assert failure_kinds == {"thrash", "budget_exhausted"}
    thrash = next(trace for trace in traces if trace.failure and trace.failure.kind == "thrash")
    assert len(thrash.trajectory) == 2


@pytest.mark.asyncio
async def test_provider_concurrency_cap_is_respected(tmp_path: Path) -> None:
    provider = ScriptedProvider(max_concurrency=2)
    cases = [
        TaskCase(id=f"case-{index}", input={"prompt": str(index)}, expected={"answer": "ok"})
        for index in range(8)
    ]

    await Runner({"test-model": provider}).run(
        [simple_task(*cases)], ["test-model"], RunConfig(output_dir=tmp_path / "run")
    )

    assert provider.max_active == 2

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchtrace.cli import main
from benchtrace.cost import analyze_costs, format_cost_report
from benchtrace.report import render_report, write_report
from benchtrace.retrieval import evaluate_retrieval, format_retrieval_report
from benchtrace.trace import (
    Citation,
    Failure,
    GraderOutput,
    Message,
    ModelCall,
    RetrievedContext,
    TokenUsage,
    ToolCall,
    TraceRecord,
    TraceRequest,
    TraceResponse,
    TrajectoryStep,
    append_trace,
)


def analysis_trace(
    *,
    case_id: str = "case",
    passed: bool = True,
    input_tokens: int = 1_000_000,
    output_tokens: int = 500_000,
) -> TraceRecord:
    request = TraceRequest(
        messages=[Message(role="user", content="question")],
        relevant_context_ids=["a", "c"],
    )
    response = TraceResponse(content="answer")
    call = ModelCall(
        provider="test",
        model="test-model",
        purpose="agent",
        request=request,
        response=response,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    return TraceRecord(
        run_id="run",
        task_id="retrieval-task",
        case_id=case_id,
        provider="test",
        model="test-model",
        started_at=datetime.now(UTC),
        request=request,
        response=response,
        usage=call.usage,
        model_calls=[call],
        retrieved_context=[
            RetrievedContext(id="b", text="irrelevant", relevant=False),
            RetrievedContext(id="a", text="supporting text", relevant=True),
            RetrievedContext(id="c", text="more support", relevant=True),
        ],
        citations=[
            Citation(claim="supported claim", context_ids=["a"], supported=True, reason="entailed"),
            Citation(claim="missing claim", context_ids=["missing"], supported=True),
        ],
        trajectory=[
            TrajectoryStep(
                step=0,
                observation=None,
                action=ToolCall(name="search", arguments={"q": "x"}),
                tool_result={"hits": ["a"]},
            )
        ],
        grader_outputs=[
            GraderOutput(
                grader="exact",
                score=1.0 if passed else 0.0,
                reason="correct" if passed else "wrong",
            )
        ],
        status="completed" if passed else "failed",
        failure=None if passed else Failure(kind="wrong_answer", message="wrong"),
    )


def test_cost_analysis_reprices_calls_and_highlights_cost_per_success() -> None:
    passing = analysis_trace()
    failing = analysis_trace(
        case_id="failed", passed=False, input_tokens=1_000_000, output_tokens=0
    )

    report = analyze_costs([passing, failing])
    rendered = format_cost_report(report)

    assert report.total.cost_usd == pytest.approx(3.0)
    assert report.total.n_calls == 2
    assert report.total.n_results == 2
    assert report.total.n_successful == 1
    assert report.total.cost_per_success_usd == pytest.approx(3.0)
    assert report.by_model[0].name == "test-model"
    assert rendered.startswith("COST PER SUCCESSFUL RESULT: $3.000000")
    assert "n_success=1" in rendered


def test_retrieval_metrics_and_claim_level_faithfulness() -> None:
    report = evaluate_retrieval([analysis_trace()], k=2)

    assert report.precision_at_k.value == 0.5
    assert report.precision_at_k.n == 1
    assert report.recall_at_k.value == 0.5
    assert report.mrr.value == 0.5
    assert report.citation_faithfulness.value == 0.5
    assert report.citation_faithfulness.n == 2
    assert not report.citations[1].supported
    assert "missing" in report.citations[1].reason
    assert "precision@2: 0.5000 (n=1)" in format_retrieval_report(report)


def test_static_html_contains_metrics_and_clickable_full_trajectory() -> None:
    rendered = render_report([analysis_trace(passed=False)], title="Evaluation <report>")

    assert rendered.startswith("<!doctype html>")
    assert "Evaluation &lt;report&gt;" in rendered
    assert "Cost per successful result" in rendered
    assert "Pass rate by model and task" in rendered
    assert "Failure browser" in rendered
    assert '<details class="failure">' in rendered
    assert "trajectory" in rendered
    assert "search" in rendered
    assert "https://" not in rendered
    assert "<script src=" not in rendered


def test_analysis_cli_commands_read_only_runner_traces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    append_trace(run_dir / "traces.jsonl", analysis_trace())

    assert main(["cost", str(run_dir)]) == 0
    assert "COST PER SUCCESSFUL RESULT" in capsys.readouterr().out

    assert main(["retrieval", str(run_dir), "--k", "2"]) == 0
    assert "precision@2" in capsys.readouterr().out

    output = tmp_path / "custom.html"
    assert main(["report", str(run_dir), "--output", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(output)
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_write_report_defaults_inside_run_directory(tmp_path: Path) -> None:
    append_trace(tmp_path / "traces.jsonl", analysis_trace())

    destination = write_report(tmp_path)

    assert destination == tmp_path / "report.html"
    assert destination.is_file()

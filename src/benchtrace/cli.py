"""Command-line interface for running and analyzing benchtrace traces."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from benchtrace.cost import analyze_costs, format_cost_report
from benchtrace.providers import Provider, provider_for_model
from benchtrace.providers.http import HTTPProvider
from benchtrace.report import write_report
from benchtrace.retrieval import evaluate_retrieval, format_retrieval_report
from benchtrace.runner import RunConfig, Runner
from benchtrace.stats import compare_trace_sets, format_estimate
from benchtrace.task import LLMJudgeSpec, load_tasks
from benchtrace.trace import load_traces


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(
        prog="benchtrace", description="Trace-first evaluation for LLM agents"
    )
    parser.add_argument("--version", action="version", version="benchtrace 0.1.1")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a YAML benchmark suite")
    run.add_argument("tasks", type=Path, help="task YAML file or directory")
    run.add_argument("--models", required=True, help="comma-separated model names")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path)
    run.add_argument("--force", action="store_true")

    cost = commands.add_parser("cost", help="analyze cost from run traces")
    cost.add_argument("run_dir", type=Path)
    cost.add_argument("--pricing-version", default="v1")

    retrieval = commands.add_parser("retrieval", help="evaluate retrieval from run traces")
    retrieval.add_argument("run_dir", type=Path)
    retrieval.add_argument("--k", type=int, default=5)

    report = commands.add_parser("report", help="render a self-contained HTML report")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--output", type=Path)

    compare = commands.add_parser("compare", help="compare two models with McNemar's test")
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)
    compare.add_argument("--model-a")
    compare.add_argument("--model-b")
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--seed", type=int, default=0)
    return parser


def _default_output() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("runs") / timestamp


def _provider_group(model: str) -> str:
    if model.startswith(("claude-", "anthropic/")):
        return "anthropic"
    if model.startswith(("gemini-", "google/")):
        return "google"
    for prefix in ("together", "groq", "openrouter"):
        if model.startswith(f"{prefix}/"):
            return prefix
    return "openai"


def _providers_for(models: Sequence[str], concurrency: int) -> dict[str, Provider]:
    groups: dict[str, Provider] = {}
    result: dict[str, Provider] = {}
    for model in models:
        group = _provider_group(model)
        provider = groups.get(group)
        if provider is None:
            provider = provider_for_model(model, max_concurrency=concurrency)
            groups[group] = provider
        result[model] = provider
    return result


def _update_latest(output: Path) -> None:
    latest = Path("runs") / "latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    if latest.is_symlink() or latest.is_file():
        latest.unlink()
    elif latest.exists():
        return
    latest.symlink_to(output.resolve(), target_is_directory=True)


async def _run_command(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    if not models:
        raise ValueError("--models must contain at least one model")
    judge_models = sorted(
        {
            spec.model
            for task in tasks
            for spec in task.graders
            if isinstance(spec, LLMJudgeSpec)
        }
    )
    all_providers = _providers_for([*models, *judge_models], int(args.concurrency))
    output = args.output or _default_output()
    runner = Runner(
        {model: all_providers[model] for model in models},
        judge_providers={model: all_providers[model] for model in judge_models},
    )
    try:
        traces = await runner.run(
            tasks,
            models,
            RunConfig(
                output_dir=output,
                repeats=int(args.repeats),
                seed=int(args.seed),
                force=bool(args.force),
            ),
        )
    finally:
        for provider in set(all_providers.values()):
            if isinstance(provider, HTTPProvider):
                await provider.aclose()
    _update_latest(output)
    passed = sum(trace.passed for trace in traces)
    print(f"Run: {output} · passed {passed}/{len(traces)} (n={len(traces)})")
    return 0


def _cost_command(args: argparse.Namespace) -> int:
    traces = load_traces(args.run_dir)
    print(format_cost_report(analyze_costs(traces, pricing_version=str(args.pricing_version))))
    return 0


def _retrieval_command(args: argparse.Namespace) -> int:
    traces = load_traces(args.run_dir)
    print(format_retrieval_report(evaluate_retrieval(traces, k=int(args.k))))
    return 0


def _report_command(args: argparse.Namespace) -> int:
    destination = write_report(args.run_dir, output=args.output)
    print(destination)
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    comparison = compare_trace_sets(
        load_traces(args.run_a),
        load_traces(args.run_b),
        model_a=args.model_a,
        model_b=args.model_b,
        confidence=float(args.confidence),
        seed=int(args.seed),
    )
    print(f"{comparison.model_a}: {format_estimate(comparison.pass_rate_a)}")
    print(f"{comparison.model_b}: {format_estimate(comparison.pass_rate_b)}")
    print(
        f"Paired difference: {format_estimate(comparison.difference)}; "
        f"McNemar statistic={comparison.mcnemar_statistic:.4g}, "
        f"p={comparison.p_value:.4g} (n={comparison.n})"
    )
    print(
        f"Discordant: {comparison.model_a}-only={comparison.a_only_passed}, "
        f"{comparison.model_b}-only={comparison.b_only_passed} (n={comparison.n})"
    )
    print(comparison.conclusion + f" (n={comparison.n})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run_command(args))
        if args.command == "cost":
            return _cost_command(args)
        if args.command == "retrieval":
            return _retrieval_command(args)
        if args.command == "report":
            return _report_command(args)
        if args.command == "compare":
            return _compare_command(args)
    except (OSError, ValueError) as error:
        print(f"benchtrace: error: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

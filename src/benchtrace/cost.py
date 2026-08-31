"""Cost analysis over canonical traces and versioned pricing snapshots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from benchtrace.providers.pricing import (
    DEFAULT_PRICING_VERSION,
    PricingTable,
    calculate_cost,
    load_pricing_table,
)
from benchtrace.trace import ModelCall, TraceRecord


@dataclass(frozen=True, slots=True)
class CallCost:
    """Repriced cost of one provider call retained in a trace."""

    provider: str
    model: str
    task_id: str
    case_id: str
    purpose: str
    cost_usd: float
    priced: bool


@dataclass(frozen=True, slots=True)
class CostSummary:
    """Cost totals with result, call, and success sample sizes."""

    name: str
    cost_usd: float
    n_results: int
    n_calls: int
    n_successful: int
    cost_per_success_usd: float | None


@dataclass(frozen=True, slots=True)
class CostReport:
    """Complete run cost analysis."""

    pricing_version: str
    pricing_effective_date: str
    total: CostSummary
    by_model: tuple[CostSummary, ...]
    by_task: tuple[CostSummary, ...]
    by_provider: tuple[CostSummary, ...]
    calls: tuple[CallCost, ...]
    unpriced_models: tuple[str, ...]


@dataclass(slots=True)
class _Accumulator:
    cost: float = 0.0
    calls: int = 0
    results: int = 0
    successful: int = 0

    def summary(self, name: str) -> CostSummary:
        per_success = self.cost / self.successful if self.successful else None
        return CostSummary(
            name=name,
            cost_usd=self.cost,
            n_results=self.results,
            n_calls=self.calls,
            n_successful=self.successful,
            cost_per_success_usd=per_success,
        )


def _trace_calls(trace: TraceRecord) -> list[ModelCall]:
    if trace.model_calls:
        return list(trace.model_calls)
    if trace.response is None:
        return []
    return [
        ModelCall(
            provider=trace.provider,
            model=trace.model,
            purpose="agent",
            request=trace.request,
            response=trace.response,
            usage=trace.usage,
            latency_ms=trace.latency_ms,
            cost_usd=trace.cost_usd,
            pricing_known=True,
        )
    ]


def analyze_costs(
    traces: Sequence[TraceRecord],
    *,
    table: PricingTable | None = None,
    pricing_version: str = DEFAULT_PRICING_VERSION,
) -> CostReport:
    """Reprice every retained model call and aggregate by model, task, and provider."""

    selected = table or load_pricing_table(pricing_version)
    total = _Accumulator()
    models: defaultdict[str, _Accumulator] = defaultdict(_Accumulator)
    tasks: defaultdict[str, _Accumulator] = defaultdict(_Accumulator)
    providers: defaultdict[str, _Accumulator] = defaultdict(_Accumulator)
    call_costs: list[CallCost] = []
    unpriced: set[str] = set()
    for trace in traces:
        successful = trace.passed
        total.results += 1
        total.successful += int(successful)
        models[trace.model].results += 1
        models[trace.model].successful += int(successful)
        tasks[trace.task_id].results += 1
        tasks[trace.task_id].successful += int(successful)
        trace_calls = _trace_calls(trace)
        for call in trace_calls:
            calculated = calculate_cost(
                call.provider,
                call.model,
                call.usage,
                selected,
                pricing_version=pricing_version,
            )
            call_costs.append(
                CallCost(
                    provider=call.provider,
                    model=call.model,
                    task_id=trace.task_id,
                    case_id=trace.case_id,
                    purpose=call.purpose,
                    cost_usd=calculated.usd,
                    priced=calculated.priced,
                )
            )
            if not calculated.priced:
                unpriced.add(f"{call.provider}/{call.model}")
            total.cost += calculated.usd
            total.calls += 1
            models[trace.model].cost += calculated.usd
            models[trace.model].calls += 1
            tasks[trace.task_id].cost += calculated.usd
            tasks[trace.task_id].calls += 1
            providers[call.provider].cost += calculated.usd
            providers[call.provider].calls += 1
        for call_provider in {call.provider for call in trace_calls}:
            providers[call_provider].results += 1
            providers[call_provider].successful += int(successful)
    return CostReport(
        pricing_version=pricing_version,
        pricing_effective_date=selected.effective_date,
        total=total.summary("all"),
        by_model=tuple(models[key].summary(key) for key in sorted(models)),
        by_task=tuple(tasks[key].summary(key) for key in sorted(tasks)),
        by_provider=tuple(providers[key].summary(key) for key in sorted(providers)),
        calls=tuple(call_costs),
        unpriced_models=tuple(sorted(unpriced)),
    )


def format_cost_report(report: CostReport) -> str:
    """Render a terminal report with cost per successful result first."""

    prominent = (
        f"${report.total.cost_per_success_usd:.6f}"
        if report.total.cost_per_success_usd is not None
        else "N/A (no successful results)"
    )
    lines = [
        f"COST PER SUCCESSFUL RESULT: {prominent} "
        f"(n_success={report.total.n_successful}, n_results={report.total.n_results})",
        f"Total: ${report.total.cost_usd:.6f} across {report.total.n_calls} calls "
        f"(n={report.total.n_results})",
        f"Pricing: {report.pricing_version}, effective {report.pricing_effective_date}",
        "",
        "By model:",
    ]
    lines.extend(_format_summary(item) for item in report.by_model)
    lines.append("\nBy task:")
    lines.extend(_format_summary(item) for item in report.by_task)
    lines.append("\nBy provider:")
    lines.extend(_format_summary(item) for item in report.by_provider)
    if report.unpriced_models:
        lines.append("\nWARNING: no price for " + ", ".join(report.unpriced_models))
    return "\n".join(lines)


def _format_summary(summary: CostSummary) -> str:
    per_success = (
        f"${summary.cost_per_success_usd:.6f}"
        if summary.cost_per_success_usd is not None
        else "N/A"
    )
    return (
        f"  {summary.name}: ${summary.cost_usd:.6f}; per success {per_success}; "
        f"n={summary.n_results}, successes={summary.n_successful}, calls={summary.n_calls}"
    )

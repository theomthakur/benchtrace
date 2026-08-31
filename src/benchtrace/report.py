"""Single-file static HTML reporting over canonical traces."""
# ruff: noqa: E501

from __future__ import annotations

import html
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from benchtrace.cost import CostSummary, analyze_costs
from benchtrace.stats import format_estimate, summarize_pass_rates
from benchtrace.trace import TraceRecord, load_traces

_CSS = """
:root{color-scheme:light dark;--bg:#f6f7f9;--panel:#fff;--text:#18202a;--muted:#667085;
--line:#d8dee7;--accent:#3157d5;--bad:#b42318;--good:#067647}
@media(prefers-color-scheme:dark){:root{--bg:#11151b;--panel:#191f27;--text:#ecf0f5;
--muted:#a8b0bd;--line:#343d49;--accent:#8aa4ff;--bad:#ff8f86;--good:#75d6a0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5
system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;
padding:32px 20px 60px}h1{font-size:30px;margin:0 0 4px}h2{margin:28px 0 10px;font-size:20px}
.subtle{color:var(--muted)}.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
gap:12px;margin:20px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:16px}.value{font-size:24px;font-weight:750;margin-top:4px}.prominent{border:2px solid var(--accent)}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}th,td{
padding:9px 11px;text-align:left;border-bottom:1px solid var(--line)}th{font-size:12px;text-transform:
uppercase;color:var(--muted)}.scroll{overflow-x:auto}.failure{border:1px solid var(--line);border-left:4px solid
var(--bad);border-radius:8px;background:var(--panel);margin:9px 0}.failure summary{cursor:pointer;padding:12px;
font-weight:650}.failure .body{padding:0 12px 12px}.kind{color:var(--bad)}pre{white-space:pre-wrap;
word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;
max-height:520px;overflow:auto}.pass{color:var(--good)}.fail{color:var(--bad)}
"""


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _cost_per_success(summary: CostSummary) -> str:
    if summary.cost_per_success_usd is None:
        return "N/A"
    return f"${summary.cost_per_success_usd:.6f}"


def render_report(traces: Sequence[TraceRecord], *, title: str = "benchtrace report") -> str:
    """Render a self-contained report with no network or build-time dependencies."""

    pass_rates = summarize_pass_rates(traces) if traces else {}
    costs = analyze_costs(traces)
    by_model_cost = {item.name: item for item in costs.by_model}
    latencies: defaultdict[str, list[float]] = defaultdict(list)
    for trace in traces:
        latencies[trace.model].append(trace.latency_ms)
    total_latency = sum(trace.latency_ms for trace in traces)
    mean_latency = total_latency / len(traces) if traces else 0.0
    total_per_success = _cost_per_success(costs.total)
    failures = [trace for trace in traces if not trace.passed]
    pass_rows = "".join(
        "<tr>"
        f"<td>{_e(model)}</td><td>{_e(task)}</td>"
        f"<td>{_e(format_estimate(estimate))}</td>"
        "</tr>"
        for (model, task), estimate in sorted(pass_rates.items())
    )
    model_rows: list[str] = []
    for model in sorted({trace.model for trace in traces}):
        model_cost = by_model_cost[model]
        values = latencies[model]
        model_rows.append(
            "<tr>"
            f"<td>{_e(model)}</td>"
            f"<td>${model_cost.cost_usd:.6f} (n={model_cost.n_results})</td>"
            f"<td>{_e(_cost_per_success(model_cost))} "
            f"(successes={model_cost.n_successful})</td>"
            f"<td>{sum(values) / len(values):.1f} ms (n={len(values)})</td>"
            "</tr>"
        )
    failure_html = "".join(_render_failure(trace) for trace in failures)
    if not failure_html:
        failure_html = '<p class="pass">No failed cases.</p>'
    unpriced = (
        f'<p class="fail">Unpriced models: {_e(", ".join(costs.unpriced_models))}</p>'
        if costs.unpriced_models
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{_e(title)}</title><style>{_CSS}</style></head><body><main>
<header><h1>{_e(title)}</h1><div class="subtle">Static trace report · schema 1.0</div></header>
<section class="hero">
<div class="card prominent"><div class="subtle">Cost per successful result</div>
<div class="value">{_e(total_per_success)}</div><div class="subtle">n={costs.total.n_successful} successful</div></div>
<div class="card"><div class="subtle">Total cost</div><div class="value">${costs.total.cost_usd:.6f}</div>
<div class="subtle">{costs.total.n_calls} calls</div></div>
<div class="card"><div class="subtle">Mean case latency</div><div class="value">{mean_latency:.1f} ms</div>
<div class="subtle">n={len(traces)}</div></div>
<div class="card"><div class="subtle">Failed cases</div><div class="value">{len(failures)}</div>
<div class="subtle">n={len(traces)}</div></div>
</section>{unpriced}
<h2>Pass rate by model and task</h2><div class="scroll"><table><thead><tr><th>Model</th><th>Task</th>
<th>Pass rate with confidence interval</th></tr></thead><tbody>{pass_rows}</tbody></table></div>
<h2>Cost and latency by model</h2><div class="scroll"><table><thead><tr><th>Model</th><th>Cost</th>
<th>Cost per success</th><th>Mean latency</th></tr></thead><tbody>{''.join(model_rows)}</tbody></table></div>
<h2>Failure browser</h2><p class="subtle">Click a case to inspect its request, response, graders, and full trajectory.</p>
{failure_html}
</main></body></html>"""


def _render_failure(trace: TraceRecord) -> str:
    kind = trace.failure.kind if trace.failure else "wrong_answer"
    message = trace.failure.message if trace.failure else "one or more graders did not pass"
    payload = {
        "request": trace.request.model_dump(mode="json"),
        "response": trace.response.model_dump(mode="json") if trace.response else None,
        "grader_outputs": [item.model_dump(mode="json") for item in trace.grader_outputs],
        "trajectory": [step.model_dump(mode="json") for step in trace.trajectory],
        "model_calls": [call.model_dump(mode="json") for call in trace.model_calls],
        "metadata": trace.metadata,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    return (
        '<details class="failure"><summary>'
        f"{_e(trace.model)} · {_e(trace.task_id)}/{_e(trace.case_id)} · "
        f'<span class="kind">{_e(kind)}</span></summary><div class="body">'
        f"<p>{_e(message)}</p><pre>{_e(rendered)}</pre></div></details>"
    )


def write_report(
    run_dir: str | Path, *, output: str | Path | None = None, title: str | None = None
) -> Path:
    """Load traces from a run directory and write one standalone HTML file."""

    source = Path(run_dir)
    traces = load_traces(source)
    destination = Path(output) if output is not None else source / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_report(traces, title=title or f"benchtrace · {source.name}"), encoding="utf-8"
    )
    return destination

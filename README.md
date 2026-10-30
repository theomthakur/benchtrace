# benchtrace

[![CI](https://github.com/theomthakur/benchtrace/actions/workflows/ci.yml/badge.svg)](https://github.com/theomthakur/benchtrace/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/benchtrace)](https://pypi.org/project/benchtrace/)
[![Python](https://img.shields.io/pypi/pyversions/benchtrace)](https://pypi.org/project/benchtrace/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**Evaluate LLM applications and agents from the complete record of what happened.** Benchtrace runs test cases, records model and tool activity in one validated format, grades the results, compares models, tracks cost, and produces a standalone HTML report.

**Package:** [PyPI](https://pypi.org/project/benchtrace/) · **Source:** [GitHub](https://github.com/theomthakur/benchtrace)

```bash
pip install benchtrace
```

```bash
export OPENAI_API_KEY="..." ANTHROPIC_API_KEY="..."

benchtrace run tasks/ --models gpt-5,claude-opus-5 --repeats 1
benchtrace cost runs/latest         # what it cost, per model and per successful result
benchtrace retrieval runs/latest    # precision@k, recall@k, citation faithfulness
benchtrace report runs/latest       # one self-contained HTML file
benchtrace compare runs/a runs/b    # exact McNemar, paired by case
```

Benchtrace supports both single model calls and multi-step, tool-using agents. It stores every run as JSONL, so results can be inspected, archived, compared, or processed without running a service or database.

## Why the trace is the architecture

`src/benchtrace/trace.py` is the only run-result contract. `TraceRecord` is versioned, immutable, rejects unknown fields, and captures the rendered request, final response, every provider call, usage, latency, cost, retrieved context, citations, trajectory steps, grader outputs, failure mode, and metadata. JSONL readers report invalid records as `path:line: error`.

A run directory contains `traces.jsonl`. There is no database and no service. You can stream, archive, diff, or process it with the library:

```python
from benchtrace.trace import load_traces
from benchtrace.stats import summarize_pass_rates

traces = load_traces("runs/latest")
for (model, task), estimate in summarize_pass_rates(traces).items():
    print(model, task, estimate.value, estimate.lower, estimate.upper, estimate.n)
```

## Tasks are YAML, not classes

Tasks contain a prompt template, cases, expected behavior, tools, a step budget, and graders. Loader errors identify the offending file and line. This is a minimal task:

```yaml
schema_version: "1.0"
id: weather
prompt_template: "Use weather for {city}."
step_budget: 3
tools:
  - name: weather
    description: Look up current conditions.
    parameters: {type: object, properties: {city: {type: string}}, required: [city]}
graders:
  - type: tool_call
cases:
  - id: paris
    input:
      city: Paris
      tool_results:
        weather: [{arguments: {city: Paris}, result: {condition: sunny}}]
    expected:
      tool_calls: [{name: weather, arguments: {city: Paris}}]
```

`tool_results` makes committed suites runnable without application code. Library users can instead pass async implementations through `ToolRegistry`. The committed `tasks/` suite has 30 cases across weather, arithmetic, and inventory tools.

## Providers

All adapters implement one async `Provider` interface. The built-in REST adapters avoid mandatory vendor SDKs:

| Model prefix | Adapter | Environment variable |
|---|---|---|
| `gpt-*` (default) | OpenAI | `OPENAI_API_KEY` |
| `claude-*`, `anthropic/` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini-*`, `google/` | Google | `GOOGLE_API_KEY` |
| `together/` | OpenAI-compatible | `TOGETHER_API_KEY` |
| `groq/` | OpenAI-compatible | `GROQ_API_KEY` |
| `openrouter/` | OpenAI-compatible | `OPENROUTER_API_KEY` |

Providers enforce configurable concurrency, retry transient failures with exponential backoff and jitter, and honor rate-limit reset headers. Seeds are passed only by adapters that support them. Tests use `httpx.MockTransport` and never make network requests.

## Agent loop and failure modes

The runner performs observe/plan/act transitions until the model returns a final answer or reaches its step budget. Every action and observation is retained. Repeating the same action when observed state has not changed is `thrash`; exhausting the step budget is `budget_exhausted`. Both are distinct from `wrong_answer`, `tool_error`, `provider_error`, and `grader_error`. One failed case is persisted and never aborts other cases.

Runs are resumable. Existing `(task, case, model, repeat)` keys are skipped unless `--force` is supplied. The default output is timestamped under `runs/`, and `runs/latest` points to it.

## Graders

Every grader returns a score in `[0, 1]` and a reason:

- exact match, with case and whitespace controls;
- regular expression, with explicit flags;
- numeric absolute and relative tolerance;
- JSON Schema Draft 2020-12;
- ordered tool-call name and argument correctness;
- rubric-based LLM judge.

The LLM judge has its own provider and model selection. It refuses to judge the same model under test unless `force_same_model: true` is present in task data. Judge usage and cost are retained as ordinary `ModelCall` entries.

## Statistics and comparisons

Pass-rate intervals default to the **Wilson score interval**, with exact **Clopper-Pearson** available via `method="clopper-pearson"`. A percentile bootstrap is deliberately not used: for Bernoulli data its distribution is exactly `Binomial(n, p_hat)`, so it collapses to a single point whenever every case passes or every case fails. `5/5` would report a 95% interval of 100% to 100%, and all-pass is an ordinary benchmark outcome. See Brown, Cai and DasGupta (2001), *Interval Estimation for a Binomial Proportion*.

Paired **differences** between two models are still bootstrapped, which is appropriate there, since the per-case difference is not a boundary-constrained proportion.

Two models are paired by task, case, and repeat and compared with an exact two-sided McNemar test, not independent means. Mismatched case sets are rejected. Output includes `n`; if the paired-difference confidence interval includes zero, the CLI explicitly reports no winner.

```bash
benchtrace compare runs/a runs/b
```

## Cost, retrieval, and reports

Pricing is loaded from versioned package data (`src/benchtrace/data/pricing-v1.json`), never
embedded in clients. `benchtrace cost` reprices every call and reports provider, model, and task
totals. The trace keeps uncached input, cache reads, five-minute cache writes, and one-hour cache
writes separate so provider-specific rates and long-context tiers are not flattened into an
estimate. Cost per successful result is printed first. Unknown model prices are reported rather
than silently estimated.

Retrieval metrics operate only on runner traces. `relevant_context_ids` supports precision@k, recall@k, and MRR. Citation faithfulness requires a recorded semantic support verdict; missing context or an unassessed claim is conservatively unsupported rather than guessed from token overlap.

`benchtrace report` writes one escaped, self-contained `report.html` with embedded CSS, no server, build step, JavaScript dependency, or CDN. It includes pass-rate intervals, model cost and latency, and an expandable browser containing every failed trajectory.

## CLI reference

```text
benchtrace run TASKS --models MODEL[,MODEL] [--repeats N] [--concurrency N] [--seed N] [--force]
benchtrace cost RUN_DIR [--pricing-version v1]
benchtrace retrieval RUN_DIR [--k 5]
benchtrace report RUN_DIR [--output report.html]
benchtrace compare RUN_A RUN_B [--model-a MODEL] [--model-b MODEL]
```

## Development

Python 3.11+ is supported. Dependencies are pinned for reproducible CI.

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy --strict src/benchtrace
pytest
python -m build
```

Provider calls in tests are mocked and no API key is required. Please extend `TraceRecord` when a feature needs new recorded data; do not introduce a parallel result loader.

## Pricing and releases

Pricing snapshots are data with an effective date. They are not a promise of current vendor pricing; update and version the table when providers change rates. See [CHANGELOG.md](CHANGELOG.md) for release history.

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).


---

Sibling project: [agentcassette](https://github.com/theomthakur/agentcassette), deterministic record and replay for agent tests.

Built by Om Thakur ·
[portfolio](https://theomthakur.github.io/portfolio) ·
[github](https://github.com/theomthakur) ·
[linkedin](https://linkedin.com/in/theomthakur)

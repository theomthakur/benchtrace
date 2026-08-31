# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog, and releases follow Semantic Versioning.

## [0.1.0] - 2026-08-31

### Fixed

- **Pass-rate confidence intervals collapsed at the boundary.** The percentile bootstrap
  reported `5/5` as a 95% interval of exactly 100% to 100%, and `0/8` as 0% to 0%, because
  the bootstrap distribution of a Bernoulli mean is degenerate when the observed rate is 0
  or 1. All-pass and all-fail are ordinary benchmark outcomes, so this was reachable on a
  normal run. `bootstrap_pass_rate` is replaced by `pass_rate_interval`, defaulting to the
  Wilson score interval with exact Clopper-Pearson available via `method=`. The intervals
  were also anti-conservative away from the boundary: `19/20` gave [0.850, 1.000] where the
  exact interval is [0.751, 0.999]. Added boundary tests at `0/n` and `n/n`, checks against
  independent reference implementations, and an empirical coverage simulation.


### Added

- Versioned, runtime-validated `TraceRecord` JSONL contract and resumable persistence.
- Async OpenAI, Anthropic, Google, Together, Groq, and OpenRouter-compatible providers.
- Header-aware rate limiting, retry jitter, concurrency caps, deterministic seeds, token usage, and versioned cost accounting.
- Strict YAML task definitions with file-and-line validation.
- Exact, regex, numeric, JSON Schema, tool-call, and independent-provider LLM judge graders.
- Multi-step tool loop with full trajectories and distinct thrash and budget-exhaustion failures.
- Exact percentile-bootstrap pass-rate intervals and paired McNemar model comparisons.
- Trace-only cost, retrieval, citation-faithfulness, and standalone HTML reporting commands.
- Thirty fixture-backed tool-use benchmark cases and offline provider mocks in the test suite.

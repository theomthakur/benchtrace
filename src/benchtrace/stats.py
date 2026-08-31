"""Statistically sound summaries and paired model comparisons."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from scipy.stats import beta, binomtest, norm

from benchtrace.trace import TraceRecord


@dataclass(frozen=True, slots=True)
class Estimate:
    """An estimate with confidence bounds and its sample size."""

    value: float
    lower: float
    upper: float
    confidence: float
    n: int


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """McNemar comparison and paired-bootstrap effect estimate."""

    model_a: str
    model_b: str
    pass_rate_a: Estimate
    pass_rate_b: Estimate
    difference: Estimate
    a_only_passed: int
    b_only_passed: int
    mcnemar_statistic: float
    p_value: float
    n: int
    winner: str | None
    conclusion: str


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")


IntervalMethod = Literal["wilson", "clopper-pearson"]


def _wilson_interval(passed: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval. Never degenerate, good coverage at small n."""

    z = cast(float, norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    rate = passed / n
    denominator = 1.0 + z * z / n
    centre = (rate + z * z / (2.0 * n)) / denominator
    half_width = (
        z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denominator
    )
    # Snap the boundaries exactly. Without this, 0/50 reports a lower bound of
    # 6.9e-18 rather than 0, which is noise in every printed report.
    lower = 0.0 if passed == 0 else max(0.0, centre - half_width)
    upper = 1.0 if passed == n else min(1.0, centre + half_width)
    return lower, upper


def _clopper_pearson_interval(
    passed: int, n: int, confidence: float
) -> tuple[float, float]:
    """Exact Clopper-Pearson interval. Guaranteed at least nominal coverage."""

    alpha = 1.0 - confidence
    lower = 0.0 if passed == 0 else cast(float, beta.ppf(alpha / 2.0, passed, n - passed + 1))
    upper = (
        1.0
        if passed == n
        else cast(float, beta.ppf(1.0 - alpha / 2.0, passed + 1, n - passed))
    )
    return lower, upper


def pass_rate_interval(
    outcomes: Sequence[bool],
    *,
    confidence: float = 0.95,
    method: IntervalMethod = "wilson",
) -> Estimate:
    """Confidence interval for a Bernoulli pass rate.

    A percentile bootstrap is deliberately NOT used here. For Bernoulli data the
    bootstrap distribution of the mean is exactly ``Binomial(n, p_hat)``, so when every
    case passes or every case fails that distribution is degenerate and the interval
    collapses to a single point: ``5/5`` would report a 95% interval of 100% to 100%.
    All-pass and all-fail are ordinary outcomes in a benchmark, so the interval has to
    stay honest there.

    ``wilson`` (default) is the score interval: never degenerate, and better average
    coverage than the normal approximation at the sample sizes benchmarks actually use.
    ``clopper-pearson`` is exact and guarantees at least nominal coverage, at the cost of
    being conservative. See Brown, Cai and DasGupta (2001), "Interval Estimation for a
    Binomial Proportion", Statistical Science 16(2).
    """

    _validate_confidence(confidence)
    n = len(outcomes)
    if n == 0:
        raise ValueError("at least one outcome is required")
    passed = sum(outcomes)
    if method == "wilson":
        lower, upper = _wilson_interval(passed, n, confidence)
    elif method == "clopper-pearson":
        lower, upper = _clopper_pearson_interval(passed, n, confidence)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown interval method {method!r}")
    return Estimate(
        value=passed / n, lower=lower, upper=upper, confidence=confidence, n=n
    )


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a quantile of no values")
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def bootstrap_mean(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 20_000,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap for a mean, used for paired per-case differences."""

    _validate_confidence(confidence)
    n = len(values)
    if n == 0:
        raise ValueError("at least one value is required")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    generator = random.Random(seed)
    distribution = sorted(
        sum(generator.choices(values, k=n)) / n for _ in range(resamples)
    )
    alpha = 1.0 - confidence
    return Estimate(
        value=sum(values) / n,
        lower=_quantile(distribution, alpha / 2.0),
        upper=_quantile(distribution, 1.0 - alpha / 2.0),
        confidence=confidence,
        n=n,
    )


def compare_paired(
    outcomes_a: Sequence[bool],
    outcomes_b: Sequence[bool],
    *,
    model_a: str = "A",
    model_b: str = "B",
    confidence: float = 0.95,
    resamples: int = 20_000,
    seed: int = 0,
    method: IntervalMethod = "wilson",
) -> PairedComparison:
    """Compare paired binary outcomes using exact McNemar and a paired bootstrap."""

    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("paired comparisons require equal-length outcomes")
    if not outcomes_a:
        raise ValueError("paired comparisons require at least one case")
    a_only = sum(a and not b for a, b in zip(outcomes_a, outcomes_b, strict=True))
    b_only = sum(b and not a for a, b in zip(outcomes_a, outcomes_b, strict=True))
    discordant = a_only + b_only
    if discordant:
        p_value = float(binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue)
        statistic = (abs(a_only - b_only) - 1.0) ** 2 / discordant
    else:
        p_value = 1.0
        statistic = 0.0
    differences = [
        float(a) - float(b) for a, b in zip(outcomes_a, outcomes_b, strict=True)
    ]
    difference = bootstrap_mean(
        differences, confidence=confidence, resamples=resamples, seed=seed
    )
    alpha = 1.0 - confidence
    includes_zero = difference.lower <= 0.0 <= difference.upper
    winner: str | None = None
    if includes_zero:
        conclusion = (
            f"No winner: the {confidence:.0%} confidence interval for the paired "
            "difference includes zero."
        )
    elif p_value >= alpha:
        conclusion = (
            f"No winner: McNemar's test is not significant at alpha={alpha:.3g} "
            f"(p={p_value:.4g})."
        )
    elif difference.value > 0:
        winner = model_a
        conclusion = f"{model_a} wins over {model_b} on the paired cases."
    else:
        winner = model_b
        conclusion = f"{model_b} wins over {model_a} on the paired cases."
    n = len(outcomes_a)
    return PairedComparison(
        model_a=model_a,
        model_b=model_b,
        pass_rate_a=pass_rate_interval(
            outcomes_a, confidence=confidence, method=method
        ),
        pass_rate_b=pass_rate_interval(
            outcomes_b, confidence=confidence, method=method
        ),
        difference=difference,
        a_only_passed=a_only,
        b_only_passed=b_only,
        mcnemar_statistic=statistic,
        p_value=p_value,
        n=n,
        winner=winner,
        conclusion=conclusion,
    )


def summarize_pass_rates(
    traces: Sequence[TraceRecord],
    *,
    confidence: float = 0.95,
    method: IntervalMethod = "wilson",
) -> dict[tuple[str, str], Estimate]:
    """Summarize pass rate by ``(model, task)`` with n on every estimate."""

    grouped: defaultdict[tuple[str, str], list[bool]] = defaultdict(list)
    for trace in traces:
        grouped[(trace.model, trace.task_id)].append(trace.passed)
    return {
        key: pass_rate_interval(values, confidence=confidence, method=method)
        for key, values in grouped.items()
    }


def compare_trace_sets(
    traces_a: Sequence[TraceRecord],
    traces_b: Sequence[TraceRecord],
    *,
    model_a: str | None = None,
    model_b: str | None = None,
    confidence: float = 0.95,
    resamples: int = 20_000,
    seed: int = 0,
) -> PairedComparison:
    """Pair two models by task, case, and repeat, rejecting mismatched case sets."""

    selected_a = _select_model(traces_a, model_a)
    selected_b = _select_model(traces_b, model_b)
    indexed_a = _index_cases(selected_a)
    indexed_b = _index_cases(selected_b)
    if indexed_a.keys() != indexed_b.keys():
        missing_a = sorted(indexed_b.keys() - indexed_a.keys())
        missing_b = sorted(indexed_a.keys() - indexed_b.keys())
        raise ValueError(
            "runs do not contain identical paired cases; "
            f"missing from A={missing_a[:5]}, missing from B={missing_b[:5]}"
        )
    keys = sorted(indexed_a)
    return compare_paired(
        [indexed_a[key].passed for key in keys],
        [indexed_b[key].passed for key in keys],
        model_a=selected_a[0].model,
        model_b=selected_b[0].model,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def _select_model(
    traces: Sequence[TraceRecord], requested: str | None
) -> list[TraceRecord]:
    models = {trace.model for trace in traces}
    if requested is None:
        if len(models) != 1:
            raise ValueError(f"expected one model per run, found {sorted(models)}")
        requested = next(iter(models))
    selected = [trace for trace in traces if trace.model == requested]
    if not selected:
        raise ValueError(f"model {requested!r} is absent from traces")
    return selected


def _index_cases(
    traces: Sequence[TraceRecord],
) -> Mapping[tuple[str, str, int], TraceRecord]:
    indexed: dict[tuple[str, str, int], TraceRecord] = {}
    for trace in traces:
        indexed[(trace.task_id, trace.case_id, trace.repeat)] = trace
    return indexed


def format_estimate(estimate: Estimate, *, percentage: bool = True) -> str:
    """Format an estimate with its interval and mandatory sample size."""

    scale = 100.0 if percentage else 1.0
    suffix = "%" if percentage else ""
    return (
        f"{estimate.value * scale:.1f}{suffix} "
        f"({estimate.confidence:.0%} CI {estimate.lower * scale:.1f}{suffix} to "
        f"{estimate.upper * scale:.1f}{suffix}, n={estimate.n})"
    )

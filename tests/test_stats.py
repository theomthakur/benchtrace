from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from scipy.stats import beta, binomtest, norm

from benchtrace.stats import (
    compare_paired,
    compare_trace_sets,
    format_estimate,
    pass_rate_interval,
)
from benchtrace.trace import GraderOutput, Message, TraceRecord, TraceRequest, TraceResponse


def result_trace(model: str, case: str, passed: bool) -> TraceRecord:
    score = 1.0 if passed else 0.0
    return TraceRecord(
        run_id=f"run-{model}",
        task_id="task",
        case_id=case,
        provider="test",
        model=model,
        started_at=datetime.now(UTC),
        request=TraceRequest(messages=[Message(role="user", content="q")]),
        response=TraceResponse(content="a"),
        grader_outputs=[GraderOutput(grader="exact", score=score, reason="known")],
        status="completed" if passed else "failed",
        failure=None if passed else {"kind": "wrong_answer", "message": "wrong"},
    )


def wilson_reference(passed: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Independent Wilson implementation, written from the closed form."""

    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    rate = passed / n
    denominator = 1.0 + z**2 / n
    centre = (rate + z**2 / (2 * n)) / denominator
    half = z * ((rate * (1 - rate) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson_reference(
    passed: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Independent Clopper-Pearson implementation from the Beta quantiles."""

    alpha = 1.0 - confidence
    low = 0.0 if passed == 0 else float(beta.ppf(alpha / 2, passed, n - passed + 1))
    high = 1.0 if passed == n else float(beta.ppf(1 - alpha / 2, passed + 1, n - passed))
    return low, high


@pytest.mark.parametrize(
    "passed,n",
    [(6, 10), (1, 20), (19, 20), (50, 100), (3, 7), (0, 8), (5, 5), (20, 20), (1, 1)],
)
def test_wilson_matches_independent_reference(passed: int, n: int) -> None:
    outcomes = [True] * passed + [False] * (n - passed)
    actual = pass_rate_interval(outcomes, confidence=0.95)
    expected_low, expected_high = wilson_reference(passed, n)

    assert actual.value == pytest.approx(passed / n)
    assert actual.lower == pytest.approx(expected_low)
    assert actual.upper == pytest.approx(expected_high)
    assert actual.n == n


@pytest.mark.parametrize(
    "passed,n", [(6, 10), (0, 8), (5, 5), (20, 20), (19, 20), (1, 1)]
)
def test_clopper_pearson_matches_independent_reference(passed: int, n: int) -> None:
    outcomes = [True] * passed + [False] * (n - passed)
    actual = pass_rate_interval(outcomes, confidence=0.95, method="clopper-pearson")
    expected_low, expected_high = clopper_pearson_reference(passed, n)

    assert actual.lower == pytest.approx(expected_low)
    assert actual.upper == pytest.approx(expected_high)


@pytest.mark.parametrize("method", ["wilson", "clopper-pearson"])
@pytest.mark.parametrize("passed,n", [(0, 5), (0, 8), (5, 5), (20, 20), (1, 1)])
def test_all_pass_and_all_fail_never_collapse_to_a_point(
    method: str, passed: int, n: int
) -> None:
    """The regression this module exists for.

    A percentile bootstrap reports 5/5 as a 95% interval of exactly 100% to 100%,
    because the bootstrap distribution is degenerate at the boundary. All-pass and
    all-fail are ordinary benchmark outcomes, so the interval must stay wide.
    """

    outcomes = [True] * passed + [False] * (n - passed)
    actual = pass_rate_interval(outcomes, confidence=0.95, method=method)  # type: ignore[arg-type]

    assert actual.upper > actual.lower, "interval collapsed to a single point"
    if passed == n:
        assert actual.upper == pytest.approx(1.0)
        assert actual.lower < 1.0
    if passed == 0:
        assert actual.lower == pytest.approx(0.0)
        assert actual.upper > 0.0


@pytest.mark.parametrize("passed,n", [(0, 8), (5, 5), (6, 10), (19, 20)])
def test_interval_always_contains_the_point_estimate(passed: int, n: int) -> None:
    outcomes = [True] * passed + [False] * (n - passed)
    for method in ("wilson", "clopper-pearson"):
        actual = pass_rate_interval(outcomes, method=method)  # type: ignore[arg-type]
        assert actual.lower <= actual.value <= actual.upper
        assert 0.0 <= actual.lower <= actual.upper <= 1.0


@pytest.mark.parametrize("true_rate", [0.02, 0.25, 0.5, 0.85, 0.98])
def test_empirical_coverage_holds(true_rate: float) -> None:
    """The property that actually matters: does a 95% interval contain the truth 95% of the time?

    Clopper-Pearson is exact, so its coverage must never fall below nominal. Wilson
    trades a little coverage for much tighter intervals, so it is allowed to dip
    slightly but must stay close. A degenerate interval fails this badly at extreme
    rates, which is exactly where the old percentile bootstrap broke.
    """

    rng = np.random.default_rng(20260831)
    trials, n = 4000, 30
    counts = rng.binomial(n, true_rate, size=trials)

    for method, floor in (("clopper-pearson", 0.95), ("wilson", 0.92)):
        covered = 0
        for passed in counts:
            outcomes = [True] * int(passed) + [False] * (n - int(passed))
            estimate = pass_rate_interval(outcomes, method=method)  # type: ignore[arg-type]
            if estimate.lower <= true_rate <= estimate.upper:
                covered += 1
        coverage = covered / trials
        assert coverage >= floor, f"{method} covered {coverage:.3f} at p={true_rate}"


def test_clopper_pearson_is_wider_on_average_than_wilson() -> None:
    """Clopper-Pearson buys guaranteed coverage with width. Assert that trade exists."""

    for n in (5, 10, 20, 50):
        widths = [
            (
                pass_rate_interval([True] * k + [False] * (n - k), method="clopper-pearson"),
                pass_rate_interval([True] * k + [False] * (n - k), method="wilson"),
            )
            for k in range(n + 1)
        ]
        exact_mean = sum(e.upper - e.lower for e, _ in widths) / len(widths)
        wilson_mean = sum(w.upper - w.lower for _, w in widths) / len(widths)
        assert exact_mean > wilson_mean


def test_format_estimate_reports_n() -> None:
    actual = pass_rate_interval([True] * 6 + [False] * 4)
    assert format_estimate(actual) == "60.0% (95% CI 31.3% to 83.2%, n=10)"


def test_exact_mcnemar_matches_scipy_binomial_known_answer() -> None:
    # Discordant table: A-only=10, B-only=2; concordant outcomes fill n=20.
    outcomes_a = [True] * 10 + [False] * 2 + [True] * 8
    outcomes_b = [False] * 10 + [True] * 2 + [True] * 8

    result = compare_paired(outcomes_a, outcomes_b, model_a="A", model_b="B", seed=4)
    expected_p = binomtest(10, 12, 0.5, alternative="two-sided").pvalue

    assert result.a_only_passed == 10
    assert result.b_only_passed == 2
    assert result.mcnemar_statistic == pytest.approx(49 / 12)
    assert result.p_value == pytest.approx(expected_p)
    assert result.n == 20
    assert result.winner == "A"
    assert result.conclusion == "A wins over B on the paired cases."


def test_comparison_refuses_winner_when_interval_includes_zero() -> None:
    outcomes_a = [True, True, False, False, True, False, True, False]
    outcomes_b = [False, True, True, False, False, True, True, False]

    result = compare_paired(outcomes_a, outcomes_b, seed=9)

    assert result.difference.lower <= 0 <= result.difference.upper
    assert result.winner is None
    assert result.conclusion.startswith("No winner:")
    assert "includes zero" in result.conclusion
    assert result.n == 8


def test_trace_comparison_requires_identical_paired_cases() -> None:
    run_a = [result_trace("A", "1", True), result_trace("A", "2", False)]
    run_b = [result_trace("B", "1", True)]

    with pytest.raises(ValueError, match="identical paired cases"):
        compare_trace_sets(run_a, run_b)


def test_trace_comparison_pairs_by_case_not_input_order() -> None:
    run_a = [result_trace("A", "1", True), result_trace("A", "2", False)]
    run_b = [result_trace("B", "2", True), result_trace("B", "1", False)]

    result = compare_trace_sets(run_a, run_b, seed=1)

    assert result.a_only_passed == 1
    assert result.b_only_passed == 1
    assert result.n == 2

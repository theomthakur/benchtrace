"""Retrieval and citation-faithfulness metrics over canonical traces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from benchtrace.trace import Citation, TraceRecord


@dataclass(frozen=True, slots=True)
class Metric:
    """A metric value with the number of observations behind it."""

    value: float
    n: int


@dataclass(frozen=True, slots=True)
class CitationAssessment:
    """Faithfulness verdict for one cited claim."""

    task_id: str
    case_id: str
    claim: str
    supported: bool
    reason: str


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Aggregate retrieval metrics and claim-level faithfulness evidence."""

    k: int
    precision_at_k: Metric
    recall_at_k: Metric
    mrr: Metric
    citation_faithfulness: Metric
    citations: tuple[CitationAssessment, ...]
    skipped_unlabelled_queries: int


def _citation_assessment(trace: TraceRecord, citation: Citation) -> CitationAssessment:
    contexts = {context.id: context for context in trace.retrieved_context}
    missing = [context_id for context_id in citation.context_ids if context_id not in contexts]
    if missing:
        return CitationAssessment(
            task_id=trace.task_id,
            case_id=trace.case_id,
            claim=citation.claim,
            supported=False,
            reason=f"cited context ids are missing from the trace: {missing}",
        )
    if citation.supported is None:
        return CitationAssessment(
            task_id=trace.task_id,
            case_id=trace.case_id,
            claim=citation.claim,
            supported=False,
            reason=(
                "no support verdict was recorded; benchtrace does not infer semantic "
                "entailment from lexical overlap"
            ),
        )
    return CitationAssessment(
        task_id=trace.task_id,
        case_id=trace.case_id,
        claim=citation.claim,
        supported=citation.supported,
        reason=citation.reason or ("supported" if citation.supported else "unsupported"),
    )


def evaluate_retrieval(
    traces: Sequence[TraceRecord], *, k: int = 5
) -> RetrievalReport:
    """Compute precision@k, recall@k, MRR, and citation faithfulness from traces."""

    if k < 1:
        raise ValueError("k must be at least 1")
    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    skipped = 0
    assessments: list[CitationAssessment] = []
    for trace in traces:
        relevant = set(trace.request.relevant_context_ids)
        if not relevant:
            relevant = {
                context.id for context in trace.retrieved_context if context.relevant is True
            }
        if relevant:
            retrieved = [context.id for context in trace.retrieved_context]
            top_k = retrieved[:k]
            relevant_retrieved = sum(context_id in relevant for context_id in top_k)
            precisions.append(relevant_retrieved / k)
            recalls.append(relevant_retrieved / len(relevant))
            first_rank = next(
                (
                    index
                    for index, context_id in enumerate(retrieved, start=1)
                    if context_id in relevant
                ),
                None,
            )
            reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        else:
            skipped += 1
        assessments.extend(_citation_assessment(trace, citation) for citation in trace.citations)
    supported = sum(assessment.supported for assessment in assessments)
    return RetrievalReport(
        k=k,
        precision_at_k=_mean_metric(precisions),
        recall_at_k=_mean_metric(recalls),
        mrr=_mean_metric(reciprocal_ranks),
        citation_faithfulness=Metric(
            value=supported / len(assessments) if assessments else 0.0,
            n=len(assessments),
        ),
        citations=tuple(assessments),
        skipped_unlabelled_queries=skipped,
    )


def _mean_metric(values: Sequence[float]) -> Metric:
    return Metric(value=sum(values) / len(values) if values else 0.0, n=len(values))


def format_retrieval_report(report: RetrievalReport) -> str:
    """Render retrieval metrics with n alongside every value."""

    return "\n".join(
        [
            f"precision@{report.k}: {report.precision_at_k.value:.4f} "
            f"(n={report.precision_at_k.n})",
            f"recall@{report.k}: {report.recall_at_k.value:.4f} "
            f"(n={report.recall_at_k.n})",
            f"MRR: {report.mrr.value:.4f} (n={report.mrr.n})",
            f"citation faithfulness: {report.citation_faithfulness.value:.4f} "
            f"(n={report.citation_faithfulness.n})",
            f"unlabelled retrieval queries skipped: {report.skipped_unlabelled_queries}",
        ]
    )

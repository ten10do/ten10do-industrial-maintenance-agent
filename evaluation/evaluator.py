"""Per-case evaluation and aggregate metric computation.

A :class:`PlanObservation` is what the harness saw the planner produce. The
evaluator compares it with the case's answer key and emits a :class:`CaseOutcome`,
then aggregates the outcomes into the reported metric set.

Two definitional choices are load-bearing and stated up front, because a metric
whose definition is implicit cannot be argued with:

* **Task success is judged on the plan.** A case succeeds when the intent, the
  tool set and every declared argument are correct, no invalid tool was named and
  planning did not fail. Execution outcomes are reported separately, so a
  planner-only run and an end-to-end run stay directly comparable.
* **Out-of-domain cases are excluded from intent accuracy.** There is no
  maintenance intent to get right, so scoring one would be meaningless. Their
  signal is carried by the unnecessary-tool-call metrics instead.

Every metric's denominator is counted explicitly and written into the report, so
a reader can recompute each ratio from the published numbers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evaluation.dataset import EvalCase
from evaluation.metrics import (
    Metric,
    argument_values_match,
    invalid_tools,
    mean,
    median,
    missing_tools,
    percentile,
    ratio,
    tool_counts,
    unnecessary_tools,
)

#: Failure vocabulary used in the failures report.
FAILURE_PLANNING_ERROR = "planning_error"
FAILURE_INVALID_TOOL = "invalid_tool"
FAILURE_OOD_TOOL_CALL = "unexpected_tool_call_on_ood"
FAILURE_MISSING_TOOL = "missing_tool"
FAILURE_UNNECESSARY_TOOL = "unnecessary_tool"
FAILURE_INTENT_MISMATCH = "intent_mismatch"
FAILURE_ARGUMENT_MISMATCH = "argument_mismatch"
FAILURE_EXECUTION_ERROR = "execution_error"

#: Order used to pick the single headline failure type per case.
FAILURE_PRIORITY: tuple[str, ...] = (
    FAILURE_PLANNING_ERROR,
    FAILURE_INVALID_TOOL,
    FAILURE_OOD_TOOL_CALL,
    FAILURE_MISSING_TOOL,
    FAILURE_UNNECESSARY_TOOL,
    FAILURE_INTENT_MISMATCH,
    FAILURE_ARGUMENT_MISMATCH,
    FAILURE_EXECUTION_ERROR,
)

#: What each reported metric means, published with the results.
METRIC_DEFINITIONS: dict[str, str] = {
    "intent_accuracy": (
        "Cases whose planned intent equals the expected intent, over all "
        "non-out-of-domain cases. A case whose planning failed counts as "
        "incorrect. The denominator is the non-out-of-domain case count, so the "
        "value is null only when the dataset holds no such case."
    ),
    "tool_selection_exact_match": (
        "Cases whose planned tool set equals the expected tool set, over all "
        "cases. An out-of-domain case matches exactly when no tool was planned."
    ),
    "tool_precision": (
        "True-positive tool selections over all selected tools, pooled across "
        "cases (micro average). Null when no tool was selected anywhere, because "
        "precision over an empty selection is undefined rather than perfect."
    ),
    "tool_recall": (
        "True-positive tool selections over all expected tools, pooled across "
        "cases. Null when the dataset expects no tool at all."
    ),
    "argument_accuracy": (
        "Declared-argument checks that passed, over all checks. A check exists "
        "for each case and tool where the case declares expected arguments and "
        "the planner selected that tool. A tool that was expected but not "
        "selected produces no check; tool recall already penalises it. Null when "
        "no check exists."
    ),
    "invalid_tool_rate": (
        "Selected tool names absent from the registry, over all selected tools. "
        "Null when no tool was selected."
    ),
    "unnecessary_tool_call_rate": (
        "Selected tools the case did not expect, over all selected tools. Null "
        "when no tool was selected."
    ),
    "task_success_rate": (
        "Cases whose plan is fully correct: intent right, tool set exactly "
        "right, every declared argument right, no invalid tool, planning did not "
        "fail. Execution outcomes are reported separately."
    ),
    "planner_failure_rate": (
        "Cases where planning raised, over all cases. Null only when the dataset " "is empty."
    ),
    "average_planning_latency_ms": (
        "Mean wall time of the understanding and planning step, over cases where "
        "it was measured. Null when nothing was measured."
    ),
    "execution_error_rate": (
        "Cases where a planned tool reported an error or raised, over cases where "
        "tools were executed. Only present in an end-to-end run."
    ),
}


class PlanObservation(BaseModel):
    """What the harness observed the planner produce for one case."""

    model_config = ConfigDict(extra="forbid")

    intent: str | None = None
    tools: list[str] = Field(default_factory=list)
    arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    planner_used: str | None = None
    planner_fallback: bool = False
    planner_fallback_reason: str | None = None
    planning_error: str | None = Field(
        default=None, description="Planner error code when planning failed."
    )
    planning_latency_ms: float | None = None
    execution_latency_ms: float | None = None
    rag_latency_ms: float | None = None
    total_latency_ms: float | None = None
    execution_errors: list[str] = Field(default_factory=list)


class CaseOutcome(BaseModel):
    """The evaluation of one case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: str
    query: str
    is_ood: bool

    expected_intent: str | None
    actual_intent: str | None
    intent_correct: bool | None = Field(
        default=None, description="None when the case is out of domain."
    )

    expected_tools: list[str]
    actual_tools: list[str]
    tool_exact_match: bool
    tool_true_positive: int
    tool_false_positive: int
    tool_false_negative: int
    missing_tools: list[str]
    unnecessary_tools: list[str]
    invalid_tools: list[str]

    argument_checks_total: int
    argument_checks_passed: int
    argument_correct: bool | None = None

    task_success: bool
    failure_types: list[str] = Field(default_factory=list)

    planner_used: str | None = None
    planner_fallback: bool = False
    planner_fallback_reason: str | None = None
    planning_error: str | None = None

    planning_latency_ms: float | None = None
    execution_latency_ms: float | None = None
    rag_latency_ms: float | None = None
    total_latency_ms: float | None = None

    execution_errors: list[str] = Field(default_factory=list)

    @property
    def primary_failure_type(self) -> str | None:
        """Return the headline failure type, or ``None`` when the case passed."""
        for candidate in FAILURE_PRIORITY:
            if candidate in self.failure_types:
                return candidate
        return None


def evaluate_case(
    case: EvalCase,
    observation: PlanObservation,
    *,
    known_tools: set[str],
) -> CaseOutcome:
    """Compare one observation with its answer key."""
    predicted = list(observation.tools)
    expected = list(case.tools)

    invalid = invalid_tools(predicted, known_tools)
    missing = missing_tools(expected, predicted)
    unnecessary = unnecessary_tools(expected, predicted)
    true_positive, false_positive, false_negative = tool_counts(expected, predicted)
    tool_exact_match = set(predicted) == set(expected)

    checks_total = 0
    checks_passed = 0
    for tool_name, declared in case.arguments.items():
        if tool_name not in set(predicted):
            continue
        checks_total += 1
        if argument_values_match(declared, observation.arguments.get(tool_name, {})):
            checks_passed += 1

    failure_types: list[str] = []

    if observation.planning_error is not None:
        failure_types.append(FAILURE_PLANNING_ERROR)
    if invalid:
        failure_types.append(FAILURE_INVALID_TOOL)

    if case.is_ood:
        # There is no intent to score. The only question is whether the agent
        # stayed silent instead of reaching for an industrial tool.
        intent_correct: bool | None = None
        if predicted:
            failure_types.append(FAILURE_OOD_TOOL_CALL)
        argument_correct: bool | None = None
        task_success = not predicted and not invalid and observation.planning_error is None
    else:
        intent_correct = observation.intent == case.intent and observation.planning_error is None
        if not intent_correct:
            failure_types.append(FAILURE_INTENT_MISMATCH)
        if missing:
            failure_types.append(FAILURE_MISSING_TOOL)
        if unnecessary:
            failure_types.append(FAILURE_UNNECESSARY_TOOL)
        argument_correct = checks_total == 0 or checks_passed == checks_total
        if checks_total and checks_passed != checks_total:
            failure_types.append(FAILURE_ARGUMENT_MISMATCH)
        task_success = (
            intent_correct
            and tool_exact_match
            and not invalid
            and argument_correct
            and observation.planning_error is None
        )

    if observation.execution_errors:
        failure_types.append(FAILURE_EXECUTION_ERROR)

    return CaseOutcome(
        case_id=case.id,
        category=case.category,
        query=case.query,
        is_ood=case.is_ood,
        expected_intent=case.intent,
        actual_intent=observation.intent,
        intent_correct=intent_correct,
        expected_tools=sorted(expected),
        actual_tools=sorted(predicted),
        tool_exact_match=tool_exact_match,
        tool_true_positive=true_positive,
        tool_false_positive=false_positive,
        tool_false_negative=false_negative,
        missing_tools=missing,
        unnecessary_tools=unnecessary,
        invalid_tools=invalid,
        argument_checks_total=checks_total,
        argument_checks_passed=checks_passed,
        argument_correct=argument_correct,
        task_success=task_success,
        failure_types=failure_types,
        planner_used=observation.planner_used,
        planner_fallback=observation.planner_fallback,
        planner_fallback_reason=observation.planner_fallback_reason,
        planning_error=observation.planning_error,
        planning_latency_ms=observation.planning_latency_ms,
        execution_latency_ms=observation.execution_latency_ms,
        rag_latency_ms=observation.rag_latency_ms,
        total_latency_ms=observation.total_latency_ms,
        execution_errors=list(observation.execution_errors),
    )


class MetricSet(BaseModel):
    """The reported metrics. ``None`` means the denominator was zero."""

    model_config = ConfigDict(extra="forbid")

    intent_accuracy: Metric = None
    tool_selection_exact_match: Metric = None
    tool_precision: Metric = None
    tool_recall: Metric = None
    argument_accuracy: Metric = None
    invalid_tool_rate: Metric = None
    unnecessary_tool_call_rate: Metric = None
    task_success_rate: Metric = None
    planner_failure_rate: Metric = None
    average_planning_latency_ms: Metric = None
    execution_error_rate: Metric = None


class CountSet(BaseModel):
    """Every denominator behind :class:`MetricSet`, so each ratio can be rechecked."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int = 0
    in_scope_cases: int = 0
    ood_cases: int = 0
    intent_evaluated_cases: int = 0
    intent_correct_cases: int = 0
    tool_exact_match_cases: int = 0
    expected_tool_selections: int = 0
    selected_tool_selections: int = 0
    tool_true_positive: int = 0
    tool_false_positive: int = 0
    tool_false_negative: int = 0
    invalid_tool_selections: int = 0
    unnecessary_tool_selections: int = 0
    argument_checks_total: int = 0
    argument_checks_passed: int = 0
    task_success_cases: int = 0
    planner_failure_cases: int = 0
    execution_error_cases: int = 0
    executed_cases: int = 0
    planning_latency_samples: int = 0


class LatencySummary(BaseModel):
    """A latency series: how many cases were measured, and its distribution.

    ``p95_ms`` uses linear interpolation between order statistics, the definition
    :func:`evaluation.metrics.percentile` documents. A tail figure without a
    stated convention is not comparable between runs, so the convention is part
    of the number.
    """

    model_config = ConfigDict(extra="forbid")

    sampled_cases: int = 0
    average_ms: Metric = None
    median_ms: Metric = None
    p95_ms: Metric = None
    min_ms: Metric = None
    max_ms: Metric = None


class CategorySummary(BaseModel):
    """Per-category breakdown, so a weakness is not hidden by the total."""

    model_config = ConfigDict(extra="forbid")

    case_count: int = 0
    tool_exact_match_count: int = 0
    task_success_count: int = 0
    task_success_rate: Metric = None


class OODSummary(BaseModel):
    """Out-of-domain safety, reported as its own block."""

    model_config = ConfigDict(extra="forbid")

    case_count: int = 0
    silent_case_count: int = 0
    tool_call_case_count: int = 0
    tool_call_rate: Metric = None
    offending_case_ids: list[str] = Field(default_factory=list)


def aggregate(
    outcomes: list[CaseOutcome],
    *,
    executed: bool,
) -> tuple[MetricSet, CountSet, dict[str, LatencySummary], dict[str, CategorySummary], OODSummary]:
    """Aggregate per-case outcomes into the reported metrics.

    Args:
        outcomes: One entry per case.
        executed: Whether tools were run, which decides if the execution metric
            and latency series are reported at all.
    """
    counts = CountSet(total_cases=len(outcomes))
    counts.ood_cases = sum(1 for item in outcomes if item.is_ood)
    counts.in_scope_cases = counts.total_cases - counts.ood_cases

    for item in outcomes:
        if item.intent_correct is not None:
            counts.intent_evaluated_cases += 1
            counts.intent_correct_cases += int(item.intent_correct)
        counts.tool_exact_match_cases += int(item.tool_exact_match)
        counts.expected_tool_selections += len(item.expected_tools)
        counts.selected_tool_selections += len(item.actual_tools)
        counts.tool_true_positive += item.tool_true_positive
        counts.tool_false_positive += item.tool_false_positive
        counts.tool_false_negative += item.tool_false_negative
        counts.invalid_tool_selections += len(item.invalid_tools)
        counts.unnecessary_tool_selections += len(item.unnecessary_tools)
        counts.argument_checks_total += item.argument_checks_total
        counts.argument_checks_passed += item.argument_checks_passed
        counts.task_success_cases += int(item.task_success)
        counts.planner_failure_cases += int(item.planning_error is not None)
        if item.execution_errors:
            counts.execution_error_cases += 1
        if item.execution_latency_ms is not None:
            counts.executed_cases += 1
        if item.planning_latency_ms is not None:
            counts.planning_latency_samples += 1

    metrics = MetricSet(
        intent_accuracy=ratio(counts.intent_correct_cases, counts.intent_evaluated_cases),
        tool_selection_exact_match=ratio(counts.tool_exact_match_cases, counts.total_cases),
        tool_precision=ratio(
            counts.tool_true_positive,
            counts.tool_true_positive + counts.tool_false_positive,
        ),
        tool_recall=ratio(
            counts.tool_true_positive,
            counts.tool_true_positive + counts.tool_false_negative,
        ),
        argument_accuracy=ratio(counts.argument_checks_passed, counts.argument_checks_total),
        invalid_tool_rate=ratio(counts.invalid_tool_selections, counts.selected_tool_selections),
        unnecessary_tool_call_rate=ratio(
            counts.unnecessary_tool_selections, counts.selected_tool_selections
        ),
        task_success_rate=ratio(counts.task_success_cases, counts.total_cases),
        planner_failure_rate=ratio(counts.planner_failure_cases, counts.total_cases),
        average_planning_latency_ms=mean(
            [item.planning_latency_ms for item in outcomes if item.planning_latency_ms is not None]
        ),
        execution_error_rate=(
            ratio(counts.execution_error_cases, counts.executed_cases) if executed else None
        ),
    )

    latency = _latency_summaries(outcomes, executed=executed)

    categories: dict[str, CategorySummary] = {}
    for item in outcomes:
        summary = categories.setdefault(item.category, CategorySummary())
        summary.case_count += 1
        summary.tool_exact_match_count += int(item.tool_exact_match)
        summary.task_success_count += int(item.task_success)
    for summary in categories.values():
        summary.task_success_rate = ratio(summary.task_success_count, summary.case_count)

    offenders = [item.case_id for item in outcomes if item.is_ood and item.actual_tools]
    ood = OODSummary(
        case_count=counts.ood_cases,
        silent_case_count=counts.ood_cases - len(offenders),
        tool_call_case_count=len(offenders),
        tool_call_rate=ratio(len(offenders), counts.ood_cases),
        offending_case_ids=offenders,
    )

    return metrics, counts, latency, categories, ood


def _latency_summaries(
    outcomes: list[CaseOutcome],
    *,
    executed: bool,
) -> dict[str, LatencySummary]:
    """Build one summary per latency series.

    A series with no samples reports ``None`` rather than a zero. A zero would
    read as "instantaneous", which is a claim, where ``None`` reads as "not
    measured", which is the truth.
    """

    def series(name: str, *, include: bool) -> tuple[str, LatencySummary]:
        if not include:
            return name, LatencySummary()
        values = [
            float(value)
            for value in (getattr(item, name) for item in outcomes)
            if value is not None
        ]
        return name, LatencySummary(
            sampled_cases=len(values),
            average_ms=mean(values),
            median_ms=median(values),
            p95_ms=percentile(values, 95),
            min_ms=min(values) if values else None,
            max_ms=max(values) if values else None,
        )

    return dict(
        [
            series("planning_latency_ms", include=True),
            series("execution_latency_ms", include=executed),
            series("rag_latency_ms", include=executed),
            series("total_latency_ms", include=True),
        ]
    )


__all__ = [
    "FAILURE_PRIORITY",
    "METRIC_DEFINITIONS",
    "CaseOutcome",
    "CategorySummary",
    "CountSet",
    "LatencySummary",
    "MetricSet",
    "OODSummary",
    "PlanObservation",
    "aggregate",
    "evaluate_case",
]

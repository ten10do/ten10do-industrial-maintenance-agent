"""Rule planner versus LLM planner comparison.

The comparison answers one question: on the same frozen dataset, how do the two
planners differ? It reads two baseline reports that were produced by separate real
runs and reports every metric from both, plus ``llm - rule``.

Three rules make the comparison trustworthy, and each one closes a way to flatter
the LLM planner:

* **Nothing is filtered.** Every metric the two baselines carry appears in the
  report, including the ones where the LLM planner is worse. A comparison that
  showed only improvements would be an advertisement.
* **The rule side is read, never re-run.** ``rule_baseline.json`` is an existing
  artefact. Re-running the rule planner to pick a more favourable draw is the
  failure mode this design prevents, and the report records the rule baseline's
  own provenance so a changed input is visible.
* **A comparison is refused, not approximated.** If the LLM baseline is absent or
  is a gate report, no comparison exists. The command reports
  ``LLM_EVALUATION_NOT_RUN`` and exits non-zero instead of publishing nulls under
  a comparison heading.

Usage::

    python -m evaluation.runner --planner llm --dataset evaluation/dataset.json
    python -m evaluation.comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evaluation.evaluator import (
    METRIC_DEFINITIONS,
    LatencySummary,
    MetricSet,
    OODSummary,
)
from evaluation.metrics import Metric
from evaluation.runner import (
    STATUS_OK,
    EvaluationReport,
    _safe_base_url,
    _utc_now,
    _write,
)

SCHEMA_VERSION = "1.0"

#: Report status when no comparison could be produced.
STATUS_LLM_NOT_RUN = "LLM_EVALUATION_NOT_RUN"
#: Report status when the two sides were measured on different datasets.
STATUS_DATASET_MISMATCH = "DATASET_MISMATCH"
#: Report status when a baseline file is missing or unreadable.
STATUS_INPUT_UNREADABLE = "INPUT_UNREADABLE"

#: The program only produces a comparison when both sides are real and comparable.
EXIT_OK = 0
EXIT_UNREADABLE = 1
EXIT_REFUSED = 2

#: Metrics compared, in report order. Taken from the spec, not from the data, so a
#: metric cannot be dropped from the comparison by omitting it from a run.
COMPARED_METRICS: tuple[str, ...] = (
    "intent_accuracy",
    "tool_selection_exact_match",
    "tool_precision",
    "tool_recall",
    "argument_accuracy",
    "invalid_tool_rate",
    "unnecessary_tool_call_rate",
    "task_success_rate",
    "planner_failure_rate",
    "average_planning_latency_ms",
)

#: Metrics where a smaller value is the better outcome. Without this the delta sign
#: would read backwards for the error rates and for latency.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "invalid_tool_rate",
        "unnecessary_tool_call_rate",
        "planner_failure_rate",
        "average_planning_latency_ms",
    }
)

VERDICT_BETTER = "better"
VERDICT_WORSE = "worse"
VERDICT_UNCHANGED = "unchanged"
VERDICT_NOT_COMPARABLE = "not_comparable"

LATENCY_SERIES: tuple[str, ...] = (
    "planning_latency_ms",
    "execution_latency_ms",
    "rag_latency_ms",
    "total_latency_ms",
)

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEFAULT_RULE_PATH = DEFAULT_REPORTS_DIR / "rule_baseline.json"
DEFAULT_LLM_PATH = DEFAULT_REPORTS_DIR / "llm_baseline.json"


class MetricComparison(BaseModel):
    """One metric, both sides, and the signed difference."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    lower_is_better: bool
    rule: Metric = None
    llm: Metric = None
    delta: Metric = Field(
        default=None, description="llm - rule. Null whenever either side is null."
    )
    verdict: str = VERDICT_NOT_COMPARABLE


class LatencyComparison(BaseModel):
    """One latency series, both sides. Distribution, not a single number."""

    model_config = ConfigDict(extra="forbid")

    series: str
    rule: LatencySummary
    llm: LatencySummary


class OODComparison(BaseModel):
    """Out-of-domain safety, both sides.

    For an out-of-domain case the answer key expects nothing, so every selected
    tool is an unnecessary call. The out-of-domain unnecessary-tool-call rate and
    the out-of-domain tool-call rate are therefore the same number, and this block
    reports it once. The dataset-wide ``unnecessary_tool_call_rate``, which also
    covers the in-domain cases that over-call, stays in the metric table above.
    """

    model_config = ConfigDict(extra="forbid")

    rule: OODSummary
    llm: OODSummary
    tool_call_rate_delta: Metric = None


class ComparisonReport(BaseModel):
    """The comparison artefact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    status: str
    generated_at: str
    dataset_path: str
    dataset_sha256: str
    dataset_case_count: int
    dataset_identical: bool
    rule_dataset_sha256: str
    llm_dataset_sha256: str
    provider: dict[str, str] | None = None
    model: str | None = None
    rule: MetricSet
    llm: MetricSet
    delta: dict[str, Metric] = Field(default_factory=dict)
    metrics: list[MetricComparison] = Field(default_factory=list)
    latency_ms: list[LatencyComparison] = Field(default_factory=list)
    ood_safety: OODComparison
    rule_run: dict[str, Any] = Field(default_factory=dict)
    llm_run: dict[str, Any] = Field(default_factory=dict)
    definitions: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class RefusalReport(BaseModel):
    """Why no comparison exists. It carries no metrics, by construction."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    status: str
    generated_at: str
    reason: str
    rule_report: str
    llm_report: str
    llm_report_exists: bool
    dataset_sha256: str | None = None
    rule_dataset_sha256: str | None = None
    llm_dataset_sha256: str | None = None
    required_command: list[str] = Field(default_factory=list)
    metrics: None = Field(
        default=None, description="Always null. A refusal states that nothing was compared."
    )


def _verdict(rule: Metric, llm: Metric, *, lower_is_better: bool) -> str:
    """Return how the LLM side reads against the rule side."""
    if rule is None or llm is None:
        return VERDICT_NOT_COMPARABLE
    difference = llm - rule
    if difference == 0:
        return VERDICT_UNCHANGED
    improved = difference < 0 if lower_is_better else difference > 0
    return VERDICT_BETTER if improved else VERDICT_WORSE


def compare_metrics(
    rule: MetricSet, llm: MetricSet
) -> tuple[dict[str, Metric], list[MetricComparison]]:
    """Compare every metric in :data:`COMPARED_METRICS`.

    The metric list is fixed in this module, so a baseline cannot shrink the
    comparison by omitting a field. A metric missing from a newer baseline is
    reported as not comparable rather than silently dropped.
    """
    rule_values = rule.model_dump()
    llm_values = llm.model_dump()

    delta: dict[str, Metric] = {}
    rows: list[MetricComparison] = []
    for name in COMPARED_METRICS:
        rule_value: Metric = rule_values.get(name)
        llm_value: Metric = llm_values.get(name)
        lower_is_better = name in LOWER_IS_BETTER
        difference: Metric = (
            None if rule_value is None or llm_value is None else llm_value - rule_value
        )
        delta[name] = difference
        rows.append(
            MetricComparison(
                metric=name,
                lower_is_better=lower_is_better,
                rule=rule_value,
                llm=llm_value,
                delta=difference,
                verdict=_verdict(rule_value, llm_value, lower_is_better=lower_is_better),
            )
        )
    return delta, rows


def _latency_rows(
    rule: dict[str, LatencySummary],
    llm: dict[str, LatencySummary],
) -> list[LatencyComparison]:
    """Pair up the latency series, keeping a series that only one side measured."""
    rows: list[LatencyComparison] = []
    for series in LATENCY_SERIES:
        rows.append(
            LatencyComparison(
                series=series,
                rule=rule.get(series, LatencySummary()),
                llm=llm.get(series, LatencySummary()),
            )
        )
    return rows


def _ood_comparison(rule: OODSummary, llm: OODSummary) -> OODComparison:
    """Compare out-of-domain safety, with the delta signed so negative means safer."""
    tool_call_delta: Metric = (
        None
        if rule.tool_call_rate is None or llm.tool_call_rate is None
        else llm.tool_call_rate - rule.tool_call_rate
    )
    return OODComparison(rule=rule, llm=llm, tool_call_rate_delta=tool_call_delta)


def _provider_metadata(llm: EvaluationReport) -> tuple[dict[str, str] | None, str | None]:
    """Return the provider block and model name from an LLM baseline.

    The credential is already absent from the stored report; ``base_url`` is
    re-sanitised here so a comparison cannot reintroduce userinfo.
    """
    described = llm.planner_runtime.provider
    if not described:
        return None, None
    safe = dict(described)
    if "base_url" in safe:
        safe["base_url"] = _safe_base_url(safe["base_url"])
    return safe, safe.get("model")


def _limitations(*, rule_mode: str, llm_mode: str) -> list[str]:
    """Return the caveats a reader needs to interpret the comparison."""
    return [
        "delta is llm - rule. For the error rates and for planning latency a "
        "negative delta is the better outcome; the lower_is_better flag states this "
        "per metric so the sign is never ambiguous.",
        "A metric whose value is null on either side has no delta. Null means the "
        "denominator was zero, so the metric was not measured for that planner.",
        "Every metric the two baselines produced appears here. Nothing is filtered, "
        "including the metrics where the LLM planner reads worse.",
        "The rule side is an existing baseline artefact and was not re-run for this "
        "comparison. rule_run records that artefact's own provenance.",
        "Both sides are planner-only runs, so execution and retrieval latency are "
        "null by construction and execution_error_rate is out of scope.",
        "For an out-of-domain case nothing is expected, so every selected tool is an "
        "unnecessary call and the out-of-domain unnecessary-tool-call rate equals the "
        "out-of-domain tool-call rate. That single number is reported once, in "
        "ood_safety.",
        "Latency p95 interpolates between order statistics, the convention "
        "evaluation.metrics.percentile documents. The tail is reported because a mean "
        "hides a slow provider call.",
        f"Planner modes compared: rule={rule_mode}, llm={llm_mode}.",
        "A planner-only comparison scores the plan. It says nothing about the prose "
        "either planner's downstream synthesis would write.",
    ]


def build_comparison(
    rule: EvaluationReport,
    llm: EvaluationReport,
    *,
    provider: dict[str, str] | None = None,
    model: str | None = None,
) -> ComparisonReport:
    """Build the comparison artefact from two real baseline reports.

    Args:
        rule: The frozen rule baseline.
        llm: The LLM baseline, which must have run on the identical dataset.
        provider: Provider metadata with the credential removed.
        model: The model identifier behind the LLM baseline.
    """
    delta, rows = compare_metrics(rule.metrics, llm.metrics)
    dataset_identical = rule.run.dataset_sha256 == llm.run.dataset_sha256

    return ComparisonReport(
        schema_version=SCHEMA_VERSION,
        status=STATUS_OK,
        generated_at=_utc_now(),
        dataset_path=rule.run.dataset_path,
        dataset_sha256=rule.run.dataset_sha256,
        dataset_case_count=rule.run.dataset_case_count,
        dataset_identical=dataset_identical,
        rule_dataset_sha256=rule.run.dataset_sha256,
        llm_dataset_sha256=llm.run.dataset_sha256,
        provider=provider,
        model=model,
        rule=rule.metrics,
        llm=llm.metrics,
        delta=delta,
        metrics=rows,
        latency_ms=_latency_rows(rule.latency_ms, llm.latency_ms),
        ood_safety=_ood_comparison(rule.ood_safety, llm.ood_safety),
        rule_run=_run_provenance(rule),
        llm_run=_run_provenance(llm),
        definitions={name: METRIC_DEFINITIONS[name] for name in COMPARED_METRICS},
        limitations=_limitations(
            rule_mode=rule.run.planner_mode,
            llm_mode=llm.run.planner_mode,
        ),
    )


def _run_provenance(report: EvaluationReport) -> dict[str, Any]:
    """Return the provenance fields of a baseline, so a change is visible."""
    return {
        "planner_mode": report.run.planner_mode,
        "run_mode": report.run.run_mode,
        "dataset_sha256": report.run.dataset_sha256,
        "dataset_case_count": report.run.dataset_case_count,
        "generated_at": report.run.generated_at,
        "app_version": report.run.app_version,
        "git_commit": report.run.git_commit,
        "git_worktree_dirty": report.run.git_worktree_dirty,
        "rag_provider_available": report.run.rag_provider_available,
    }


def _read_baseline(path: Path) -> EvaluationReport | None:
    """Return a baseline report, or ``None`` when it is absent or not a baseline."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != STATUS_OK:
        return None
    try:
        return EvaluationReport.model_validate(payload)
    except Exception:
        return None


def _is_gate_report(path: Path) -> bool:
    """Return ``True`` when the file exists and is a not-run gate report."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == STATUS_LLM_NOT_RUN


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the comparison CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.comparison",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rule",
        default=str(DEFAULT_RULE_PATH),
        help="Path to the rule baseline report.",
    )
    parser.add_argument(
        "--llm",
        default=str(DEFAULT_LLM_PATH),
        help="Path to the LLM baseline report. Absent means nothing to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory for the generated comparison report.",
    )
    parser.add_argument(
        "--tag",
        default="planner",
        help="Report filename prefix. The default produces planner_comparison.json.",
    )
    return parser


def _refuse(
    *,
    status: str,
    reason: str,
    rule_path: Path,
    llm_path: Path,
    output_dir: Path,
    tag: str,
    rule_hash: str | None = None,
    llm_hash: str | None = None,
) -> int:
    """Write and print a refusal, then return the refusal exit code."""
    refusal = RefusalReport(
        schema_version=SCHEMA_VERSION,
        status=status,
        generated_at=_utc_now(),
        reason=reason,
        rule_report=str(rule_path),
        llm_report=str(llm_path),
        llm_report_exists=llm_path.exists(),
        dataset_sha256=rule_hash,
        rule_dataset_sha256=rule_hash,
        llm_dataset_sha256=llm_hash,
        required_command=[
            "python",
            "-m",
            "evaluation.runner",
            "--planner",
            "llm",
            "--dataset",
            "evaluation/dataset.json",
        ],
    )
    _write(output_dir / f"{tag}_comparison_status.json", refusal)
    print(json.dumps(refusal.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return EXIT_REFUSED


def main(argv: list[str] | None = None) -> int:
    """Build the comparison. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    rule_path = Path(args.rule).resolve()
    llm_path = Path(args.llm).resolve()
    output_dir = Path(args.output_dir)

    rule = _read_baseline(rule_path)
    if rule is None:
        return _refuse(
            status=STATUS_INPUT_UNREADABLE,
            reason=(
                f"the rule baseline at {rule_path} is missing, unreadable, or not a "
                "successful run. Nothing can be compared without it."
            ),
            rule_path=rule_path,
            llm_path=llm_path,
            output_dir=output_dir,
            tag=args.tag,
        )

    llm = _read_baseline(llm_path)
    if llm is None:
        if _is_gate_report(llm_path):
            reason = (
                "the LLM baseline is a gate report, so no LLM planner was measured. "
                "Configure LLM_BASE_URL, LLM_API_KEY and LLM_MODEL, run the LLM "
                "benchmark, and re-run this command."
            )
        else:
            reason = (
                f"no successful LLM baseline exists at {llm_path}. Run the LLM "
                "benchmark first; a comparison cannot be approximated from the rule side."
            )
        return _refuse(
            status=STATUS_LLM_NOT_RUN,
            reason=reason,
            rule_path=rule_path,
            llm_path=llm_path,
            output_dir=output_dir,
            tag=args.tag,
            rule_hash=rule.run.dataset_sha256,
        )

    if rule.run.dataset_sha256 != llm.run.dataset_sha256:
        return _refuse(
            status=STATUS_DATASET_MISMATCH,
            reason=(
                "the two baselines were measured on different datasets, so their "
                "metrics are not comparable. Re-run the LLM benchmark on the dataset "
                "the rule baseline used."
            ),
            rule_path=rule_path,
            llm_path=llm_path,
            output_dir=output_dir,
            tag=args.tag,
            rule_hash=rule.run.dataset_sha256,
            llm_hash=llm.run.dataset_sha256,
        )

    provider, model = _provider_metadata(llm)
    comparison = build_comparison(rule, llm, provider=provider, model=model)

    _write(output_dir / f"{args.tag}_comparison.json", comparison)
    print(json.dumps(comparison.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

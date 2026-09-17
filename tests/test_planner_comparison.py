"""Tests for the Rule versus LLM planner comparison (V0.7).

The comparison is the artefact a reader will use to decide whether the LLM planner
is worth its cost, so the tests here attack the ways such a comparison can mislead:

1. **Dropping an inconvenient metric.** The compared list is fixed in the module and
   the report must carry every metric, including the ones where the LLM reads worse.
2. **A delta with an ambiguous sign.** ``delta`` is always ``llm - rule``, and the
   error rates and latency carry ``lower_is_better`` so a positive delta on those is
   not read as an improvement.
3. **Comparing across datasets.** Two baselines measured on different datasets are
   refused, and the refusal names both hashes.
4. **Publishing nulls under a comparison heading.** With no LLM baseline there is no
   comparison, so the command refuses, writes no comparison file, and exits non-zero.

No test here contacts a provider or needs a credential. The baselines are built from
the real report models, so a schema change breaks these tests rather than passing
silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evaluation.comparison import (
    COMPARED_METRICS,
    EXIT_OK,
    EXIT_REFUSED,
    LATENCY_SERIES,
    LOWER_IS_BETTER,
    STATUS_DATASET_MISMATCH,
    STATUS_INPUT_UNREADABLE,
    STATUS_LLM_NOT_RUN,
    VERDICT_BETTER,
    VERDICT_NOT_COMPARABLE,
    VERDICT_UNCHANGED,
    VERDICT_WORSE,
    ComparisonReport,
    MetricComparison,
    RefusalReport,
    build_comparison,
    build_parser,
    compare_metrics,
    main,
)
from evaluation.dataset import DEFAULT_DATASET_PATH, dataset_sha256
from evaluation.evaluator import (
    METRIC_DEFINITIONS,
    CountSet,
    LatencySummary,
    MetricSet,
    OODSummary,
)
from evaluation.runner import (
    STATUS_OK,
    EvaluationReport,
    GateReport,
    PlannerRuntime,
    RunMetadata,
)

#: The real rule baseline's metric values, so a synthetic LLM side can be compared
#: against a realistic reference rather than an invented one.
BASE_METRICS: dict[str, float] = {
    "intent_accuracy": 0.975609756097561,
    "tool_selection_exact_match": 0.7959183673469388,
    "tool_precision": 0.88,
    "tool_recall": 0.9850746268656716,
    "argument_accuracy": 1.0,
    "invalid_tool_rate": 0.0,
    "unnecessary_tool_call_rate": 0.12,
    "task_success_rate": 0.7959183673469388,
    "planner_failure_rate": 0.0,
    "average_planning_latency_ms": 0.24,
}

PLANNING_LATENCY = LatencySummary(
    sampled_cases=49,
    average_ms=0.24,
    median_ms=0.215,
    p95_ms=0.3368,
    min_ms=0.184,
    max_ms=0.496,
)

UNMEASURED = LatencySummary()

SILENT_OOD = OODSummary(
    case_count=8,
    silent_case_count=5,
    tool_call_case_count=3,
    tool_call_rate=0.375,
    offending_case_ids=["ood-006", "ood-007", "ood-008"],
)


def make_report(
    *,
    planner_mode: str,
    metrics: dict[str, float | None] | None = None,
    dataset_hash: str | None = None,
    ood: OODSummary | None = None,
    provider: dict[str, str] | None = None,
) -> EvaluationReport:
    """Return a baseline report shaped like a real one, with overrides applied."""
    values: dict[str, float | None] = dict(BASE_METRICS)
    if metrics:
        values.update(metrics)
    return EvaluationReport(
        schema_version="1.0",
        status=STATUS_OK,
        metrics=MetricSet(**values),
        denominators=CountSet(total_cases=49, ood_cases=8),
        latency_ms={
            "planning_latency_ms": PLANNING_LATENCY,
            "execution_latency_ms": UNMEASURED,
            "rag_latency_ms": UNMEASURED,
            "total_latency_ms": PLANNING_LATENCY,
        },
        categories={},
        ood_safety=ood if ood is not None else SILENT_OOD,
        planner_runtime=PlannerRuntime(provider=provider),
        run=RunMetadata(
            generated_at="2026-09-15T00:00:00+00:00",
            dataset_path=str(DEFAULT_DATASET_PATH),
            dataset_sha256=dataset_hash or dataset_sha256(DEFAULT_DATASET_PATH),
            dataset_case_count=49,
            planner_mode=planner_mode,
            run_mode="planner_only",
            execute_tools=False,
            app_version="0.9.0",
            python_version="3.11.9",
            git_commit="a" * 40,
            git_worktree_dirty=False,
        ),
        definitions=METRIC_DEFINITIONS,
    )


def rows_by_metric(report: ComparisonReport) -> dict[str, MetricComparison]:
    """Return the metric rows keyed by metric name."""
    return {row.metric: row for row in report.metrics}


def write_baseline(path: Path, report: EvaluationReport) -> Path:
    """Write a baseline report where the comparison CLI expects to read it."""
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    return path


def refusal_payload(output_dir: Path) -> dict[str, Any]:
    """Return the refusal artefact the CLI wrote."""
    text = (output_dir / "planner_comparison_status.json").read_text(encoding="utf-8")
    return json.loads(text)


# --------------------------------------------------------------------------- #
# The metric table: nothing filtered, signs unambiguous
# --------------------------------------------------------------------------- #


def test_every_compared_metric_appears_even_when_the_llm_is_worse() -> None:
    rule = make_report(planner_mode="rule")
    llm = make_report(planner_mode="llm", metrics={"intent_accuracy": 0.80, "tool_recall": 1.0})

    comparison = build_comparison(rule, llm)
    rows = rows_by_metric(comparison)

    assert [row.metric for row in comparison.metrics] == list(COMPARED_METRICS)
    assert set(comparison.delta) == set(COMPARED_METRICS)
    # The metric where the LLM is worse is present and marked worse. A comparison
    # that hid it would be an advertisement.
    assert rows["intent_accuracy"].verdict == VERDICT_WORSE
    assert rows["tool_recall"].verdict == VERDICT_BETTER


def test_delta_is_always_llm_minus_rule() -> None:
    rule = make_report(planner_mode="rule")
    llm = make_report(planner_mode="llm", metrics={"intent_accuracy": 0.80})

    comparison = build_comparison(rule, llm)
    rows = rows_by_metric(comparison)
    expected = 0.80 - BASE_METRICS["intent_accuracy"]

    assert rows["intent_accuracy"].delta == pytest.approx(expected)
    assert comparison.delta["intent_accuracy"] == pytest.approx(expected)


def test_a_lower_error_rate_reads_as_better() -> None:
    rule = make_report(planner_mode="rule", metrics={"unnecessary_tool_call_rate": 0.12})
    llm = make_report(planner_mode="llm", metrics={"unnecessary_tool_call_rate": 0.04})

    rows = rows_by_metric(build_comparison(rule, llm))

    assert rows["unnecessary_tool_call_rate"].lower_is_better is True
    assert rows["unnecessary_tool_call_rate"].delta == pytest.approx(-0.08)
    assert rows["unnecessary_tool_call_rate"].verdict == VERDICT_BETTER


def test_a_positive_delta_on_a_lower_is_better_metric_reads_as_worse() -> None:
    rule = make_report(planner_mode="rule", metrics={"invalid_tool_rate": 0.0})
    llm = make_report(planner_mode="llm", metrics={"invalid_tool_rate": 0.04})

    rows = rows_by_metric(build_comparison(rule, llm))

    assert rows["invalid_tool_rate"].delta == pytest.approx(0.04)
    assert rows["invalid_tool_rate"].verdict == VERDICT_WORSE


def test_an_identical_metric_reads_as_unchanged() -> None:
    rows = rows_by_metric(
        build_comparison(make_report(planner_mode="rule"), make_report(planner_mode="llm"))
    )

    assert rows["tool_precision"].delta == 0.0
    assert rows["tool_precision"].verdict == VERDICT_UNCHANGED


def test_a_metric_null_on_either_side_has_no_delta() -> None:
    rule = make_report(planner_mode="rule", metrics={"tool_precision": None})
    llm = make_report(planner_mode="llm")

    rows = rows_by_metric(build_comparison(rule, llm))

    # Null means the denominator was zero, so the metric was never measured for
    # that planner. An unmeasured metric has no verdict.
    assert rows["tool_precision"].rule is None
    assert rows["tool_precision"].delta is None
    assert rows["tool_precision"].verdict == VERDICT_NOT_COMPARABLE


def test_compare_metrics_returns_exactly_the_declared_metrics() -> None:
    rule = make_report(planner_mode="rule")
    llm = make_report(planner_mode="llm")

    delta, rows = compare_metrics(rule.metrics, llm.metrics)

    assert [row.metric for row in rows] == list(COMPARED_METRICS)
    assert set(delta) == set(COMPARED_METRICS)


def test_the_compared_metric_list_is_fixed_in_the_module() -> None:
    assert set(COMPARED_METRICS) <= set(METRIC_DEFINITIONS)
    assert set(COMPARED_METRICS) <= set(MetricSet.model_fields)
    assert LOWER_IS_BETTER <= set(COMPARED_METRICS)
    # A planner-only comparison has no execution step, so execution_error_rate is
    # deliberately outside the compared list.
    assert "execution_error_rate" not in COMPARED_METRICS


# --------------------------------------------------------------------------- #
# Provenance and the dataset lock
# --------------------------------------------------------------------------- #


def test_the_rule_side_is_carried_as_provenance_not_recomputed() -> None:
    rule = make_report(planner_mode="rule")
    llm = make_report(planner_mode="llm")

    comparison = build_comparison(rule, llm)

    assert comparison.rule_run["planner_mode"] == "rule"
    assert comparison.rule_run["git_commit"] == rule.run.git_commit
    assert comparison.rule_run["dataset_sha256"] == rule.run.dataset_sha256
    assert comparison.rule.intent_accuracy == rule.metrics.intent_accuracy
    assert comparison.llm.intent_accuracy == llm.metrics.intent_accuracy


def test_the_comparison_is_locked_to_the_shipped_dataset() -> None:
    comparison = build_comparison(make_report(planner_mode="rule"), make_report(planner_mode="llm"))

    assert comparison.dataset_sha256 == dataset_sha256(DEFAULT_DATASET_PATH)
    assert comparison.dataset_case_count == 49
    assert comparison.dataset_identical is True


def test_baselines_measured_on_different_datasets_are_flagged() -> None:
    comparison = build_comparison(
        make_report(planner_mode="rule"),
        make_report(planner_mode="llm", dataset_hash="f" * 64),
    )

    assert comparison.dataset_identical is False
    assert comparison.rule_dataset_sha256 != comparison.llm_dataset_sha256


# --------------------------------------------------------------------------- #
# Latency distribution and OOD safety
# --------------------------------------------------------------------------- #


def test_the_comparison_pairs_all_four_latency_series() -> None:
    comparison = build_comparison(make_report(planner_mode="rule"), make_report(planner_mode="llm"))

    assert [row.series for row in comparison.latency_ms] == list(LATENCY_SERIES)
    planning = next(row for row in comparison.latency_ms if row.series == "planning_latency_ms")
    assert planning.rule.p95_ms == pytest.approx(0.3368)
    assert planning.llm.p95_ms == pytest.approx(0.3368)


def test_an_unmeasured_latency_series_stays_null_on_both_sides() -> None:
    comparison = build_comparison(make_report(planner_mode="rule"), make_report(planner_mode="llm"))

    rag = next(row for row in comparison.latency_ms if row.series == "rag_latency_ms")
    assert rag.rule.sampled_cases == 0
    assert rag.llm.sampled_cases == 0
    assert rag.rule.p95_ms is None
    assert rag.llm.average_ms is None


def test_the_ood_block_reports_both_sides_and_the_delta() -> None:
    safer = OODSummary(
        case_count=8,
        silent_case_count=8,
        tool_call_case_count=0,
        tool_call_rate=0.0,
        offending_case_ids=[],
    )
    comparison = build_comparison(
        make_report(planner_mode="rule"),
        make_report(planner_mode="llm", ood=safer),
    )

    assert comparison.ood_safety.rule.tool_call_rate == pytest.approx(0.375)
    assert comparison.ood_safety.llm.tool_call_rate == 0.0
    # Negative means the LLM planner reached for fewer industrial tools out of domain.
    assert comparison.ood_safety.tool_call_rate_delta == pytest.approx(-0.375)


def test_the_comparison_states_why_the_ood_rates_collapse_into_one_number() -> None:
    comparison = build_comparison(make_report(planner_mode="rule"), make_report(planner_mode="llm"))

    assert any("unnecessary-tool-call rate equals" in note for note in comparison.limitations)
    assert any("Nothing is filtered" in note for note in comparison.limitations)
    assert any("percentile" in note for note in comparison.limitations)


def test_a_comparison_report_revalidates_after_serialisation() -> None:
    comparison = build_comparison(make_report(planner_mode="rule"), make_report(planner_mode="llm"))

    restored = ComparisonReport.model_validate(json.loads(comparison.model_dump_json()))

    assert restored == comparison


# --------------------------------------------------------------------------- #
# The CLI: refuse rather than approximate
# --------------------------------------------------------------------------- #


def test_a_missing_llm_baseline_is_refused_and_names_the_fixing_command(tmp_path: Path) -> None:
    rule_path = write_baseline(tmp_path / "rule.json", make_report(planner_mode="rule"))
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--rule",
            str(rule_path),
            "--llm",
            str(tmp_path / "absent.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == EXIT_REFUSED
    payload = refusal_payload(output_dir)
    assert payload["status"] == STATUS_LLM_NOT_RUN
    assert payload["metrics"] is None
    assert payload["llm_report_exists"] is False
    assert "evaluation.runner" in payload["required_command"]
    # No comparison file exists, so nulls can never be read as a result.
    assert not (output_dir / "planner_comparison.json").exists()


def test_a_gate_report_on_the_llm_side_is_refused_as_not_run(tmp_path: Path) -> None:
    rule_path = write_baseline(tmp_path / "rule.json", make_report(planner_mode="rule"))
    gate = GateReport(
        schema_version="1.0",
        status=STATUS_LLM_NOT_RUN,
        planner_mode="llm",
        generated_at="2026-09-15T00:00:00+00:00",
        dataset_path=str(DEFAULT_DATASET_PATH),
        dataset_sha256=dataset_sha256(DEFAULT_DATASET_PATH),
        dataset_case_count=49,
        reason="no provider",
    )
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps(gate.model_dump(mode="json")), encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = main(
        ["--rule", str(rule_path), "--llm", str(llm_path), "--output-dir", str(output_dir)]
    )

    assert exit_code == EXIT_REFUSED
    payload = refusal_payload(output_dir)
    assert payload["status"] == STATUS_LLM_NOT_RUN
    assert "gate report" in payload["reason"]
    assert not (output_dir / "planner_comparison.json").exists()


def test_mismatched_datasets_are_refused_with_both_hashes(tmp_path: Path) -> None:
    rule_path = write_baseline(tmp_path / "rule.json", make_report(planner_mode="rule"))
    llm_path = write_baseline(
        tmp_path / "llm.json", make_report(planner_mode="llm", dataset_hash="f" * 64)
    )
    output_dir = tmp_path / "out"

    exit_code = main(
        ["--rule", str(rule_path), "--llm", str(llm_path), "--output-dir", str(output_dir)]
    )

    assert exit_code == EXIT_REFUSED
    payload = refusal_payload(output_dir)
    assert payload["status"] == STATUS_DATASET_MISMATCH
    assert payload["rule_dataset_sha256"] != payload["llm_dataset_sha256"]
    assert payload["metrics"] is None
    assert not (output_dir / "planner_comparison.json").exists()


def test_an_unreadable_rule_baseline_is_refused(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--rule",
            str(tmp_path / "nope.json"),
            "--llm",
            str(tmp_path / "nope2.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == EXIT_REFUSED
    assert refusal_payload(output_dir)["status"] == STATUS_INPUT_UNREADABLE


def test_a_complete_pair_produces_the_comparison_and_strips_the_credential(
    tmp_path: Path,
) -> None:
    rule_path = write_baseline(tmp_path / "rule.json", make_report(planner_mode="rule"))
    llm_path = write_baseline(
        tmp_path / "llm.json",
        make_report(
            planner_mode="llm",
            metrics={"intent_accuracy": 0.90},
            provider={
                "model": "example-model",
                "base_url": "https://user:secret@api.example.com/v1",
            },
        ),
    )
    output_dir = tmp_path / "out"

    exit_code = main(
        ["--rule", str(rule_path), "--llm", str(llm_path), "--output-dir", str(output_dir)]
    )

    assert exit_code == EXIT_OK
    raw = (output_dir / "planner_comparison.json").read_text(encoding="utf-8")
    report = ComparisonReport.model_validate(json.loads(raw))

    assert report.status == STATUS_OK
    assert report.model == "example-model"
    assert report.provider is not None
    assert report.provider["base_url"] == "https://api.example.com/v1"
    # A credential embedded in the endpoint never reaches the artefact.
    assert "secret" not in raw
    assert "user:" not in raw
    assert report.delta["intent_accuracy"] == pytest.approx(0.90 - BASE_METRICS["intent_accuracy"])


def test_a_refusal_report_carries_no_metrics_by_construction() -> None:
    refusal = RefusalReport(
        schema_version="1.0",
        status=STATUS_LLM_NOT_RUN,
        generated_at="2026-09-15T00:00:00+00:00",
        reason="no baseline",
        rule_report="a.json",
        llm_report="b.json",
        llm_report_exists=False,
    )

    assert refusal.metrics is None
    with pytest.raises(ValidationError):
        RefusalReport.model_validate({**refusal.model_dump(), "metrics": {"intent_accuracy": 1.0}})


def test_the_comparison_cli_defaults_match_the_documented_command() -> None:
    args = build_parser().parse_args([])

    assert args.tag == "planner"
    assert Path(args.rule).name == "rule_baseline.json"
    assert Path(args.llm).name == "llm_baseline.json"
    assert Path(args.output_dir).name == "reports"

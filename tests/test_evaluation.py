"""Tests for the Agent Evaluation Framework (V0.6).

This suite is the acceptance test for the evaluation harness, and it guards four
promises that the harness makes. Each promise is a way for a benchmark to lie:

1. **A metric with no observations is ``null``.** A planner that never selects a
   tool has no precision. Reporting ``1.0`` for it would be a fabricated success,
   so every zero denominator must surface as ``None`` and stay recomputable from
   the published counts.
2. **Task success is judged on the plan.** Execution outcomes are reported
   separately, so a planner-only run and an end-to-end run of the same planner
   stay directly comparable.
3. **An out-of-domain query must not reach for an industrial tool.** The signal
   is carried by the OOD block rather than by an intent score, because there is no
   maintenance intent to get right.
4. **An LLM run without a provider measures nothing.** It reports
   ``LLM_EVALUATION_NOT_RUN`` and writes no metrics at all. A stub or a fabricated
   score would be worse than no number.

The last promise is also a regression guard on the answer key. The rule planner is
a frozen baseline and the dataset is a frozen artifact, so the cases where the
baseline is wrong are pinned here by identifier. If a future change makes those
cases pass, this suite fails: the honest reading is that the answer key moved, not
that the planner improved.

Nothing here needs a credential, a network socket or a database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evaluation import runner as runner_module
from evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    INTENT_VOCABULARY,
    EvalCase,
    dataset_sha256,
    load_dataset,
    parse_dataset,
)
from evaluation.evaluator import (
    FAILURE_ARGUMENT_MISMATCH,
    FAILURE_EXECUTION_ERROR,
    FAILURE_INTENT_MISMATCH,
    FAILURE_INVALID_TOOL,
    FAILURE_MISSING_TOOL,
    FAILURE_OOD_TOOL_CALL,
    FAILURE_PLANNING_ERROR,
    FAILURE_PRIORITY,
    FAILURE_UNNECESSARY_TOOL,
    METRIC_DEFINITIONS,
    aggregate,
    evaluate_case,
)
from evaluation.metrics import (
    argument_values_match,
    invalid_tools,
    mean,
    median,
    missing_tools,
    ratio,
    tool_counts,
    unnecessary_tools,
)
from evaluation.runner import (
    STATUS_LLM_NOT_RUN,
    STATUS_OK,
    EvaluationReport,
    FailureReport,
    GateReport,
    build_parser,
    main,
)

DEVICE_TOOL = "get_device_status"
ALARM_TOOL = "query_alarm_code"
MANUAL_TOOL = "search_maintenance_manual"

#: The registry as the shipped application defines it. Passed explicitly so the
#: evaluator's "is this a real tool" question is answered the same way every run.
KNOWN_TOOLS = {DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL}

#: The ten metrics the framework must compute. ``execution_error_rate`` is the
#: eleventh and is only meaningful once tools have run.
REQUIRED_METRICS = {
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
}


def make_case(**overrides: Any) -> EvalCase:
    """Return a minimal in-scope case with the given answer-key overrides."""
    payload: dict[str, Any] = {
        "id": "case-001",
        "category": "device_status",
        "query": "PLC-001 现在什么状态",
        "expected_intent": "device_status",
        "expected_tools": [DEVICE_TOOL],
        "expected_arguments": {DEVICE_TOOL: {"device_id": "PLC-001"}},
    }
    payload.update(overrides)
    return EvalCase.model_validate(payload)


def make_ood_case(case_id: str = "ood-001", query: str = "今天北京天气怎么样") -> EvalCase:
    """Return an out-of-domain case, which must expect no tool and no intent."""
    return EvalCase.model_validate(
        {
            "id": case_id,
            "category": "ood",
            "query": query,
            "expected_intent": None,
            "expected_tools": [],
            "expected_arguments": {},
        }
    )


def make_mini_payload() -> dict[str, Any]:
    """Return a two-case synthetic dataset, for CLI plumbing tests."""
    return {
        "schema_version": "1.0",
        "name": "mini",
        "description": "synthetic two-case dataset",
        "categories": {"device_status": "device condition", "ood": "out of domain"},
        "cases": [
            {
                "id": "m-1",
                "category": "device_status",
                "query": "PLC-001 现在什么状态",
                "expected_intent": "device_status",
                "expected_tools": [DEVICE_TOOL],
                "expected_arguments": {DEVICE_TOOL: {"device_id": "PLC-001"}},
            },
            {
                "id": "m-2",
                "category": "ood",
                "query": "今天北京天气怎么样",
                "expected_intent": None,
                "expected_tools": [],
            },
        ],
    }


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #


def test_ratio_returns_none_rather_than_a_perfect_score_when_nothing_was_measured() -> None:
    assert ratio(0, 0) is None
    assert ratio(3, 0) is None


def test_ratio_keeps_a_genuine_zero_distinct_from_an_unmeasured_none() -> None:
    assert ratio(1, 2) == 0.5
    # A real zero: four observations, none of them positive.
    assert ratio(0, 4) == 0.0


def test_mean_and_median_report_none_for_an_empty_sample() -> None:
    assert mean([]) is None
    assert median([]) is None


def test_median_handles_odd_and_even_samples() -> None:
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_tool_counts_treats_selection_as_a_set() -> None:
    # Order is not scored, and a duplicate mention neither helps nor hurts.
    assert tool_counts([DEVICE_TOOL, ALARM_TOOL], [ALARM_TOOL, DEVICE_TOOL]) == (2, 0, 0)
    assert tool_counts([DEVICE_TOOL], [DEVICE_TOOL, DEVICE_TOOL]) == (1, 0, 0)


def test_tool_counts_separates_a_wrong_tool_from_a_missing_one() -> None:
    true_positive, false_positive, false_negative = tool_counts(
        [DEVICE_TOOL, MANUAL_TOOL], [DEVICE_TOOL, ALARM_TOOL]
    )
    assert (true_positive, false_positive, false_negative) == (1, 1, 1)


def test_tool_set_differences_are_sorted_and_deduplicated() -> None:
    assert missing_tools([ALARM_TOOL, DEVICE_TOOL], [DEVICE_TOOL]) == [ALARM_TOOL]
    assert unnecessary_tools([DEVICE_TOOL], [MANUAL_TOOL, ALARM_TOOL]) == [ALARM_TOOL, MANUAL_TOOL]
    assert invalid_tools([DEVICE_TOOL, "teleport_device"], KNOWN_TOOLS) == ["teleport_device"]


def test_argument_values_match_scores_only_the_declared_keys() -> None:
    # An optional argument the planner legitimately adds is not an error.
    assert (
        argument_values_match({"device_id": "PLC-001"}, {"device_id": "PLC-001", "top_k": 4})
        is True
    )
    assert argument_values_match({"device_id": "PLC-001"}, {"device_id": "PLC-002"}) is False
    # Declared but absent counts as a mismatch, not as a pass.
    assert argument_values_match({"device_id": "PLC-001"}, {}) is False


def test_argument_values_match_passes_vacuously_when_nothing_is_declared() -> None:
    assert argument_values_match({}, {"anything": 1}) is True


# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #


def test_a_correct_plan_succeeds_and_records_no_failure() -> None:
    observation = runner_module.PlanObservation(
        intent="device_status",
        tools=[DEVICE_TOOL],
        arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
    )
    outcome = evaluate_case(make_case(), observation, known_tools=KNOWN_TOOLS)

    assert outcome.task_success is True
    assert outcome.failure_types == []
    assert outcome.primary_failure_type is None
    assert outcome.intent_correct is True
    assert outcome.tool_exact_match is True
    assert outcome.argument_correct is True


def test_an_intent_mismatch_is_the_headline_failure_and_fails_the_case() -> None:
    observation = runner_module.PlanObservation(
        intent="maintenance_advice",
        tools=[DEVICE_TOOL],
        arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
    )
    outcome = evaluate_case(make_case(), observation, known_tools=KNOWN_TOOLS)

    assert FAILURE_INTENT_MISMATCH in outcome.failure_types
    assert outcome.primary_failure_type == FAILURE_INTENT_MISMATCH
    assert outcome.task_success is False


def test_a_missing_tool_lowers_recall_without_touching_precision() -> None:
    case = make_case(expected_tools=[DEVICE_TOOL, MANUAL_TOOL])
    observation = runner_module.PlanObservation(
        intent="device_status",
        tools=[DEVICE_TOOL],
        arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
    )
    outcome = evaluate_case(case, observation, known_tools=KNOWN_TOOLS)

    assert outcome.missing_tools == [MANUAL_TOOL]
    assert outcome.unnecessary_tools == []
    assert outcome.tool_false_negative == 1
    assert outcome.tool_false_positive == 0
    assert FAILURE_MISSING_TOOL in outcome.failure_types
    assert outcome.task_success is False


def test_an_extra_tool_is_recorded_as_unnecessary() -> None:
    observation = runner_module.PlanObservation(
        intent="device_status",
        tools=[DEVICE_TOOL, MANUAL_TOOL],
        arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
    )
    outcome = evaluate_case(make_case(), observation, known_tools=KNOWN_TOOLS)

    assert outcome.unnecessary_tools == [MANUAL_TOOL]
    assert outcome.tool_false_positive == 1
    assert outcome.tool_exact_match is False
    assert FAILURE_UNNECESSARY_TOOL in outcome.failure_types


def test_an_invented_tool_is_invalid_and_takes_priority_over_being_unnecessary() -> None:
    observation = runner_module.PlanObservation(
        intent="device_status",
        tools=[DEVICE_TOOL, "teleport_device"],
        arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
    )
    outcome = evaluate_case(make_case(), observation, known_tools=KNOWN_TOOLS)

    assert outcome.invalid_tools == ["teleport_device"]
    # A name the executor cannot run is a distinct, more serious problem than a
    # real tool that was not needed, so it is the headline.
    assert FAILURE_INVALID_TOOL in outcome.failure_types
    assert FAILURE_UNNECESSARY_TOOL in outcome.failure_types
    assert outcome.primary_failure_type == FAILURE_INVALID_TOOL


def test_a_wrong_declared_argument_fails_the_argument_check() -> None:
    case = make_case(expected_arguments={DEVICE_TOOL: {"device_id": "PLC-999"}})
    observation = runner_module.PlanObservation(
        intent="device_status",
        tools=[DEVICE_TOOL],
        arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
    )
    outcome = evaluate_case(case, observation, known_tools=KNOWN_TOOLS)

    assert outcome.argument_checks_total == 1
    assert outcome.argument_checks_passed == 0
    assert outcome.argument_correct is False
    assert FAILURE_ARGUMENT_MISMATCH in outcome.failure_types
    assert outcome.task_success is False


def test_arguments_are_not_scored_for_a_tool_that_was_never_selected() -> None:
    observation = runner_module.PlanObservation(intent="device_status", tools=[])
    outcome = evaluate_case(make_case(), observation, known_tools=KNOWN_TOOLS)

    # No check exists, because tool recall already penalises the omission.
    assert outcome.argument_checks_total == 0
    assert outcome.argument_correct is True
    assert FAILURE_MISSING_TOOL in outcome.failure_types
    assert FAILURE_ARGUMENT_MISMATCH not in outcome.failure_types


def test_a_silent_out_of_domain_case_succeeds_and_is_not_intent_scored() -> None:
    observation = runner_module.PlanObservation(intent="unknown", tools=[])
    outcome = evaluate_case(make_ood_case(), observation, known_tools=KNOWN_TOOLS)

    # There is no maintenance intent to classify, so the field is not a score.
    assert outcome.intent_correct is None
    assert outcome.task_success is True
    assert outcome.failure_types == []


def test_an_out_of_domain_case_that_reaches_for_a_tool_is_flagged() -> None:
    observation = runner_module.PlanObservation(tools=[MANUAL_TOOL])
    outcome = evaluate_case(make_ood_case(), observation, known_tools=KNOWN_TOOLS)

    assert FAILURE_OOD_TOOL_CALL in outcome.failure_types
    assert outcome.primary_failure_type == FAILURE_OOD_TOOL_CALL
    assert outcome.task_success is False


def test_a_planning_error_fails_the_case_and_becomes_the_headline() -> None:
    observation = runner_module.PlanObservation(planning_error="LLM_PLANNER_TIMEOUT")
    outcome = evaluate_case(make_case(), observation, known_tools=KNOWN_TOOLS)

    assert FAILURE_PLANNING_ERROR in outcome.failure_types
    assert outcome.primary_failure_type == FAILURE_PLANNING_ERROR
    assert outcome.intent_correct is False
    assert outcome.task_success is False


def test_execution_errors_are_reported_without_changing_plan_level_success() -> None:
    observation = runner_module.PlanObservation(
        intent="device_status",
        tools=[DEVICE_TOOL],
        arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
        execution_errors=[f"{MANUAL_TOOL}: provider unavailable"],
    )
    outcome = evaluate_case(make_case(), observation, known_tools=KNOWN_TOOLS)

    # The plan was right, so the plan-level metric stays a success. The execution
    # problem is carried by failure_types and, in aggregate, by the execution rate.
    assert outcome.task_success is True
    assert outcome.failure_types == [FAILURE_EXECUTION_ERROR]


def test_a_plan_observation_rejects_undeclared_fields() -> None:
    with pytest.raises(ValidationError):
        runner_module.PlanObservation.model_validate({"intent": "device_status", "typo": 1})


# --------------------------------------------------------------------------- #
# Aggregation: precision, recall, denominators
# --------------------------------------------------------------------------- #


def _tool_outcomes() -> list[Any]:
    """Three cases whose tool selections pool to a known precision and recall."""
    correct = evaluate_case(
        make_case(id="t-1"),
        runner_module.PlanObservation(
            intent="device_status",
            tools=[DEVICE_TOOL],
            arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
        ),
        known_tools=KNOWN_TOOLS,
    )
    extra = evaluate_case(
        make_case(id="t-2"),
        runner_module.PlanObservation(
            intent="device_status",
            tools=[DEVICE_TOOL, MANUAL_TOOL],
            arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
        ),
        known_tools=KNOWN_TOOLS,
    )
    short = evaluate_case(
        make_case(id="t-3", expected_tools=[DEVICE_TOOL, MANUAL_TOOL]),
        runner_module.PlanObservation(
            intent="device_status",
            tools=[DEVICE_TOOL],
            arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
        ),
        known_tools=KNOWN_TOOLS,
    )
    return [correct, extra, short]


def test_precision_and_recall_are_pooled_micro_across_cases() -> None:
    metrics, counts, _, _, _ = aggregate(_tool_outcomes(), executed=False)

    assert (counts.tool_true_positive, counts.tool_false_positive, counts.tool_false_negative) == (
        3,
        1,
        1,
    )
    assert metrics.tool_precision == pytest.approx(3 / 4)
    assert metrics.tool_recall == pytest.approx(3 / 4)


def test_every_published_ratio_can_be_recomputed_from_its_denominator() -> None:
    metrics, counts, _, _, _ = aggregate(_tool_outcomes(), executed=False)

    assert metrics.tool_selection_exact_match == pytest.approx(
        counts.tool_exact_match_cases / counts.total_cases
    )
    assert metrics.intent_accuracy == pytest.approx(
        counts.intent_correct_cases / counts.intent_evaluated_cases
    )
    assert metrics.unnecessary_tool_call_rate == pytest.approx(
        counts.unnecessary_tool_selections / counts.selected_tool_selections
    )
    assert metrics.average_planning_latency_ms is None
    assert counts.planning_latency_samples == 0


def test_out_of_domain_cases_are_excluded_from_intent_accuracy() -> None:
    in_scope = evaluate_case(
        make_case(id="i-1"),
        runner_module.PlanObservation(
            intent="device_status",
            tools=[DEVICE_TOOL],
            arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
        ),
        known_tools=KNOWN_TOOLS,
    )
    ood = evaluate_case(
        make_ood_case(),
        runner_module.PlanObservation(tools=[]),
        known_tools=KNOWN_TOOLS,
    )

    metrics, counts, _, _, ood_summary = aggregate([in_scope, ood], executed=False)

    assert counts.total_cases == 2
    assert counts.ood_cases == 1
    assert counts.intent_evaluated_cases == 1
    assert metrics.intent_accuracy == 1.0
    assert ood_summary.case_count == 1


def test_the_ood_block_reports_silence_and_offenders_separately() -> None:
    silent = evaluate_case(
        make_ood_case("ood-a"), runner_module.PlanObservation(tools=[]), known_tools=KNOWN_TOOLS
    )
    noisy = evaluate_case(
        make_ood_case("ood-b"),
        runner_module.PlanObservation(tools=[MANUAL_TOOL]),
        known_tools=KNOWN_TOOLS,
    )

    _, _, _, _, summary = aggregate([silent, noisy], executed=False)

    assert summary.case_count == 2
    assert summary.silent_case_count == 1
    assert summary.tool_call_case_count == 1
    assert summary.tool_call_rate == 0.5
    assert summary.offending_case_ids == ["ood-b"]


def test_a_planner_that_selects_nothing_has_no_precision_rather_than_perfect_precision() -> None:
    outcome = evaluate_case(
        make_case(),
        runner_module.PlanObservation(intent="device_status", tools=[]),
        known_tools=KNOWN_TOOLS,
    )

    metrics, counts, _, _, _ = aggregate([outcome], executed=False)

    assert counts.selected_tool_selections == 0
    assert metrics.tool_precision is None
    assert metrics.invalid_tool_rate is None
    assert metrics.unnecessary_tool_call_rate is None
    # Recall is a genuine zero: one tool was expected and none arrived.
    assert metrics.tool_recall == 0.0


def test_an_empty_dataset_measures_nothing_rather_than_everything() -> None:
    metrics, counts, latency, categories, summary = aggregate([], executed=False)

    assert counts.total_cases == 0
    assert metrics.intent_accuracy is None
    assert metrics.tool_selection_exact_match is None
    assert metrics.task_success_rate is None
    assert metrics.planner_failure_rate is None
    assert metrics.average_planning_latency_ms is None
    assert summary.tool_call_rate is None
    assert categories == {}
    assert latency["planning_latency_ms"].sampled_cases == 0


def test_a_planner_only_run_reports_no_execution_metric_even_after_an_error() -> None:
    outcome = evaluate_case(
        make_case(),
        runner_module.PlanObservation(
            intent="device_status",
            tools=[DEVICE_TOOL],
            arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
            execution_errors=[f"{MANUAL_TOOL}: provider unavailable"],
        ),
        known_tools=KNOWN_TOOLS,
    )

    metrics, counts, latency, _, _ = aggregate([outcome], executed=False)

    # Nothing was executed, so the rate is not applicable. Zero would be a claim.
    assert metrics.execution_error_rate is None
    assert counts.executed_cases == 0
    assert latency["execution_latency_ms"].sampled_cases == 0
    assert latency["execution_latency_ms"].average_ms is None


def test_an_end_to_end_run_does_report_the_execution_error_rate() -> None:
    def executed(case_id: str, *, broke: bool) -> Any:
        return evaluate_case(
            make_case(id=case_id),
            runner_module.PlanObservation(
                intent="device_status",
                tools=[DEVICE_TOOL],
                arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
                execution_latency_ms=10.0,
                execution_errors=["x"] if broke else [],
            ),
            known_tools=KNOWN_TOOLS,
        )

    metrics, counts, latency, _, _ = aggregate(
        [executed("e-1", broke=True), executed("e-2", broke=False)], executed=True
    )

    assert counts.executed_cases == 2
    assert counts.execution_error_cases == 1
    assert metrics.execution_error_rate == 0.5
    assert latency["execution_latency_ms"].sampled_cases == 2
    assert latency["execution_latency_ms"].average_ms == 10.0


def test_a_latency_series_with_no_samples_is_null_not_zero() -> None:
    outcome = evaluate_case(
        make_case(),
        runner_module.PlanObservation(
            intent="device_status",
            tools=[DEVICE_TOOL],
            arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
        ),
        known_tools=KNOWN_TOOLS,
    )

    _, _, latency, _, _ = aggregate([outcome], executed=True)

    assert latency["rag_latency_ms"].sampled_cases == 0
    assert latency["rag_latency_ms"].average_ms is None
    assert latency["rag_latency_ms"].median_ms is None


def test_per_category_success_is_reported_separately_from_the_total() -> None:
    good = evaluate_case(
        make_case(id="c-1"),
        runner_module.PlanObservation(
            intent="device_status",
            tools=[DEVICE_TOOL],
            arguments={DEVICE_TOOL: {"device_id": "PLC-001"}},
        ),
        known_tools=KNOWN_TOOLS,
    )
    bad = evaluate_case(
        make_case(id="c-2"),
        runner_module.PlanObservation(intent="unknown", tools=[]),
        known_tools=KNOWN_TOOLS,
    )

    _, _, _, categories, _ = aggregate([good, bad], executed=False)

    summary = categories["device_status"]
    assert summary.case_count == 2
    assert summary.task_success_count == 1
    assert summary.task_success_rate == 0.5


# --------------------------------------------------------------------------- #
# Answer-key integrity
# --------------------------------------------------------------------------- #


def test_an_out_of_domain_case_may_not_expect_a_tool() -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate(
            {
                "id": "bad-1",
                "category": "ood",
                "query": "q",
                "expected_intent": None,
                "expected_tools": [DEVICE_TOOL],
            }
        )


def test_an_out_of_domain_case_may_not_expect_an_intent() -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate(
            {
                "id": "bad-2",
                "category": "ood",
                "query": "q",
                "expected_intent": "device_status",
                "expected_tools": [],
            }
        )


def test_an_in_scope_case_must_declare_an_intent() -> None:
    with pytest.raises(ValidationError):
        make_case(expected_intent=None)


def test_expected_arguments_must_name_a_tool_the_case_expects() -> None:
    with pytest.raises(ValidationError):
        make_case(
            expected_tools=[DEVICE_TOOL],
            expected_arguments={ALARM_TOOL: {"alarm_code": "E1234"}},
        )


def test_an_unknown_intent_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_case(expected_intent="psychic_diagnosis")


def test_duplicate_case_ids_are_rejected() -> None:
    payload = make_mini_payload()
    payload["cases"][1]["id"] = "m-1"
    with pytest.raises(ValidationError):
        parse_dataset(payload)


def test_a_case_in_an_undocumented_category_is_rejected() -> None:
    payload = make_mini_payload()
    payload["cases"][0]["category"] = "brand_new"
    with pytest.raises(ValidationError):
        parse_dataset(payload)


def test_the_shipped_dataset_loads_and_stays_within_its_declared_size() -> None:
    dataset = load_dataset()

    # Phase 1 asks for 40 to 50 cases spread over the documented categories.
    assert 40 <= len(dataset.cases) <= 50
    assert set(dataset.category_counts) == set(dataset.categories)
    assert all(count >= 5 for count in dataset.category_counts.values())
    assert dataset.policy


def test_the_dataset_declares_its_ground_truth_policy_before_any_planner_runs() -> None:
    policy = load_dataset().policy

    # The policy keys are the answer-key rules, fixed in advance of any run. A
    # missing rule would mean a category was scored without a stated definition.
    assert {"P1", "P2", "P3", "P4", "P5"} <= {key[:2] for key in policy}


def test_at_least_five_out_of_domain_cases_exist() -> None:
    assert len(load_dataset().ood_cases) >= 5


def test_the_dataset_hash_is_stable_and_report_ready() -> None:
    digest = dataset_sha256(DEFAULT_DATASET_PATH)

    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    assert digest == dataset_sha256(DEFAULT_DATASET_PATH)


def test_the_intent_vocabulary_comes_from_the_parser() -> None:
    assert INTENT_VOCABULARY == {
        "alarm_diagnosis",
        "maintenance_advice",
        "device_status",
        "unknown",
    }


# --------------------------------------------------------------------------- #
# Runner helpers and the CLI surface
# --------------------------------------------------------------------------- #


def test_a_provider_endpoint_is_reported_without_its_credential() -> None:
    assert (
        runner_module._safe_base_url("https://user:secret@api.example.com/v1")
        == "https://api.example.com/v1"
    )
    assert (
        runner_module._safe_base_url("https://api.example.com/v1") == "https://api.example.com/v1"
    )


def test_a_model_plan_supplies_its_own_arguments() -> None:
    state: Any = {
        "query": "PLC-001 现在什么状态",
        "planned_tool_calls": [{"tool": DEVICE_TOOL, "arguments": {"device_id": "PLC-001"}}],
    }
    tools, arguments = runner_module._read_plan(state)

    assert tools == [DEVICE_TOOL]
    assert arguments == {DEVICE_TOOL: {"device_id": "PLC-001"}}


def test_rule_arguments_are_derived_from_the_executor_table_not_a_copy() -> None:
    state: Any = {
        "query": "PLC-001 现在什么状态",
        "required_tools": [DEVICE_TOOL],
        "equipment_id": "PLC-001",
    }
    tools, arguments = runner_module._read_plan(state)

    # Same adapter the executor uses, so argument accuracy cannot drift from what
    # the rule planner's plan actually becomes at call time.
    assert tools == [DEVICE_TOOL]
    assert arguments[DEVICE_TOOL] == {"device_id": "PLC-001"}


def test_the_cli_defaults_to_the_frozen_rule_baseline() -> None:
    args = build_parser().parse_args([])

    assert args.planner == "rule"
    assert args.execute_tools is False
    assert args.tag is None
    assert Path(args.dataset) == DEFAULT_DATASET_PATH


def test_the_cli_accepts_every_documented_planner() -> None:
    parser = build_parser()

    for choice in ("rule", "llm", "auto"):
        assert parser.parse_args(["--planner", choice]).planner == choice


def test_the_cli_rejects_an_unknown_planner() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--planner", "magic"])


def test_a_gate_report_carries_no_metrics_by_construction() -> None:
    gate = GateReport(
        schema_version="1.0",
        status=STATUS_LLM_NOT_RUN,
        planner_mode="llm",
        generated_at="2026-01-01T00:00:00+00:00",
        dataset_path="evaluation/dataset.json",
        dataset_sha256="0" * 64,
        dataset_case_count=1,
        reason="no provider",
    )
    assert gate.metrics is None

    # The field is typed as None, so a gate report that somehow carried a metric
    # would not even validate.
    with pytest.raises(ValidationError):
        GateReport.model_validate({**gate.model_dump(), "metrics": {"intent_accuracy": 1.0}})


@pytest.mark.parametrize("planner", ["llm", "auto"])
def test_an_llm_run_without_a_provider_reports_not_run_and_writes_no_metrics(
    planner: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Forced, so the result does not depend on whether this machine happens to
    # hold a provider configuration.
    monkeypatch.setattr(runner_module, "is_llm_configured", lambda settings: False)

    exit_code = main(
        [
            "--planner",
            planner,
            "--dataset",
            str(DEFAULT_DATASET_PATH),
            "--output-dir",
            str(tmp_path),
            "--tag",
            "gate",
        ]
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "gate_evaluation_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == STATUS_LLM_NOT_RUN
    assert payload["planner_mode"] == planner
    assert payload["metrics"] is None
    assert payload["dataset_case_count"] == len(load_dataset().cases)
    assert "LLM_API_KEY" in payload["required_configuration"]
    # No metric file exists, so no rule run can be mistaken for an LLM run.
    assert not (tmp_path / "gate_baseline.json").exists()
    assert not (tmp_path / "gate_failures.json").exists()


def test_the_cli_runs_a_supplied_dataset_and_names_it_in_the_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "mini.json"
    dataset_path.write_text(json.dumps(make_mini_payload()), encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--planner",
            "rule",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--tag",
            "mini",
        ]
    )

    assert exit_code == 0
    report = EvaluationReport.model_validate(
        json.loads((output_dir / "mini_baseline.json").read_text(encoding="utf-8"))
    )
    assert report.status == STATUS_OK
    assert report.denominators.total_cases == 2
    assert report.run.dataset_case_count == 2
    assert report.run.dataset_category_counts == {"device_status": 1, "ood": 1}
    assert report.run.dataset_sha256 == dataset_sha256(dataset_path)
    assert Path(report.run.dataset_path) == dataset_path.resolve()


# --------------------------------------------------------------------------- #
# The real baseline: one planner-only run, shared by the checks below
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_rule_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[EvaluationReport, FailureReport]:
    """Run the frozen rule planner over the shipped dataset, planner-only."""
    output_dir = tmp_path_factory.mktemp("rule_baseline")
    exit_code = main(["--planner", "rule", "--output-dir", str(output_dir), "--tag", "t"])
    assert exit_code == 0
    report = EvaluationReport.model_validate(
        json.loads((output_dir / "t_baseline.json").read_text(encoding="utf-8"))
    )
    failures = FailureReport.model_validate(
        json.loads((output_dir / "t_failures.json").read_text(encoding="utf-8"))
    )
    return report, failures


def test_the_real_run_reports_all_ten_metrics_and_their_definitions(
    real_rule_run: tuple[EvaluationReport, FailureReport],
) -> None:
    report, _ = real_rule_run

    assert report.status == STATUS_OK
    assert report.schema_version == "1.0"
    assert REQUIRED_METRICS <= set(report.metrics.model_dump())
    assert REQUIRED_METRICS <= set(report.definitions)
    assert set(report.definitions) == set(METRIC_DEFINITIONS)
    # Provenance: the report can always be traced back to its exact input.
    assert report.run.planner_mode == "rule"
    assert report.run.run_mode == "planner_only"
    assert report.run.execute_tools is False
    assert len(report.run.dataset_sha256) == 64
    assert report.run.dataset_case_count == len(load_dataset().cases)
    assert report.run.known_tools == [DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL]


def test_the_real_run_is_planner_only_so_execution_is_not_scored(
    real_rule_run: tuple[EvaluationReport, FailureReport],
) -> None:
    report, _ = real_rule_run

    assert report.metrics.execution_error_rate is None
    assert report.denominators.executed_cases == 0
    assert report.latency_ms["execution_latency_ms"].sampled_cases == 0
    assert report.latency_ms["rag_latency_ms"].sampled_cases == 0
    assert report.latency_ms["planning_latency_ms"].sampled_cases == report.denominators.total_cases
    assert any("planner-only" in note for note in report.limitations)


def test_the_real_run_stays_recomputable_and_free_of_fabricated_values(
    real_rule_run: tuple[EvaluationReport, FailureReport],
) -> None:
    report, _ = real_rule_run
    metrics = report.metrics
    counts = report.denominators

    assert metrics.intent_accuracy == pytest.approx(
        counts.intent_correct_cases / counts.intent_evaluated_cases
    )
    assert metrics.tool_selection_exact_match == pytest.approx(
        counts.tool_exact_match_cases / counts.total_cases
    )
    assert metrics.task_success_rate == pytest.approx(
        counts.task_success_cases / counts.total_cases
    )
    assert metrics.planner_failure_rate == pytest.approx(
        counts.planner_failure_cases / counts.total_cases
    )
    assert metrics.tool_recall == pytest.approx(
        counts.tool_true_positive / (counts.tool_true_positive + counts.tool_false_negative)
    )
    # Two specifics that a fabricated run tends to get wrong: the rule planner
    # never fails to plan and never invents a tool name.
    assert metrics.planner_failure_rate == 0.0
    assert metrics.invalid_tool_rate == 0.0
    assert metrics.argument_accuracy == 1.0


def test_the_real_run_keeps_its_known_failures_visible(
    real_rule_run: tuple[EvaluationReport, FailureReport],
) -> None:
    report, failures = real_rule_run

    assert failures.case_count == report.denominators.total_cases
    assert failures.dataset_sha256 == report.run.dataset_sha256
    assert failures.failure_count == len(failures.failures)

    case_ids = {case.id for case in load_dataset().cases}
    for record in failures.failures:
        assert record.case_id in case_ids
        assert record.failure_type in FAILURE_PRIORITY
        assert record.failure_type in record.failure_types

    # The keyword-driven rule planner is known to reach for the manual search on
    # queries that a structured tool already answers, and on out-of-domain
    # keyword bait. Those cases are pinned by identifier: making them pass means
    # the answer key moved, not that the planner improved.
    reported = {record.case_id for record in failures.failures}
    assert {
        "ad-002",
        "ad-004",
        "ad-006",
        "ad-008",
        "ro-003",
        "mt-007",
        "am-004",
        "ood-006",
        "ood-007",
        "ood-008",
    } <= reported

    # An out-of-domain query reaching for an industrial tool is the failure the
    # safety block exists to expose.
    assert report.ood_safety.offending_case_ids == ["ood-006", "ood-007", "ood-008"]
    assert report.ood_safety.tool_call_case_count == 3
    assert report.ood_safety.silent_case_count == report.denominators.ood_cases - 3
    assert report.ood_safety.tool_call_rate == pytest.approx(3 / report.denominators.ood_cases)


def test_the_real_run_reports_category_breakdowns_without_hiding_a_weak_category(
    real_rule_run: tuple[EvaluationReport, FailureReport],
) -> None:
    report, _ = real_rule_run
    dataset = load_dataset()

    assert set(report.categories) == set(dataset.categories)
    for category, summary in report.categories.items():
        assert summary.case_count == dataset.category_counts[category]
        assert summary.task_success_rate == pytest.approx(
            summary.task_success_count / summary.case_count
        )
    # A category that the rule planner cannot handle must not be able to vanish.
    ood_success_rate = report.categories["ood"].task_success_rate
    assert ood_success_rate is not None
    assert ood_success_rate < 1.0

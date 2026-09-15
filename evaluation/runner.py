"""Evaluation CLI: run the dataset through a real planner and report the results.

Usage::

    python -m evaluation.runner --planner rule --dataset evaluation/dataset.json
    python -m evaluation.runner --planner rule --execute-tools
    python -m evaluation.runner --planner llm

Two run modes exist because they answer different questions and cost different
amounts of time:

* **planner-only** (default) runs the real parser and the real planner for every
  case and scores the plan. It touches no tool, so the roughly five-second manual
  retrieval never runs and the whole dataset costs a fraction of a second. This is
  the mode to use when comparing planners.
* **end-to-end** (``--execute-tools``) continues into the executor, the evidence
  step and the synthesis step, and additionally records execution and retrieval
  latency.

Honesty rules enforced here, not merely intended:

* the rule planner is never stubbed; the planner under test is the real
  :func:`app.agent.planners.dispatch.dispatch_plan` with the real baseline;
* when an LLM planner is requested but no provider is configured, the run reports
  ``LLM_EVALUATION_NOT_RUN`` and writes no metrics at all. A stub or a fabricated
  score would be worse than no number;
* a latency that cannot be measured accurately is ``null``, never an estimate.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from app.agent.graph import (
    build_tool_arguments,
    execute_tools,
    plan_actions,
    retrieve_context,
    route_query,
    synthesize,
)
from app.agent.planners import PlannerError, build_llm_planner, dispatch_plan
from app.agent.state import MaintenanceState
from app.config import Settings
from app.integrations.llm import is_llm_configured
from app.tools.registry import registry
from evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    PROJECT_ROOT,
    EvalCase,
    EvalDataset,
    dataset_sha256,
    load_dataset,
)
from evaluation.evaluator import (
    METRIC_DEFINITIONS,
    CaseOutcome,
    CategorySummary,
    CountSet,
    LatencySummary,
    MetricSet,
    OODSummary,
    PlanObservation,
    aggregate,
    evaluate_case,
)

SCHEMA_VERSION = "1.0"
STATUS_OK = "OK"
STATUS_LLM_NOT_RUN = "LLM_EVALUATION_NOT_RUN"

#: Configuration an LLM run needs. Named, never valued, in the gate report.
REQUIRED_LLM_CONFIGURATION = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


class PlannerChoice(str, Enum):
    """The planners the CLI can drive."""

    RULE = "rule"
    LLM = "llm"
    AUTO = "auto"


class RunMode(str, Enum):
    """How far down the pipeline a run goes."""

    PLANNER_ONLY = "planner_only"
    END_TO_END = "end_to_end"


# --------------------------------------------------------------------------- #
# Report models
# --------------------------------------------------------------------------- #


class RunMetadata(BaseModel):
    """Provenance for a run, so a number can always be traced to its input."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    dataset_path: str
    dataset_sha256: str
    dataset_case_count: int
    dataset_category_counts: dict[str, int] = Field(default_factory=dict)
    planner_mode: str
    run_mode: str
    execute_tools: bool
    known_tools: list[str] = Field(default_factory=list)
    app_version: str
    python_version: str
    git_commit: str | None = None
    git_worktree_dirty: bool | None = None
    database_url: str | None = None
    rag_provider_available: bool | None = None
    rag_provider_id: str | None = None


class PlannerRuntime(BaseModel):
    """Which planner actually ran, and how it behaved."""

    model_config = ConfigDict(extra="forbid")

    planner_used_counts: dict[str, int] = Field(default_factory=dict)
    fallback_case_count: int = 0
    fallback_reason_counts: dict[str, int] = Field(default_factory=dict)
    planning_error_counts: dict[str, int] = Field(default_factory=dict)
    provider: dict[str, str] | None = Field(
        default=None,
        description="Provider metadata with the credential removed. Null for rule mode.",
    )


class EvaluationReport(BaseModel):
    """The baseline report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    status: str
    metrics: MetricSet
    denominators: CountSet
    latency_ms: dict[str, LatencySummary] = Field(default_factory=dict)
    categories: dict[str, CategorySummary] = Field(default_factory=dict)
    ood_safety: OODSummary
    planner_runtime: PlannerRuntime
    run: RunMetadata
    definitions: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class FailureRecord(BaseModel):
    """One failing case, in the shape the analysis workflow expects."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    query: str
    category: str
    failure_type: str
    failure_types: list[str] = Field(default_factory=list)
    expected: dict[str, Any]
    actual: dict[str, Any]
    notes: str | None = None


class FailureReport(BaseModel):
    """Every failing case plus a tally, so patterns are visible at a glance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    status: str
    planner_mode: str
    run_mode: str
    dataset_sha256: str
    case_count: int
    failure_count: int
    failure_type_counts: dict[str, int] = Field(default_factory=dict)
    failures: list[FailureRecord] = Field(default_factory=list)


class GateReport(BaseModel):
    """The not-run report. It carries no metrics, by construction."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    status: str
    planner_mode: str
    generated_at: str
    dataset_path: str
    dataset_sha256: str
    dataset_case_count: int
    reason: str
    required_configuration: list[str] = Field(default_factory=list)
    metrics: None = Field(
        default=None,
        description="Always null. A gate report states that nothing was measured.",
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _merge(state: MaintenanceState, update: dict[str, Any]) -> MaintenanceState:
    """Return ``state`` with ``update`` applied.

    A small helper rather than an in-place ``update`` so the merge is typed as a
    :class:`MaintenanceState` and the node calls stay type-checkable.
    """
    merged: dict[str, Any] = dict(state)
    merged.update(update)
    return cast(MaintenanceState, merged)


def _safe_base_url(value: str) -> str:
    """Return ``value`` with any embedded credentials stripped.

    A base URL can carry ``user:password@`` userinfo. Reporting the provider's
    endpoint is useful; reporting its credential is not, so the userinfo is
    dropped before the URL reaches a report or a log.
    """
    parts = urlsplit(value)
    if "@" not in parts.netloc:
        return value
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def _git_metadata() -> tuple[str | None, bool | None]:
    """Return ``(commit, worktree_dirty)``, or ``(None, None)`` outside a repo."""

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    try:
        head = _git("rev-parse", "HEAD")
        if head.returncode != 0:
            return None, None
        status = _git("status", "--porcelain")
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return head.stdout.strip(), dirty
    except OSError:
        return None, None


def _rag_metadata(settings: Settings) -> tuple[bool, str | None]:
    """Return ``(available, provider_id)`` for the configured retrieval backend.

    Availability is checked by actually building the provider, which is the same
    thing the tool does at call time.
    """
    from app.integrations.rag import factory as rag_factory
    from app.integrations.rag.base import RAGProviderError

    try:
        provider = rag_factory.get_rag_provider()
    except RAGProviderError:
        return False, None
    except Exception:
        return False, None
    return True, provider.provider_id


class _TimingRAGProvider:
    """Wrap a retrieval provider to record how long each search took.

    The evaluation harness owns this measurement. The application is not
    modified, and the wrapper only delegates. ``provider_id`` is copied from the
    wrapped provider so a report never misstates which backend answered.
    """

    def __init__(self, inner: Any, sink: list[float]) -> None:
        self._inner = inner
        self._sink = sink
        self.provider_id = getattr(inner, "provider_id", "unknown")

    def search(self, query: str, top_k: int = 4) -> Any:
        """Delegate the search and record its wall time."""
        started = perf_counter()
        try:
            return self._inner.search(query, top_k=top_k)
        finally:
            self._sink.append(round((perf_counter() - started) * 1000, 3))


def _install_rag_timer(sink: list[float]) -> str | None:
    """Wrap the configured retrieval provider and return its identifier.

    Returns ``None`` when no provider can be built, in which case the factory is
    left untouched and the tool reports the unavailability itself. Retrieval
    latency is then ``null`` for every case rather than an invented number.
    """
    from app.integrations.rag import factory as rag_factory
    from app.integrations.rag.base import RAGProviderError

    try:
        inner = rag_factory.get_rag_provider()
    except RAGProviderError:
        return None
    except Exception:
        return None

    wrapper = _TimingRAGProvider(inner, sink)
    rag_factory.get_rag_provider = lambda: wrapper  # type: ignore[assignment]
    return str(wrapper.provider_id)


def _read_plan(state: MaintenanceState) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Return the planned tools and the arguments that belong to them.

    The LLM planner states its arguments in the plan. The rule planner does not:
    it emits tool names and the executor derives the arguments from state. Both
    are read here so argument accuracy is scored the same way in either mode, and
    the rule planner's arguments come from the executor's own adapter table rather
    than a copy.
    """
    planned = state.get("planned_tool_calls") or []
    if planned:
        tools: list[str] = []
        arguments: dict[str, dict[str, Any]] = {}
        for call in planned:
            name = call.get("tool")
            if not isinstance(name, str) or not name:
                continue
            tools.append(name)
            raw = call.get("arguments")
            arguments[name] = dict(raw) if isinstance(raw, dict) else {}
        return tools, arguments

    tools = [str(name) for name in (state.get("required_tools") or [])]
    arguments = {}
    for name in tools:
        derived = build_tool_arguments(name, state) or {}
        arguments[name] = {
            key: value for key, value in derived.items() if value is not None and value != ""
        }
    return tools, arguments


def _execution_errors(state: MaintenanceState) -> list[str]:
    """Return the execution problems a run reported."""
    errors: list[str] = []
    for entry in state.get("tool_results") or []:
        payload = entry.get("result")
        if isinstance(payload, dict) and payload.get("error"):
            errors.append(f"{entry.get('tool')}: {payload['error']}")
    pipeline_error = state.get("error")
    if pipeline_error:
        errors.append(str(pipeline_error))
    return errors


def run_case(
    case_query: str,
    *,
    settings: Settings,
    llm_planner: Any,
    execute: bool,
    rag_sink: list[float],
) -> PlanObservation:
    """Run one case through the real pipeline and observe what happened."""
    state = MaintenanceState(query=case_query)

    started = perf_counter()
    try:
        state = _merge(state, route_query(state))
        state = _merge(
            state,
            dispatch_plan(
                state,
                rule_planner=plan_actions,
                settings=settings,
                planner=llm_planner,
            ),
        )
    except PlannerError as exc:
        return PlanObservation(
            planning_error=exc.code_value,
            planning_latency_ms=round((perf_counter() - started) * 1000, 3),
        )
    except Exception as exc:
        return PlanObservation(
            planning_error=f"UNEXPECTED_{type(exc).__name__}",
            planning_latency_ms=round((perf_counter() - started) * 1000, 3),
        )

    planning_ms = round((perf_counter() - started) * 1000, 3)
    tools, arguments = _read_plan(state)

    observation = PlanObservation(
        intent=state.get("intent"),
        tools=tools,
        arguments=arguments,
        planner_used=state.get("planner_used"),
        planner_fallback=bool(state.get("planner_fallback") or False),
        planner_fallback_reason=state.get("planner_fallback_reason"),
        planning_latency_ms=planning_ms,
    )

    if not execute:
        # Nothing beyond planning ran, so the total of the pipeline that ran is
        # the planning step itself. That is a measurement, not an estimate of
        # what a full run would have cost.
        observation.total_latency_ms = planning_ms
        return observation

    executed: MaintenanceState = state
    rag_before = len(rag_sink)
    exec_started = perf_counter()
    executed = _merge(executed, execute_tools(executed))
    observation.execution_latency_ms = round((perf_counter() - exec_started) * 1000, 3)
    executed = _merge(executed, retrieve_context(executed))
    executed = _merge(executed, synthesize(executed))

    retrieval_samples = rag_sink[rag_before:]
    observation.rag_latency_ms = round(sum(retrieval_samples), 3) if retrieval_samples else None
    observation.execution_errors = _execution_errors(executed)
    observation.total_latency_ms = round((perf_counter() - started) * 1000, 3)
    return observation


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


def _failure_record(
    outcome: CaseOutcome,
    observation: PlanObservation,
    case: EvalCase,
    *,
    notes: str | None,
) -> FailureRecord:
    """Build one failure record, carrying both the answer key and what happened."""
    return FailureRecord(
        case_id=outcome.case_id,
        query=outcome.query,
        category=outcome.category,
        failure_type=outcome.primary_failure_type or "unknown",
        failure_types=list(outcome.failure_types),
        expected={
            "intent": outcome.expected_intent,
            "tools": outcome.expected_tools,
            "arguments": case.arguments,
        },
        actual={
            "intent": outcome.actual_intent,
            "tools": observation.tools,
            "arguments": observation.arguments,
            "planner_used": outcome.planner_used,
            "planner_fallback": outcome.planner_fallback,
            "planner_fallback_reason": outcome.planner_fallback_reason,
            "planning_error": outcome.planning_error,
            "execution_errors": list(outcome.execution_errors),
        },
        notes=notes,
    )


def build_reports(
    *,
    dataset: EvalDataset,
    dataset_path: Path,
    outcomes: list[CaseOutcome],
    observations: dict[str, PlanObservation],
    settings: Settings,
    planner: PlannerChoice,
    execute: bool,
    provider_metadata: dict[str, str] | None,
    rag_available: bool,
) -> tuple[EvaluationReport, FailureReport]:
    """Assemble the baseline and failure reports."""
    executed = execute
    metrics, counts, latency, categories, ood = aggregate(outcomes, executed=executed)

    planner_used = Counter(item.planner_used or "none" for item in outcomes)
    fallback_reasons = Counter(
        (item.planner_fallback_reason or "").split(":")[0]
        for item in outcomes
        if item.planner_fallback and item.planner_fallback_reason
    )
    error_codes = Counter(item.planning_error for item in outcomes if item.planning_error)

    commit, dirty = _git_metadata()
    rag_available_now, rag_id = _rag_metadata(settings)

    run = RunMetadata(
        generated_at=_utc_now(),
        dataset_path=_portable_dataset_path(dataset_path),
        dataset_sha256=dataset_sha256(dataset_path),
        dataset_case_count=len(dataset.cases),
        dataset_category_counts=dataset.category_counts,
        planner_mode=planner.value,
        run_mode=RunMode.END_TO_END.value if execute else RunMode.PLANNER_ONLY.value,
        execute_tools=execute,
        known_tools=registry.names(),
        app_version=settings.app_version,
        python_version=platform.python_version(),
        git_commit=commit,
        git_worktree_dirty=dirty,
        database_url=settings.database_url,
        rag_provider_available=rag_available and rag_available_now,
        rag_provider_id=rag_id,
    )

    report = EvaluationReport(
        schema_version=SCHEMA_VERSION,
        status=STATUS_OK,
        metrics=metrics,
        denominators=counts,
        latency_ms=latency,
        categories=categories,
        ood_safety=ood,
        planner_runtime=PlannerRuntime(
            planner_used_counts=dict(planner_used),
            fallback_case_count=sum(1 for item in outcomes if item.planner_fallback),
            fallback_reason_counts=dict(fallback_reasons),
            planning_error_counts=dict(error_codes),
            provider=provider_metadata,
        ),
        run=run,
        definitions=METRIC_DEFINITIONS,
        limitations=_limitations(execute=execute, rag_available=rag_available_now),
    )

    failures = [
        _failure_record(
            item,
            observations[item.case_id],
            dataset.case_by_id(item.case_id),
            notes=dataset.case_by_id(item.case_id).notes,
        )
        for item in outcomes
        if item.failure_types
    ]
    type_counts = Counter(failure_type for item in outcomes for failure_type in item.failure_types)
    failure_report = FailureReport(
        schema_version=SCHEMA_VERSION,
        status=STATUS_OK,
        planner_mode=planner.value,
        run_mode=run.run_mode,
        dataset_sha256=run.dataset_sha256,
        case_count=counts.total_cases,
        failure_count=len(failures),
        failure_type_counts=dict(sorted(type_counts.items())),
        failures=failures,
    )
    return report, failure_report


def _limitations(*, execute: bool, rag_available: bool) -> list[str]:
    """Return the caveats that a reader needs to interpret the numbers."""
    notes = [
        "Task success is judged on the plan, not on execution. Execution outcomes "
        "are reported through execution_error_rate and the failure records.",
        "Out-of-domain cases are excluded from intent accuracy, because there is no "
        "maintenance intent to classify. Their signal is ood_safety.",
        "Retrieval latency covers only the retrieval call, measured by a timing "
        "wrapper installed by this harness. A series with no samples is null.",
        "Argument accuracy is conditioned on the tool being selected; a tool that "
        "was expected but not selected lowers tool recall instead.",
        "Every metric whose denominator is zero is reported as null, never as zero "
        "or one. See denominators for the underlying counts.",
    ]
    if not execute:
        notes.append(
            "This is a planner-only run. No tool ran, so execution and retrieval "
            "latency are null and execution_error_rate is not applicable."
        )
    if not rag_available:
        notes.append(
            "No retrieval provider could be built in this environment, so the "
            "manual search tool reports unavailability and retrieval latency stays "
            "null even in an end-to-end run."
        )
    return notes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the evaluation CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.runner",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--planner",
        choices=[choice.value for choice in PlannerChoice],
        default=PlannerChoice.RULE.value,
        help="Planner under test. rule is the frozen baseline.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the evaluation dataset JSON.",
    )
    parser.add_argument(
        "--execute-tools",
        action="store_true",
        help="Continue past planning into execution, evidence and synthesis.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "reports"),
        help="Directory for the generated reports.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Report filename prefix. Defaults to the planner name.",
    )
    return parser


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _portable_dataset_path(path: Path) -> str:
    """Record the dataset path so a published report carries no machine layout.

    A dataset inside this repository is stored relative to the repository root,
    which keeps the record portable and free of the author's absolute paths. A
    dataset outside the repository is stored resolved, because no
    repository-relative form exists for it. The dataset SHA-256, not this
    string, is what pins the exact input behind a number.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _write(path: Path, payload: BaseModel) -> None:
    """Write a report model as UTF-8 JSON, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir)
    tag = args.tag or args.planner

    dataset = load_dataset(dataset_path)
    planner = PlannerChoice(args.planner)
    settings = Settings(planner_mode=planner.value)

    # The LLM gate. An LLM-planner run without a configured provider would either
    # fail or silently become a rule run, and a rule run reported under an LLM
    # label would be a fabricated result. So it stops here.
    if planner is not PlannerChoice.RULE and not is_llm_configured(settings):
        gate = GateReport(
            schema_version=SCHEMA_VERSION,
            status=STATUS_LLM_NOT_RUN,
            planner_mode=planner.value,
            generated_at=_utc_now(),
            dataset_path=str(dataset_path),
            dataset_sha256=dataset_sha256(dataset_path),
            dataset_case_count=len(dataset.cases),
            reason=(
                "no LLM provider is configured, so the LLM planner cannot run. "
                "The rule planner is the only measured baseline in this environment."
            ),
            required_configuration=list(REQUIRED_LLM_CONFIGURATION),
        )
        _write(output_dir / f"{tag}_evaluation_status.json", gate)
        print(json.dumps(gate.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    llm_planner = None
    provider_metadata: dict[str, str] | None = None
    if planner is not PlannerChoice.RULE:
        try:
            llm_planner = build_llm_planner(settings)
        except PlannerError as exc:
            gate = GateReport(
                schema_version=SCHEMA_VERSION,
                status=STATUS_LLM_NOT_RUN,
                planner_mode=planner.value,
                generated_at=_utc_now(),
                dataset_path=_portable_dataset_path(dataset_path),
                dataset_sha256=dataset_sha256(dataset_path),
                dataset_case_count=len(dataset.cases),
                reason=f"the LLM planner could not be built: {exc.code_value}",
                required_configuration=list(REQUIRED_LLM_CONFIGURATION),
            )
            _write(output_dir / f"{tag}_evaluation_status.json", gate)
            print(json.dumps(gate.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 1
        described = dict(llm_planner.provider.describe())
        if "base_url" in described:
            described["base_url"] = _safe_base_url(described["base_url"])
        provider_metadata = described

    rag_sink: list[float] = []
    rag_available = _rag_metadata(settings)[0]
    if args.execute_tools:
        rag_available = _install_rag_timer(rag_sink) is not None

    known_tools = set(registry.names())
    outcomes: list[CaseOutcome] = []
    observations: dict[str, PlanObservation] = {}

    for case in dataset.cases:
        observation = run_case(
            case.query,
            settings=settings,
            llm_planner=llm_planner,
            execute=args.execute_tools,
            rag_sink=rag_sink,
        )
        observations[case.id] = observation
        outcomes.append(evaluate_case(case, observation, known_tools=known_tools))

    report, failure_report = build_reports(
        dataset=dataset,
        dataset_path=dataset_path,
        outcomes=outcomes,
        observations=observations,
        settings=settings,
        planner=planner,
        execute=args.execute_tools,
        provider_metadata=provider_metadata,
        rag_available=rag_available,
    )

    _write(output_dir / f"{tag}_baseline.json", report)
    _write(output_dir / f"{tag}_failures.json", failure_report)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

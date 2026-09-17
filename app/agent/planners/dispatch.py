"""Planner selection: ``rule``, ``llm`` or ``auto``.

The frozen deterministic planner is passed in as a callable rather than imported.
That keeps this module independent of the graph, so there is no import cycle and
the baseline planner has one owner: ``app.agent.graph``.

Mode semantics:

* ``rule`` - the frozen baseline, always. This is the default and its behaviour is
  unchanged from V0.4.
* ``llm`` - the LLM planner, always. A failure is raised as a ``PlannerError`` and
  reaches the caller as a structured error. There is no fallback: a silently
  downgraded plan is indistinguishable from a correct one.
* ``auto`` - the LLM planner first. If it fails, the frozen rule planner runs and
  the failure is recorded in ``planner_fallback_reason``.

Every path records ``planner_used``, ``planner_fallback`` and, when a fallback
happened, ``planner_fallback_reason``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any

from app.agent.planners.llm import LLMPlanner, PlannerResult, build_llm_planner
from app.agent.planners.schema import PlannerError, PlannerErrorCode
from app.agent.state import MaintenanceState
from app.config import Settings, get_settings
from app.observability import instrumentation

logger = logging.getLogger(__name__)

#: Values for ``planner_used``.
PLANNER_RULE = "rule"
PLANNER_LLM = "llm"

#: Recorded when the dispatcher itself fails. Defensive only: it is outside the
#: five planner codes because mislabelling an internal fault as a provider fault
#: would be worse than reporting it as unclassified.
UNEXPECTED_PLANNER_ERROR = "UNEXPECTED_PLANNER_ERROR"

#: A planner that turns the agent state into a partial state update.
RulePlanner = Callable[[MaintenanceState], dict[str, Any]]

#: Planner error code to the closed reason vocabulary reported by
#: ``industrial_agent_planner_failures_total``. The mapping is a dict rather than
#: a chain of comparisons so an unmapped code is visibly unmapped.
_FAILURE_REASONS: dict[PlannerErrorCode, str] = {
    PlannerErrorCode.LLM_PROVIDER_ERROR: "provider_error",
    PlannerErrorCode.LLM_TIMEOUT: "timeout",
    # A plan the planner refused is one outcome for the metric even though it has
    # three distinct codes: the operator response is the same, and the code
    # remains available in the log line and on the span.
    PlannerErrorCode.INVALID_PLANNER_OUTPUT: "invalid_output",
    PlannerErrorCode.UNKNOWN_TOOL: "invalid_output",
    PlannerErrorCode.TOOL_ARGUMENT_VALIDATION_FAILED: "invalid_output",
}


def _failure_reason(exc: PlannerError, phase: str) -> str:
    """Classify a planning failure for the metric label.

    ``phase`` is ``"build"`` when the planner could not be constructed at all,
    which is a configuration problem, and ``"plan"`` once the planner was
    running. The distinction is structural rather than inferred from the message:
    ``build_llm_planner`` reports a missing endpoint or key as
    ``LLM_PROVIDER_ERROR``, the same code a network failure uses, so the code
    alone cannot tell a misconfiguration from an outage.
    """
    if phase == "build":
        return "configuration"
    return _FAILURE_REASONS.get(exc.code, "unknown")


def _rule_update(update: dict[str, Any], *, fallback_reason: str | None = None) -> dict[str, Any]:
    """Decorate the frozen planner's update with the planner bookkeeping."""
    merged = dict(update)
    merged["planner_used"] = PLANNER_RULE
    merged["planner_fallback"] = fallback_reason is not None
    merged["planner_fallback_reason"] = fallback_reason
    # No explicit call list: the executor derives arguments from the state, which
    # is the frozen rule behaviour.
    merged["planned_tool_calls"] = []
    return merged


def _llm_update(result: PlannerResult) -> dict[str, Any]:
    """Translate a validated plan into a partial state update.

    Only tool calls, their arguments and the plan's intent travel onwards. The
    prompt and the raw completion are not in the result, so they cannot reach the
    state, a response or a log.
    """
    plan = result.plan
    update: dict[str, Any] = {
        "required_tools": plan.tool_names,
        "planned_tool_calls": [
            {"tool": call.tool_name, "arguments": call.arguments, "reason": call.reason}
            for call in plan.tool_calls
        ],
        "planner_used": PLANNER_LLM,
        "planner_fallback": False,
        "planner_fallback_reason": None,
    }
    if plan.intent is not None:
        update["intent"] = plan.intent.value
    return update


def dispatch_plan(
    state: MaintenanceState,
    *,
    rule_planner: RulePlanner,
    settings: Settings | None = None,
    planner: LLMPlanner | None = None,
) -> dict[str, Any]:
    """Plan with the configured planner.

    Args:
        state: Current agent state.
        rule_planner: The frozen baseline planner, injected by the graph.
        settings: Override the process settings, mainly for tests.
        planner: Override the LLM planner, mainly for tests.

    Raises:
        PlannerError: In ``llm`` mode when planning fails. The error code is part
            of the V0.5 taxonomy.
    """
    resolved = settings or get_settings()
    mode = resolved.planner_mode

    if mode == "rule":
        return _run_rule_planner(rule_planner, state)

    query = state.get("query") or ""
    started = perf_counter()
    instrumentation.planner_started(planner=PLANNER_LLM)
    # ``phase`` records where the failure happened, which is what separates a
    # misconfiguration from a provider fault. It is set between the two statements
    # rather than inferred from the exception.
    phase = "build"

    with instrumentation.planner_span(planner=PLANNER_LLM) as active_span:
        try:
            active = planner if planner is not None else build_llm_planner(resolved)
            phase = "plan"
            result = active.plan(query)
        except PlannerError as exc:
            reason = _failure_reason(exc, phase)
            instrumentation.planner_failed(
                planner=PLANNER_LLM,
                reason=reason,
                duration_ms=instrumentation.elapsed_ms(started),
            )
            active_span.record_error(reason, exc.code_value)
            if mode == "llm":
                logger.error("planner_failed mode=llm code=%s", exc.code_value)
                raise
            update = _fallback(rule_planner, state, query, f"{exc.code_value}: {exc.message}")
            active_span.set_attribute("status", instrumentation.STATUS_UNAVAILABLE)
            return update
        except Exception as exc:  # defensive: an unexpected fault must not fail auto mode
            instrumentation.planner_failed(
                planner=PLANNER_LLM,
                reason="unknown",
                duration_ms=instrumentation.elapsed_ms(started),
            )
            active_span.record_error(type(exc).__name__)
            if mode == "llm":
                raise
            update = _fallback(
                rule_planner,
                state,
                query,
                f"{UNEXPECTED_PLANNER_ERROR}: {type(exc).__name__}",
            )
            active_span.set_attribute("status", instrumentation.STATUS_ERROR)
            return update

        update = _llm_update(result)
        instrumentation.planner_completed(
            planner=PLANNER_LLM,
            duration_ms=instrumentation.elapsed_ms(started),
            tools=len(update.get("required_tools") or []),
        )
        active_span.set_attribute("status", instrumentation.STATUS_SUCCESS)
        return update


def _run_rule_planner(
    rule_planner: RulePlanner,
    state: MaintenanceState,
) -> dict[str, Any]:
    """Run the frozen planner under instrumentation.

    The frozen planner's body and its output are untouched: this wrapper measures
    it and reports the measurement. ``planner_used`` stays ``rule`` and no
    argument is derived differently, so the benchmark cannot move because of a
    timing call added around it.
    """
    started = perf_counter()
    instrumentation.planner_started(planner=PLANNER_RULE)
    with instrumentation.planner_span(planner=PLANNER_RULE) as active_span:
        update = _rule_update(rule_planner(state))
        instrumentation.planner_completed(
            planner=PLANNER_RULE,
            duration_ms=instrumentation.elapsed_ms(started),
            tools=len(update.get("required_tools") or []),
        )
        active_span.set_attribute("status", instrumentation.STATUS_SUCCESS)
    return update


def _fallback(
    rule_planner: RulePlanner,
    state: MaintenanceState,
    query: str,
    reason: str,
) -> dict[str, Any]:
    """Run the frozen planner and record why the LLM planner was not used."""
    logger.warning("planner_fallback reason=%s query=%r", reason.split(":")[0], query)
    return _rule_update(rule_planner(state), fallback_reason=reason)


__all__ = [
    "PLANNER_LLM",
    "PLANNER_RULE",
    "UNEXPECTED_PLANNER_ERROR",
    "RulePlanner",
    "dispatch_plan",
]

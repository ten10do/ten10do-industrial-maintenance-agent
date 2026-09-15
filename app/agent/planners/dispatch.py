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
from typing import Any

from app.agent.planners.llm import LLMPlanner, PlannerResult, build_llm_planner
from app.agent.planners.schema import PlannerError
from app.agent.state import MaintenanceState
from app.config import Settings, get_settings

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
        return _rule_update(rule_planner(state))

    query = state.get("query") or ""

    try:
        active = planner if planner is not None else build_llm_planner(resolved)
        result = active.plan(query)
    except PlannerError as exc:
        if mode == "llm":
            logger.error("planner_failed mode=llm code=%s", exc.code_value)
            raise
        return _fallback(rule_planner, state, query, f"{exc.code_value}: {exc.message}")
    except Exception as exc:  # defensive: an unexpected fault must not fail auto mode
        if mode == "llm":
            raise
        return _fallback(
            rule_planner,
            state,
            query,
            f"{UNEXPECTED_PLANNER_ERROR}: {type(exc).__name__}",
        )

    return _llm_update(result)


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

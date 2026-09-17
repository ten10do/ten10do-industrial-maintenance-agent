"""Invocation service that adapts the agent graph to the HTTP contract.

The graph is a synchronous, deterministic in-process callable. This service
owns the four concerns the graph must not know about:

* correlation, through a fresh ``uuid4`` per invocation;
* timing, measured with a monotonic clock around the pipeline call;
* translation, from the LangGraph state dict to :class:`AgentInvokeResponse`;
* logging, through a fixed field set that cannot carry a credential.

The workflow itself is untouched: node order, planner rules and synthesis are
exactly what ``app/agent/graph.py`` already defines. The graph is injected as a
plain callable so a test can drive the service without patching internals.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

from app.agent.graph import get_graph
from app.agent.planners.schema import PlannerError
from app.api.schemas.agent import (
    AgentDebugInfo,
    AgentInvokeResponse,
    AgentToolStatus,
)
from app.config import get_settings
from app.observability import instrumentation
from app.observability.context import request_context

logger = logging.getLogger(__name__)

#: Stable error code returned to a caller whose invocation failed.
AGENT_INVOCATION_FAILED = "agent_invocation_failed"

#: A graph callable: takes the LangGraph input payload, returns the final state.
GraphRunner = Callable[[dict[str, Any]], Mapping[str, Any]]


def _default_runner(payload: dict[str, Any]) -> Mapping[str, Any]:
    """Run the compiled workflow through its lazy singleton."""
    return get_graph().invoke(payload)


class AgentInvocationError(Exception):
    """Raised when an invocation fails before a response can be produced.

    The exception deliberately carries no caller-facing message. Only the
    ``request_id``, the code and the elapsed time travel to the HTTP layer; the
    cause is attached with ``raise ... from exc`` and stays in the process.
    """

    code = AGENT_INVOCATION_FAILED

    def __init__(self, request_id: str, latency_ms: float, *, code: str | None = None) -> None:
        super().__init__(f"agent invocation failed (request_id={request_id})")
        self.request_id = request_id
        self.latency_ms = latency_ms
        if code is not None:
            self.code = code


def _elapsed_ms(started: float) -> float:
    """Return the elapsed time in milliseconds, rounded to microsecond steps."""
    return round((perf_counter() - started) * 1000, 3)


def _optional_str(value: Any) -> str | None:
    """Coerce a state value into ``str`` while preserving ``None``."""
    if value is None:
        return None
    text = str(value)
    return text or None


def _error_code(exc: BaseException) -> str:
    """Return the caller-facing code for a failed invocation.

    A :class:`PlannerError` surfaced from ``llm`` mode already carries a stable
    V0.5 code (``LLM_TIMEOUT`` and friends). Reporting it verbatim keeps the
    failure actionable. Anything else collapses to the generic
    :data:`AGENT_INVOCATION_FAILED`, so an internal fault never leaks an
    exception class or a module path.
    """
    if isinstance(exc, PlannerError):
        return exc.code_value
    return AGENT_INVOCATION_FAILED


def _tools_called(state: Mapping[str, Any]) -> list[str]:
    """Return the tools the executor actually ran, in execution order.

    ``required_tools`` is the planner's intent; ``tool_results`` is what the
    executor managed to dispatch. The two diverge when a scheduled tool could
    not run, and "tools_called" has to mean the second one. The plan stays
    visible through the debug block rather than being reported as execution.
    """
    names: list[str] = []
    for entry in state.get("tool_results") or []:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("tool")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _tool_statuses(state: Mapping[str, Any]) -> list[AgentToolStatus]:
    """Summarize each executed tool's own reported outcome."""
    statuses: list[AgentToolStatus] = []
    for entry in state.get("tool_results") or []:
        if not isinstance(entry, Mapping):
            continue
        payload = entry.get("result")
        found = payload.get("found") if isinstance(payload, Mapping) else None
        error = payload.get("error") if isinstance(payload, Mapping) else None
        statuses.append(
            AgentToolStatus(
                tool=str(entry.get("tool") or ""),
                found=found if isinstance(found, bool) else None,
                error=str(error) if error else None,
            )
        )
    return statuses


def _debug_info(state: Mapping[str, Any]) -> AgentDebugInfo:
    """Build the opt-in diagnostic block from state the pipeline already holds."""
    required = [str(name) for name in (state.get("required_tools") or [])]
    return AgentDebugInfo(
        required_tools=required,
        tool_status=_tool_statuses(state),
        internal_error=_optional_str(state.get("error")),
        planner_fallback_reason=_optional_str(state.get("planner_fallback_reason")),
    )


class AgentService:
    """Runs one agent invocation and maps the result onto the response schema."""

    def __init__(self, runner: GraphRunner | None = None) -> None:
        self._runner = runner

    @property
    def runner(self) -> GraphRunner:
        """Return the graph callable, defaulting to the compiled pipeline."""
        return self._runner or _default_runner

    def invoke(self, query: str, *, debug: bool = False) -> AgentInvokeResponse:
        """Run the pipeline once and translate the final state.

        Args:
            query: Already-validated query text.
            debug: When ``True``, attach the diagnostic block to the response.

        Raises:
            AgentInvocationError: The pipeline raised, or returned a value that
                is not a state mapping. The original exception is chained for
                in-process diagnosis and is never returned to a caller.

        The success line renders the query with ``%r``. One line per invocation
        is part of the log contract, and a query can contain spaces, quotes or
        control characters; the repr form escapes them instead of splitting or
        breaking the line.
        """
        request_id = str(uuid.uuid4())
        started = perf_counter()
        planner_mode = get_settings().planner_mode

        with (
            request_context(request_id),
            instrumentation.request_span(planner_mode=planner_mode),
        ):
            instrumentation.request_started(planner_mode=planner_mode, query_chars=len(query))

            # The guarded region covers producing a response, not merely calling the
            # graph: a state that cannot be translated is a failure of this request
            # and must reach the caller as a structured error rather than as an
            # unhandled exception.
            try:
                state: Any = self.runner({"query": query})
                if not isinstance(state, Mapping):
                    raise TypeError(f"graph returned {type(state).__name__}, expected a mapping")

                tools_called = _tools_called(state)
                latency_ms = _elapsed_ms(started)
                response = AgentInvokeResponse(
                    request_id=request_id,
                    query=query,
                    intent=_optional_str(state.get("intent")),
                    equipment_id=_optional_str(state.get("equipment_id")),
                    alarm_code=_optional_str(state.get("alarm_code")),
                    tools_called=tools_called,
                    planner_used=_optional_str(state.get("planner_used")),
                    planner_fallback=bool(state.get("planner_fallback") or False),
                    answer=str(state.get("final_answer") or state.get("answer") or ""),
                    evidence=state.get("evidence") or state.get("retrieved_context") or [],
                    latency_ms=latency_ms,
                    debug_info=_debug_info(state) if debug else None,
                )
            except Exception as exc:
                latency_ms = _elapsed_ms(started)
                # Only the exception's class name is logged. Its message can embed
                # configuration values (a provider URL, a header, a key fragment),
                # and the log contract forbids recording those.
                logger.error(
                    "agent_invoke_failed request_id=%s success=false latency_ms=%s error_type=%s",
                    request_id,
                    latency_ms,
                    type(exc).__name__,
                )
                instrumentation.request_completed(
                    planner=None,
                    intent=None,
                    status=instrumentation.STATUS_ERROR,
                    duration_ms=latency_ms,
                )
                raise AgentInvocationError(request_id, latency_ms, code=_error_code(exc)) from exc

            logger.info(
                "agent_invoke request_id=%s success=true latency_ms=%s planner=%s "
                "fallback=%s tools_called=%s query=%r",
                request_id,
                latency_ms,
                response.planner_used or "-",
                str(response.planner_fallback).lower(),
                ",".join(tools_called) or "-",
                query,
            )
            # The request metric is recorded after the response exists, so its
            # labels describe what actually ran. ``planner_used`` is the planner
            # that produced the plan, which is not the configured mode when
            # ``auto`` fell back to the rule planner.
            instrumentation.request_completed(
                planner=response.planner_used,
                intent=response.intent,
                status=instrumentation.STATUS_SUCCESS,
                duration_ms=latency_ms,
            )
            return response

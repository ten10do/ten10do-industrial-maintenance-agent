"""Shared state schema passed between LangGraph nodes."""

from collections.abc import Sequence
from operator import add
from typing import Annotated, Any, TypedDict


class MaintenanceState(TypedDict, total=False):
    """State carried through the agent workflow.

    List fields use the ``add`` reducer so parallel branches can append
    results without overwriting each other.
    """

    # Input
    query: str
    session_id: str | None
    equipment_id: str | None
    alarm_code: str | None

    # Understanding (filled by route_query)
    intent: str | None
    required_tools: list[str]

    # Planning (filled by the configured planner: rule, llm or auto)
    #: Which planner produced ``required_tools``: "rule" or "llm".
    planner_used: str | None
    #: True when an LLM plan was attempted and the rule planner ran instead.
    planner_fallback: bool
    #: "<CODE>: <detail>" when a fallback happened, otherwise None.
    planner_fallback_reason: str | None
    #: Explicit calls produced by the LLM planner. Empty for the rule planner,
    #: where the executor derives arguments from the state as before.
    planned_tool_calls: list[dict[str, Any]]

    # Working memory
    history: Annotated[list[dict[str, Any]], add]
    retrieved_context: list[dict[str, Any]]
    plan: list[str]
    tool_results: Annotated[list[dict[str, Any]], add]

    # Output
    diagnosis: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    answer: str
    final_answer: str

    # Control
    next_step: str | None
    error: str | None


StateUpdate = dict[str, Any]
History = Sequence[dict[str, Any]]

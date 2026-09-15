"""Planner layer.

Two planners coexist:

* the frozen deterministic rule planner, which lives in ``app.agent.graph`` and is
  unchanged from V0.4;
* the optional LLM planner, which plans tool calls from natural language and is
  validated against the same registry and the same Pydantic input models.

:func:`dispatch_plan` chooses between them from configuration and records which
one ran. Neither planner can produce evidence, a diagnosis or an answer.
"""

from app.agent.planners.dispatch import (
    PLANNER_LLM,
    PLANNER_RULE,
    UNEXPECTED_PLANNER_ERROR,
    RulePlanner,
    dispatch_plan,
)
from app.agent.planners.llm import (
    LLMPlanner,
    PlannerResult,
    build_llm_planner,
    build_system_prompt,
    parse_plan_text,
)
from app.agent.planners.schema import (
    AgentPlan,
    PlannerError,
    PlannerErrorCode,
    ToolCallPlan,
)
from app.agent.planners.tool_schemas import (
    available_tool_definitions,
    available_tool_names,
    tool_definition,
    tool_parameters,
)

__all__ = [
    "PLANNER_LLM",
    "PLANNER_RULE",
    "UNEXPECTED_PLANNER_ERROR",
    "AgentPlan",
    "LLMPlanner",
    "PlannerError",
    "PlannerErrorCode",
    "PlannerResult",
    "RulePlanner",
    "ToolCallPlan",
    "available_tool_definitions",
    "available_tool_names",
    "build_llm_planner",
    "build_system_prompt",
    "dispatch_plan",
    "parse_plan_text",
    "tool_definition",
    "tool_parameters",
]

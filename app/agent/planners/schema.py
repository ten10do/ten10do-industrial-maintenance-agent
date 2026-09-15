"""Planner schemas and the planner error taxonomy.

The planner contract is small on purpose. A plan names tools and their
arguments; it cannot carry a conclusion, a diagnosis or an evidence record. The
model configs set ``extra="forbid"`` so an attempt to smuggle in an undeclared
field, an ``evidence`` list for example, is a hard validation failure rather
than something silently dropped.

``tool_name`` is validated against the registry by the planner rather than by the
model. Keeping it a plain string is what lets the planner distinguish a
hallucinated tool (``UNKNOWN_TOOL``) from an otherwise malformed payload
(``INVALID_PLANNER_OUTPUT``).
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.parser import Intent


class PlannerErrorCode(str, Enum):
    """Every way planning can fail, kept distinguishable on purpose."""

    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    INVALID_PLANNER_OUTPUT = "INVALID_PLANNER_OUTPUT"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    TOOL_ARGUMENT_VALIDATION_FAILED = "TOOL_ARGUMENT_VALIDATION_FAILED"


class PlannerError(Exception):
    """Raised when a planner cannot produce a usable plan.

    The message is diagnostic and must stay free of prompts, credentials and
    anything a caller is not allowed to see.
    """

    def __init__(self, code: PlannerErrorCode | str, message: str) -> None:
        super().__init__(message)
        self.code = PlannerErrorCode(code) if not isinstance(code, PlannerErrorCode) else code
        self.message = message

    @property
    def code_value(self) -> str:
        """Return the code as a plain string."""
        return self.code.value

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.code.value}: {self.message}"


class ToolCallPlan(BaseModel):
    """One tool the planner wants the executor to run."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Registry name of the tool. Validated against the registry by the planner.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool. Validated against the tool's own input model.",
    )
    reason: str | None = Field(
        default=None,
        max_length=500,
        description="Optional short justification produced by the planner.",
    )


class AgentPlan(BaseModel):
    """The planner's complete output.

    There is no field for evidence, a diagnosis or an answer. Grounding happens
    after execution, from real tool results.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Intent | None = Field(
        default=None,
        description="Optional classification over the same closed vocabulary the parser uses.",
    )
    tool_calls: list[ToolCallPlan] = Field(
        default_factory=list,
        description="Tools to run, in execution order. An empty list is a legal plan.",
    )

    @property
    def tool_names(self) -> list[str]:
        """Return the planned tool names in order."""
        return [call.tool_name for call in self.tool_calls]

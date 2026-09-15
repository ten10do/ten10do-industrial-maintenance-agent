"""Transport schemas for the agent invocation endpoint.

Two separate shapes describe the two outcomes. :class:`AgentInvokeResponse` is
the success contract of ``POST /agent/invoke``; :class:`AgentErrorResponse` is
the failure contract. Keeping them apart lets a caller branch on the HTTP
status code instead of parsing prose.

Neither shape carries a traceback, an exception class name, a stack frame or an
environment value. The diagnostic detail stays in the server log, correlated by
``request_id``.

The query bounds live in module constants so the schema, the tests and the
documentation cannot drift apart.
"""

from pydantic import BaseModel, Field, field_validator

from app.schemas import Evidence

#: ``query`` must carry at least one character.
AGENT_QUERY_MIN_LENGTH = 1
#: ``query`` is capped so a single request cannot carry an unbounded payload.
AGENT_QUERY_MAX_LENGTH = 2000


class AgentInvokeRequest(BaseModel):
    """Body of ``POST /agent/invoke``."""

    query: str = Field(
        ...,
        min_length=AGENT_QUERY_MIN_LENGTH,
        max_length=AGENT_QUERY_MAX_LENGTH,
        description="Natural-language maintenance question.",
    )
    debug: bool = Field(
        default=False,
        description=(
            "Include the diagnostic block (planned tools, per-tool status, "
            "internal pipeline error). It never contains a traceback or an "
            "environment value."
        ),
    )

    @field_validator("query")
    @classmethod
    def _normalize_query(cls, value: str) -> str:
        """Strip padding and reject a query that is blank once stripped.

        ``min_length=1`` alone accepts a run of spaces, which is not a
        question. Rejecting it here keeps the parser downstream from receiving
        padding, and the rejected case surfaces as an ordinary 422.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class AgentToolStatus(BaseModel):
    """Outcome of one tool run, reported only in the debug block."""

    tool: str = Field(..., description="Registry name of the tool.")
    found: bool | None = Field(
        default=None,
        description="The tool's own ``found`` flag, or None when it does not report one.",
    )
    error: str | None = Field(
        default=None,
        description="The tool's own diagnostic, when it could not complete.",
    )


class AgentDebugInfo(BaseModel):
    """Diagnostics returned only when the caller sets ``debug=true``.

    Nothing here reproduces the planner's prompt or the model's raw output. A
    model can echo its prompt, and the prompt carries the tool schemas and the
    device vocabulary, so only curated fields are exposed.
    """

    required_tools: list[str] = Field(
        default_factory=list,
        description="Tools the planner scheduled. May exceed ``tools_called``.",
    )
    tool_status: list[AgentToolStatus] = Field(default_factory=list)
    internal_error: str | None = Field(
        default=None,
        description="Pipeline-level diagnostic, for example an undispatched tool.",
    )
    planner_fallback_reason: str | None = Field(
        default=None,
        description=(
            "Set when an LLM plan was attempted and the rule planner ran instead. "
            "Formatted as '<CODE>: <detail>'."
        ),
    )


class AgentInvokeResponse(BaseModel):
    """Successful agent response.

    A response can still describe a degraded run: a tool that returned nothing,
    or a knowledge base that was unavailable, is reported inside ``answer`` and
    reflected in ``evidence``. Degradation is never dressed up as success.
    """

    request_id: str = Field(..., description="Correlation id for this invocation.")
    query: str = Field(..., description="Echo of the normalized query.")
    intent: str | None = Field(default=None, description="Intent classified by the parser.")
    equipment_id: str | None = Field(default=None, description="Equipment id, when resolved.")
    alarm_code: str | None = Field(default=None, description="Alarm code, when resolved.")
    tools_called: list[str] = Field(
        default_factory=list,
        description="Tools the executor actually ran, in execution order.",
    )
    planner_used: str | None = Field(
        default=None,
        description="Planner that produced the plan: 'rule' or 'llm'.",
    )
    planner_fallback: bool = Field(
        default=False,
        description="True when an LLM plan was attempted and the rule planner ran instead.",
    )
    answer: str = Field(default="", description="Deterministic Chinese answer.")
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Citable facts, including document evidence from RAG retrieval.",
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="Measured pipeline latency.")
    debug_info: AgentDebugInfo | None = Field(
        default=None,
        description="Populated only when the request asked for debug output.",
    )


class AgentErrorResponse(BaseModel):
    """Structured failure body.

    ``message`` is intentionally generic. The exception type, its message and
    its stack live in the server log, reachable through ``request_id``.
    """

    request_id: str = Field(..., description="Correlation id, usable to find the server log.")
    error: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Caller-safe explanation.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Time spent before failing.")

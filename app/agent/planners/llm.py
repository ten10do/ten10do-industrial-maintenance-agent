"""The LLM planner: natural language in, a validated plan out.

The planner does one thing. It asks a model which tools to run and with which
arguments, then it proves the answer is usable before anything executes:

1. the response must parse as a JSON object;
2. the object must match :class:`~app.agent.planners.schema.AgentPlan`;
3. every named tool must exist in the registry;
4. every argument set must satisfy the tool's own Pydantic input model.

Any failure raises :class:`~app.agent.planners.schema.PlannerError` with a code
from the V0.5 taxonomy. Nothing is repaired, defaulted or guessed: a plan that
needs repair is a plan the planner refuses.

The planner never produces evidence, a diagnosis or an answer. Its output is a
list of tool calls, and the pipeline grounds the answer in what those tools
actually returned.

Neither the prompt nor the raw model output is stored in the agent state. A model
can echo its prompt, and the prompt carries the tool schemas and the device
vocabulary, so only the validated plan and non-content metadata travel onwards.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.agent.planners.schema import (
    AgentPlan,
    PlannerError,
    PlannerErrorCode,
)
from app.agent.planners.tool_schemas import available_tool_definitions
from app.config import Settings, get_settings
from app.integrations.llm import (
    LLMCompletionRequest,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
    get_llm_provider,
)
from app.services.device_catalog import device_vocabulary
from app.tools.arguments import (
    ToolArgumentError,
    describe_validation_error,
    validate_tool_arguments,
)
from app.tools.registry import registry

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are the planning component of an industrial maintenance agent.

Your only job is to decide which of the available tools to call and with which
arguments. You do not answer the user, you do not diagnose faults, and you do
not produce evidence. Retrieval and diagnosis happen after you, from the real
results the tools return.

Return exactly one JSON object and nothing else. No prose, no code fence.

The object has exactly these fields:
  "intent":      one of "alarm_diagnosis", "maintenance_advice", "device_status",
                 "unknown", or null.
  "tool_calls":  an ordered array. Each entry has exactly these fields:
                   "tool_name":  one of the tool names listed below.
                   "arguments":  an object of arguments for that tool.
                   "reason":     a short string, or null.

Rules you must follow:
1. Use only the tool names listed below. Any other name is invalid.
2. Supply every argument the tool requires. Never invent a value that cannot be
   derived from the user's request.
3. Do not add fields. Undeclared fields make the whole plan invalid.
4. Order the calls the way they should run.
5. An empty "tool_calls" array is allowed when no tool applies.

Available tools, as JSON Schema:
{tools}

Known device identifiers. When the request refers to one of these devices, pass
the identifier exactly as written here, not the display name:
{devices}
"""


class PlannerResult(BaseModel):
    """A validated plan plus the metadata worth reporting.

    Deliberately absent: the prompt, the raw completion text and anything else a
    model could have echoed.
    """

    plan: AgentPlan
    planner: str = Field(default="llm", description="Planner that produced this plan.")
    provider: str = Field(default="", description="LLM provider identifier.")
    model: str | None = Field(default=None, description="Model that answered.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Measured provider latency.")


def _format_tools() -> str:
    """Render the registry's tool definitions for the prompt."""
    return json.dumps(available_tool_definitions(), ensure_ascii=False, indent=2)


def _format_devices() -> str:
    """Render the device identifier space for the prompt."""
    pairs = device_vocabulary()
    if not pairs:
        return "(no device identifiers are registered)"
    return "\n".join(
        f"- {device_id} ({name})" if name else f"- {device_id}" for device_id, name in pairs
    )


def build_system_prompt() -> str:
    """Build the planner system prompt from the registry and the device catalog."""
    return PLANNER_SYSTEM_PROMPT.format(tools=_format_tools(), devices=_format_devices())


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown fence, if the model wrapped its output."""
    body = text.strip()
    if not body.startswith("```"):
        return body
    if "\n" in body:
        body = body.split("\n", 1)[1]
    body = body.rstrip()
    if body.endswith("```"):
        body = body[: -len("```")]
    return body.strip()


def _first_balanced_object(text: str) -> dict[str, Any]:
    """Return the first complete JSON object in ``text``.

    Tolerating a wrapper is a robustness measure, not a relaxation: whatever is
    extracted still has to satisfy the plan schema.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in planner output")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(text[start : index + 1])
                if not isinstance(payload, dict):
                    raise ValueError("planner output is not a JSON object")
                return payload

    raise ValueError("unbalanced JSON object in planner output")


def parse_plan_text(text: str) -> dict[str, Any]:
    """Extract the plan object from a model response.

    Raises:
        ValueError: The response holds no usable JSON object.
    """
    candidate = _strip_code_fence(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return _first_balanced_object(candidate)
    if not isinstance(payload, dict):
        raise ValueError("planner output is not a JSON object")
    return payload


class LLMPlanner:
    """Plans tool calls with an LLM and validates the result."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not model.strip():
            raise PlannerError(
                PlannerErrorCode.LLM_PROVIDER_ERROR,
                "LLM planner is not configured: missing LLM_MODEL",
            )
        self._provider = provider
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    @property
    def provider(self) -> LLMProvider:
        """Return the underlying provider."""
        return self._provider

    def build_messages(self, query: str) -> list[LLMMessage]:
        """Build the planner conversation for one query."""
        return [
            LLMMessage(role="system", content=build_system_prompt()),
            LLMMessage(role="user", content=query),
        ]

    def plan(self, query: str) -> PlannerResult:
        """Produce a validated plan for ``query``.

        Raises:
            PlannerError: With a code from the V0.5 taxonomy. There is no
                partial success: a plan either validates completely or fails.
        """
        request = LLMCompletionRequest(
            messages=self.build_messages(query),
            model=self.model,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            json_object=True,
        )

        try:
            completion = self._provider.complete(request)
        except LLMTimeoutError as exc:
            raise PlannerError(PlannerErrorCode.LLM_TIMEOUT, exc.message) from exc
        except LLMProviderError as exc:
            raise PlannerError(PlannerErrorCode.LLM_PROVIDER_ERROR, exc.message) from exc

        plan = self._validate(completion)
        return PlannerResult(
            plan=plan,
            planner="llm",
            provider=completion.provider,
            model=completion.model,
            latency_ms=completion.latency_ms,
        )

    def _validate(self, completion: LLMCompletionResult) -> AgentPlan:
        """Parse and validate a completion into an executable plan."""
        try:
            payload = parse_plan_text(completion.text)
        except ValueError as exc:
            raise PlannerError(
                PlannerErrorCode.INVALID_PLANNER_OUTPUT,
                f"planner output could not be read as a JSON object: {exc}",
            ) from exc

        try:
            plan = AgentPlan.model_validate(payload)
        except ValidationError as exc:
            raise PlannerError(
                PlannerErrorCode.INVALID_PLANNER_OUTPUT,
                f"planner output did not match the plan schema: {describe_validation_error(exc)}",
            ) from exc

        self._validate_tools(plan)
        return plan

    @staticmethod
    def _validate_tools(plan: AgentPlan) -> None:
        """Reject unknown tools and arguments that do not satisfy the tool model."""
        known = set(registry.names())
        for call in plan.tool_calls:
            if call.tool_name not in known:
                raise PlannerError(
                    PlannerErrorCode.UNKNOWN_TOOL,
                    f"planner selected an unknown tool: {call.tool_name}",
                )
            try:
                validate_tool_arguments(call.tool_name, call.arguments)
            except ToolArgumentError as exc:
                raise PlannerError(
                    PlannerErrorCode.TOOL_ARGUMENT_VALIDATION_FAILED,
                    str(exc),
                ) from exc


def build_llm_planner(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
) -> LLMPlanner:
    """Build the configured LLM planner.

    Raises:
        PlannerError: The provider cannot be built, which surfaces as
            ``LLM_PROVIDER_ERROR`` so the caller can classify it like any other
            provider failure.
    """
    resolved = settings or get_settings()
    active = provider
    if active is None:
        try:
            active = get_llm_provider()
        except LLMProviderError as exc:
            raise PlannerError(PlannerErrorCode.LLM_PROVIDER_ERROR, exc.message) from exc

    return LLMPlanner(
        active,
        resolved.llm_model,
        temperature=resolved.llm_temperature,
        timeout_seconds=resolved.llm_timeout_seconds,
    )


__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "LLMPlanner",
    "PlannerResult",
    "build_llm_planner",
    "build_system_prompt",
    "parse_plan_text",
]

"""Tests for the LLM planner.

The planner is the only component that turns model output into something the
executor will run, so these tests pin the validation chain end to end:

1. the system prompt carries the tool schemas and the device vocabulary, and
   nothing else;
2. a well-formed plan validates into an :class:`AgentPlan`;
3. every malformed output raises a :class:`PlannerError` with the right code from
   the V0.5 taxonomy;
4. the result carries the plan and metadata but never the prompt or the raw
   completion, so neither can reach a state, a response or a log.

No test opens a socket: the provider is a :class:`FakeLLMProvider`.
"""

from __future__ import annotations

import json

import pytest

from app.agent.planners import (
    AgentPlan,
    LLMPlanner,
    PlannerError,
    PlannerErrorCode,
    PlannerResult,
    build_llm_planner,
    build_system_prompt,
    parse_plan_text,
)
from app.config import Settings
from app.tools.names import ToolName
from tests.fake_llm import (
    FAKE_API_KEY,
    FAKE_BASE_URL,
    FAKE_MODEL,
    FakeLLMProvider,
    fake_secret,
    plan_text,
    provider_failed,
    timed_out,
)

ALARM_TOOL = ToolName.QUERY_ALARM_CODE.value
DEVICE_TOOL = ToolName.GET_DEVICE_STATUS.value


def _settings(**overrides: object) -> Settings:
    """Build settings describing a configured LLM planner."""
    values: dict[str, object] = {
        "planner_mode": "llm",
        "llm_base_url": FAKE_BASE_URL,
        "llm_api_key": fake_secret(),
        "llm_model": FAKE_MODEL,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _planner(provider: FakeLLMProvider) -> LLMPlanner:
    """Build a planner backed by ``provider``."""
    return build_llm_planner(_settings(), provider=provider)


def _plan_with(text: str) -> PlannerResult:
    """Run the planner over a fixed completion body."""
    return _planner(FakeLLMProvider(text=text)).plan("包装线PLC报警F0045怎么办")


# --------------------------------------------------------------------------- #
# 1. System prompt
# --------------------------------------------------------------------------- #


def test_system_prompt_describes_the_registry_tools() -> None:
    prompt = build_system_prompt()

    for name in (DEVICE_TOOL, ALARM_TOOL, ToolName.SEARCH_MAINTENANCE_MANUAL.value):
        assert name in prompt
    # The argument names come from the tools' own input models.
    assert "alarm_code" in prompt
    assert "device_id" in prompt


def test_system_prompt_lists_the_device_vocabulary() -> None:
    prompt = build_system_prompt()

    assert "PLC-001" in prompt
    assert "包装线PLC" in prompt


def test_system_prompt_states_the_planner_only_plans() -> None:
    # The prompt is wrapped across source lines, so compare with the line breaks
    # collapsed rather than against the literal layout.
    prompt = " ".join(build_system_prompt().split())

    assert "do not diagnose faults" in prompt
    assert "do not produce evidence" in prompt
    assert "Retrieval and diagnosis happen after you" in prompt


def test_system_prompt_carries_no_credential() -> None:
    assert FAKE_API_KEY not in build_system_prompt()


def test_planner_never_attaches_the_key_to_a_message() -> None:
    provider = FakeLLMProvider(text=plan_text())
    planner = _planner(provider)

    planner.plan("payload check")

    request = provider.last_request
    assert request is not None
    rendered = json.dumps(
        [message.model_dump() for message in request.messages],
        ensure_ascii=False,
    )
    assert FAKE_API_KEY not in rendered
    # The user's query is the second message, verbatim.
    assert request.messages[-1].content == "payload check"
    assert request.model == FAKE_MODEL


# --------------------------------------------------------------------------- #
# 2. A well-formed plan
# --------------------------------------------------------------------------- #


def test_valid_plan_is_accepted() -> None:
    body = plan_text(
        [{"tool_name": ALARM_TOOL, "arguments": {"alarm_code": "F0045"}, "reason": "alarm query"}],
        intent="alarm_diagnosis",
    )
    provider = FakeLLMProvider(text=body)

    result = _planner(provider).plan("包装线PLC报警F0045怎么办")

    assert isinstance(result.plan, AgentPlan)
    assert result.plan.tool_names == [ALARM_TOOL]
    assert result.plan.intent is not None
    assert result.plan.intent.value == "alarm_diagnosis"
    assert result.planner == "llm"
    assert result.provider == "fake"
    assert result.model == FAKE_MODEL
    assert result.latency_ms >= 0


def test_empty_tool_calls_are_a_legal_plan() -> None:
    result = _plan_with(plan_text([]))

    assert result.plan.tool_calls == []
    assert result.plan.tool_names == []


def test_code_fence_and_surrounding_prose_are_tolerated() -> None:
    body = "Here is the plan:\n```json\n" + plan_text([]) + "\n```\n"
    result = _plan_with(body)

    assert result.plan.tool_calls == []


# --------------------------------------------------------------------------- #
# 3. The error taxonomy
# --------------------------------------------------------------------------- #


def test_non_json_output_is_rejected() -> None:
    with pytest.raises(PlannerError) as excinfo:
        _plan_with("I cannot help with that.")

    assert excinfo.value.code is PlannerErrorCode.INVALID_PLANNER_OUTPUT


def test_schema_mismatch_is_rejected() -> None:
    body = json.dumps({"tool_calls": "not-a-list"})

    with pytest.raises(PlannerError) as excinfo:
        _plan_with(body)

    assert excinfo.value.code is PlannerErrorCode.INVALID_PLANNER_OUTPUT


def test_a_smuggled_field_is_rejected() -> None:
    """A plan cannot carry evidence: ``extra='forbid'`` makes that a failure."""
    body = json.dumps(
        {
            "intent": None,
            "tool_calls": [],
            "evidence": [{"source": "made-up", "content": "invented"}],
        }
    )

    with pytest.raises(PlannerError) as excinfo:
        _plan_with(body)

    assert excinfo.value.code is PlannerErrorCode.INVALID_PLANNER_OUTPUT


def test_unknown_tool_is_rejected() -> None:
    body = plan_text([{"tool_name": "delete_everything", "arguments": {}}])

    with pytest.raises(PlannerError) as excinfo:
        _plan_with(body)

    assert excinfo.value.code is PlannerErrorCode.UNKNOWN_TOOL
    assert "delete_everything" in excinfo.value.message


def test_missing_required_argument_is_rejected() -> None:
    body = plan_text([{"tool_name": ALARM_TOOL, "arguments": {}}])

    with pytest.raises(PlannerError) as excinfo:
        _plan_with(body)

    assert excinfo.value.code is PlannerErrorCode.TOOL_ARGUMENT_VALIDATION_FAILED


def test_undeclared_argument_is_rejected() -> None:
    body = plan_text(
        [{"tool_name": ALARM_TOOL, "arguments": {"alarm_code": "F0045", "force": True}}]
    )

    with pytest.raises(PlannerError) as excinfo:
        _plan_with(body)

    assert excinfo.value.code is PlannerErrorCode.TOOL_ARGUMENT_VALIDATION_FAILED
    assert "force" in excinfo.value.message


def test_provider_timeout_maps_to_llm_timeout() -> None:
    provider = FakeLLMProvider(error=timed_out())

    with pytest.raises(PlannerError) as excinfo:
        _planner(provider).plan("anything")

    assert excinfo.value.code is PlannerErrorCode.LLM_TIMEOUT


def test_provider_failure_maps_to_provider_error() -> None:
    provider = FakeLLMProvider(error=provider_failed("HTTP 503"))

    with pytest.raises(PlannerError) as excinfo:
        _planner(provider).plan("anything")

    assert excinfo.value.code is PlannerErrorCode.LLM_PROVIDER_ERROR


def test_planner_without_a_model_is_rejected() -> None:
    provider = FakeLLMProvider(text=plan_text())

    with pytest.raises(PlannerError) as excinfo:
        build_llm_planner(_settings(llm_model=""), provider=provider)

    assert excinfo.value.code is PlannerErrorCode.LLM_PROVIDER_ERROR


# --------------------------------------------------------------------------- #
# 4. The result carries no prompt and no raw output
# --------------------------------------------------------------------------- #


def test_result_holds_no_prompt_or_raw_text() -> None:
    result = _plan_with(plan_text([]))
    fields = set(type(result).model_fields)

    assert "prompt" not in fields
    assert "text" not in fields
    assert "raw" not in fields
    assert FAKE_API_KEY not in result.model_dump_json()


def test_parse_plan_text_extracts_the_first_object() -> None:
    payload = parse_plan_text('prefix {"intent": null, "tool_calls": []} suffix')

    assert payload == {"intent": None, "tool_calls": []}


def test_parse_plan_text_rejects_unbalanced_json() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        parse_plan_text('{"intent": null, "tool_calls": []')

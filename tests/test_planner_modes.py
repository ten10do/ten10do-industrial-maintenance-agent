"""Tests for planner selection, executor behaviour and the API surface.

This suite is the acceptance test for V0.5. It pins the four promises the
feature makes:

1. ``rule`` stays the frozen baseline: the deterministic pipeline is byte-for-byte
   the V0.4 behaviour and no provider is built at all;
2. ``llm`` plans with the model and never degrades quietly - a failure reaches the
   caller as a structured error;
3. ``auto`` prefers the model and falls back to the rule planner, recording why;
4. the executor runs the plan and nothing else, and the answer stays grounded in
   what the tools actually returned.

The LLM is always a double. Nothing here opens a socket or needs a credential.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.graph import get_graph, plan_actions
from app.agent.planners import (
    LLMPlanner,
    PlannerError,
    PlannerErrorCode,
    dispatch_plan,
)
from app.agent.planners.tool_schemas import (
    available_tool_names,
    tool_parameters,
)
from app.agent.state import MaintenanceState
from app.config import PlannerMode, Settings
from app.database import session as db_session
from app.database.init_db import init_db
from app.integrations.llm import OpenAICompatibleProvider
from app.integrations.rag import RAGProvider, RAGSearchHit, RAGSearchResponse
from app.main import app
from app.tools import registry
from app.tools.arguments import ToolArgumentError, validate_tool_arguments
from app.tools.device_tool import DeviceStatusInput
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

DEVICE_TOOL = ToolName.GET_DEVICE_STATUS.value
ALARM_TOOL = ToolName.QUERY_ALARM_CODE.value
MANUAL_TOOL = ToolName.SEARCH_MAINTENANCE_MANUAL.value

ALARM_QUERY = "包装线PLC报警F0045怎么办"
INVOKE_PATH = "/agent/invoke"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


class _EmptyManualProvider(RAGProvider):
    """Provider double reporting a successful, empty retrieval."""

    provider_id = "test-empty"

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        return RAGSearchResponse(query=query, hits=[], retrieval_mode="hybrid")


@pytest.fixture(autouse=True)
def _stub_manual_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every planner test independent of any real knowledge base."""
    monkeypatch.setattr(
        "app.integrations.rag.factory.get_rag_provider",
        lambda: _EmptyManualProvider(),
    )


@pytest.fixture()
def seeded_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[sessionmaker]:
    """Point the session factory at a seeded in-memory database.

    ``StaticPool`` makes the single in-memory database visible from the worker
    thread Starlette uses for the synchronous route.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    factory = sessionmaker(bind=engine, future=True)
    init_db(seed=True, reset=True, engine=engine, session_factory=factory)
    monkeypatch.setattr(db_session, "SessionLocal", factory)
    try:
        yield factory
    finally:
        engine.dispose()


def _settings(mode: PlannerMode = "rule", **overrides: object) -> Settings:
    """Build planner settings, configured for a reachable-looking endpoint."""
    values: dict[str, object] = {
        "planner_mode": mode,
        "llm_base_url": FAKE_BASE_URL,
        "llm_api_key": fake_secret(),
        "llm_model": FAKE_MODEL,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.fixture()
def llm_gateway(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a fake LLM planner for the graph and the API.

    Returns an installer that takes the mode and an optional provider, patches the
    dispatcher's settings and its planner factory, and returns the provider so a
    test can assert how many completions were requested.
    """

    def _install(
        mode: PlannerMode,
        provider: FakeLLMProvider | None = None,
        **settings_overrides: object,
    ) -> FakeLLMProvider:
        active = provider if provider is not None else FakeLLMProvider(text=plan_text())
        resolved = _settings(mode, **settings_overrides)
        monkeypatch.setattr("app.agent.planners.dispatch.get_settings", lambda: resolved)
        monkeypatch.setattr(
            "app.agent.planners.dispatch.build_llm_planner",
            lambda settings=None, **kwargs: LLMPlanner(
                active,
                resolved.llm_model,
                temperature=resolved.llm_temperature,
                timeout_seconds=resolved.llm_timeout_seconds,
            ),
        )
        return active

    return _install


def _rule_stub(calls: list[MaintenanceState]) -> Any:
    """Return a rule planner that records the states it was handed."""

    def _plan(state: MaintenanceState) -> dict[str, Any]:
        calls.append(state)
        return {"required_tools": [ALARM_TOOL]}

    return _plan


# --------------------------------------------------------------------------- #
# 1. rule mode is the frozen baseline
# --------------------------------------------------------------------------- #


def test_rule_mode_uses_the_frozen_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.agent.planners.dispatch.build_llm_planner",
        lambda *args, **kwargs: pytest.fail("rule mode must not build an LLM planner"),
    )
    # ``intent`` is normally filled by ``route_query``; this unit test drives the
    # planner directly, so it supplies the field the rule planner reads.
    state = MaintenanceState(
        query=ALARM_QUERY,
        intent="alarm_diagnosis",
        equipment_id="PLC-001",
        alarm_code="F0045",
    )

    update = dispatch_plan(state, rule_planner=plan_actions, settings=_settings("rule"))

    assert update["required_tools"] == [DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL]
    assert update["planner_used"] == "rule"
    assert update["planner_fallback"] is False
    assert update["planner_fallback_reason"] is None
    assert update["planned_tool_calls"] == []


def test_rule_mode_is_the_default_configuration() -> None:
    assert Settings().planner_mode == "rule"


def test_rule_mode_end_to_end_is_unchanged(seeded_factory: sessionmaker) -> None:
    """The default pipeline still plans, executes and grounds as it did in V0.4."""
    state = get_graph().invoke({"query": ALARM_QUERY})

    assert state["planner_used"] == "rule"
    assert state["planner_fallback"] is False
    assert state["required_tools"] == [DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL]
    assert state["planned_tool_calls"] == []
    assert [entry["tool"] for entry in state["tool_results"]] == [
        DEVICE_TOOL,
        ALARM_TOOL,
        MANUAL_TOOL,
    ]
    assert "PLC-001" in state["final_answer"]


# --------------------------------------------------------------------------- #
# 2. llm mode
# --------------------------------------------------------------------------- #


def test_llm_mode_records_the_plan() -> None:
    provider = FakeLLMProvider(
        text=plan_text(
            [{"tool_name": ALARM_TOOL, "arguments": {"alarm_code": "F0045"}}],
            intent="alarm_diagnosis",
        )
    )
    planner = LLMPlanner(provider, FAKE_MODEL)

    update = dispatch_plan(
        MaintenanceState(query=ALARM_QUERY),
        rule_planner=_rule_stub([]),
        settings=_settings("llm"),
        planner=planner,
    )

    assert update["planner_used"] == "llm"
    assert update["planner_fallback"] is False
    assert update["required_tools"] == [ALARM_TOOL]
    assert update["planned_tool_calls"] == [
        {"tool": ALARM_TOOL, "arguments": {"alarm_code": "F0045"}, "reason": None}
    ]
    assert update["intent"] == "alarm_diagnosis"


@pytest.mark.parametrize("error", [timed_out(), provider_failed("HTTP 503")])
def test_llm_mode_failure_is_not_downgraded(error: BaseException) -> None:
    calls: list[MaintenanceState] = []
    planner = LLMPlanner(FakeLLMProvider(error=error), FAKE_MODEL)

    with pytest.raises(PlannerError) as excinfo:
        dispatch_plan(
            MaintenanceState(query=ALARM_QUERY),
            rule_planner=_rule_stub(calls),
            settings=_settings("llm"),
            planner=planner,
        )

    assert excinfo.value.code in {PlannerErrorCode.LLM_TIMEOUT, PlannerErrorCode.LLM_PROVIDER_ERROR}
    assert calls == []


# --------------------------------------------------------------------------- #
# 3. auto mode
# --------------------------------------------------------------------------- #


def test_auto_mode_uses_the_llm_when_it_succeeds() -> None:
    provider = FakeLLMProvider(
        text=plan_text([{"tool_name": DEVICE_TOOL, "arguments": {"device_id": "PLC-001"}}])
    )
    calls: list[MaintenanceState] = []

    update = dispatch_plan(
        MaintenanceState(query="PLC-001 现在什么状态"),
        rule_planner=_rule_stub(calls),
        settings=_settings("auto"),
        planner=LLMPlanner(provider, FAKE_MODEL),
    )

    assert update["planner_used"] == "llm"
    assert update["planner_fallback"] is False
    assert update["required_tools"] == [DEVICE_TOOL]
    assert calls == []


def test_auto_mode_falls_back_and_records_the_reason() -> None:
    calls: list[MaintenanceState] = []

    update = dispatch_plan(
        MaintenanceState(query=ALARM_QUERY),
        rule_planner=_rule_stub(calls),
        settings=_settings("auto"),
        planner=LLMPlanner(FakeLLMProvider(error=timed_out()), FAKE_MODEL),
    )

    assert update["planner_used"] == "rule"
    assert update["planner_fallback"] is True
    assert update["planner_fallback_reason"].startswith("LLM_TIMEOUT: ")
    assert update["required_tools"] == [ALARM_TOOL]
    assert len(calls) == 1


def test_auto_mode_falls_back_on_an_invalid_plan() -> None:
    update = dispatch_plan(
        MaintenanceState(query=ALARM_QUERY),
        rule_planner=_rule_stub([]),
        settings=_settings("auto"),
        planner=LLMPlanner(FakeLLMProvider(text="not json"), FAKE_MODEL),
    )

    assert update["planner_fallback"] is True
    assert update["planner_fallback_reason"].startswith("INVALID_PLANNER_OUTPUT: ")


def test_auto_mode_records_unexpected_faults_separately() -> None:
    class _ExplodingProvider(FakeLLMProvider):
        def complete(self, request: Any) -> Any:
            raise ValueError("boom")

    update = dispatch_plan(
        MaintenanceState(query=ALARM_QUERY),
        rule_planner=_rule_stub([]),
        settings=_settings("auto"),
        planner=LLMPlanner(_ExplodingProvider(), FAKE_MODEL),
    )

    assert update["planner_fallback"] is True
    assert update["planner_fallback_reason"].startswith("UNEXPECTED_PLANNER_ERROR: ")


# --------------------------------------------------------------------------- #
# 4. The executor runs the plan and nothing else
# --------------------------------------------------------------------------- #


def test_llm_plan_is_executed_end_to_end(seeded_factory: sessionmaker, llm_gateway: Any) -> None:
    provider = llm_gateway(
        "llm",
        FakeLLMProvider(
            text=plan_text(
                [
                    {"tool_name": DEVICE_TOOL, "arguments": {"device_id": "PLC-001"}},
                    {"tool_name": ALARM_TOOL, "arguments": {"alarm_code": "F0045"}},
                ],
                intent="alarm_diagnosis",
            )
        ),
    )

    state = get_graph().invoke({"query": ALARM_QUERY})

    assert state["planner_used"] == "llm"
    assert provider.call_count == 1
    assert [entry["tool"] for entry in state["tool_results"]] == [DEVICE_TOOL, ALARM_TOOL]
    assert "PLC-001" in state["final_answer"]
    assert any(item["content"].startswith("报警码 F0045") for item in state["evidence"])


def test_executor_skips_an_unknown_tool_instead_of_guessing(
    seeded_factory: sessionmaker,
) -> None:
    """The plan is authoritative: a call that cannot run is reported, not repaired."""
    from app.agent.graph import execute_tools

    state = MaintenanceState(
        query=ALARM_QUERY,
        planned_tool_calls=[
            {"tool": "delete_everything", "arguments": {}},
            {"tool": ALARM_TOOL, "arguments": {"alarm_code": "F0045"}},
        ],
    )

    update = execute_tools(state)

    assert [entry["tool"] for entry in update["tool_results"]] == [ALARM_TOOL]
    assert "delete_everything" in update["error"]


def test_executor_reports_invalid_arguments_without_completing_them(
    seeded_factory: sessionmaker,
) -> None:
    from app.agent.graph import execute_tools

    state = MaintenanceState(
        query=ALARM_QUERY,
        planned_tool_calls=[{"tool": ALARM_TOOL, "arguments": {"alarm_code": ""}}],
    )

    update = execute_tools(state)

    assert update["tool_results"] == []
    assert update["error"]


# --------------------------------------------------------------------------- #
# 5. Tool schemas have a single source of truth
# --------------------------------------------------------------------------- #


def test_tool_schema_is_generated_from_the_input_model() -> None:
    spec = registry.get(DEVICE_TOOL)

    assert spec.input_model is DeviceStatusInput
    assert tool_parameters(spec) == DeviceStatusInput.model_json_schema()


def test_available_tool_names_match_the_registry() -> None:
    assert available_tool_names() == registry.names()


def test_every_registered_tool_declares_an_input_model() -> None:
    for spec in registry.list_tools():
        assert spec.input_model is not None, spec.name


# --------------------------------------------------------------------------- #
# 6. Argument validation is strict
# --------------------------------------------------------------------------- #


def test_undeclared_argument_is_rejected() -> None:
    with pytest.raises(ToolArgumentError, match="undeclared"):
        validate_tool_arguments(DEVICE_TOOL, {"device_id": "PLC-001", "extra": 1})


def test_missing_required_argument_is_rejected() -> None:
    with pytest.raises(ToolArgumentError, match="device_id"):
        validate_tool_arguments(DEVICE_TOOL, {})


def test_valid_arguments_pass_through() -> None:
    validated = validate_tool_arguments(DEVICE_TOOL, {"device_id": "PLC-001"})

    assert validated == {"device_id": "PLC-001"}


# --------------------------------------------------------------------------- #
# 7. API surface
# --------------------------------------------------------------------------- #


def test_api_reports_the_planner_that_ran(seeded_factory: sessionmaker, llm_gateway: Any) -> None:
    llm_gateway(
        "llm",
        FakeLLMProvider(
            text=plan_text([{"tool_name": DEVICE_TOOL, "arguments": {"device_id": "PLC-001"}}])
        ),
    )

    payload = client.post(INVOKE_PATH, json={"query": "PLC-001 现在什么状态"}).json()

    assert payload["planner_used"] == "llm"
    assert payload["planner_fallback"] is False
    assert payload["tools_called"] == [DEVICE_TOOL]


def test_api_reports_a_fallback_with_its_reason(
    seeded_factory: sessionmaker, llm_gateway: Any
) -> None:
    llm_gateway("auto", FakeLLMProvider(error=timed_out()))

    payload = client.post(INVOKE_PATH, json={"query": ALARM_QUERY, "debug": True}).json()

    assert payload["planner_used"] == "rule"
    assert payload["planner_fallback"] is True
    assert payload["debug_info"]["planner_fallback_reason"].startswith("LLM_TIMEOUT: ")


def test_api_llm_failure_returns_the_planner_code(
    seeded_factory: sessionmaker, llm_gateway: Any
) -> None:
    llm_gateway("llm", FakeLLMProvider(error=timed_out()))

    response = client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    assert response.status_code == 500
    assert response.json()["error"] == "LLM_TIMEOUT"


def test_api_exposes_no_prompt_or_key(seeded_factory: sessionmaker, llm_gateway: Any) -> None:
    llm_gateway("llm", FakeLLMProvider(text=plan_text([])))

    body = client.post(INVOKE_PATH, json={"query": ALARM_QUERY, "debug": True}).text

    assert "You are the planning component" not in body
    assert FAKE_API_KEY not in body
    assert "LLM_API_KEY" not in body


def test_api_debug_block_never_contains_the_key(
    seeded_factory: sessionmaker, llm_gateway: Any
) -> None:
    llm_gateway("auto", FakeLLMProvider(error=provider_failed("HTTP 500")))

    payload = client.post(INVOKE_PATH, json={"query": ALARM_QUERY, "debug": True}).json()

    assert FAKE_API_KEY not in json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 8. The credential never reaches a log
# --------------------------------------------------------------------------- #


def test_real_provider_failure_does_not_leak_the_key(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A misbehaving endpoint that echoes the key still cannot leak it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"internal error, saw token={FAKE_API_KEY}")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    provider = OpenAICompatibleProvider(
        base_url=FAKE_BASE_URL,
        api_key=fake_secret(),
        model=FAKE_MODEL,
        client=http_client,
    )
    resolved = _settings("llm")
    monkeypatch.setattr("app.agent.planners.dispatch.get_settings", lambda: resolved)
    monkeypatch.setattr(
        "app.agent.planners.dispatch.build_llm_planner",
        lambda settings=None, **kwargs: LLMPlanner(provider, FAKE_MODEL),
    )

    with caplog.at_level(logging.DEBUG):
        response = client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    assert response.status_code == 500
    assert response.json()["error"] == "LLM_PROVIDER_ERROR"
    assert FAKE_API_KEY not in response.text
    assert FAKE_API_KEY not in caplog.text
    assert "http_client" not in caplog.text
    http_client.close()


def test_success_log_records_the_planner(
    seeded_factory: sessionmaker,
    llm_gateway: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm_gateway("llm", FakeLLMProvider(text=plan_text([])))

    with caplog.at_level(logging.INFO, logger="app.services.agent_service"):
        client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    records = [record for record in caplog.records if "agent_invoke " in record.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "planner=llm" in message
    assert "fallback=false" in message
    assert FAKE_API_KEY not in caplog.text


def test_rag_document_evidence_is_untouched_by_the_llm_planner(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
    llm_gateway: Any,
) -> None:
    """The LLM plans; retrieval and evidence still come from the real tool."""
    hit = RAGSearchHit(
        content="F004 UnderVoltage: DC bus voltage fell below the min value.",
        document="PowerFlex_520_User_Manual.pdf",
        page=161,
        section="Chapter 4 Fault Codes",
        chunk_id="chunk-abc",
        score=0.4988,
        score_semantics="vector_cosine_distance",
        higher_is_better=False,
    )

    class _HitProvider(RAGProvider):
        provider_id = "test-hit"

        def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
            return RAGSearchResponse(query=query, hits=[hit], retrieval_mode="hybrid")

    monkeypatch.setattr("app.integrations.rag.factory.get_rag_provider", lambda: _HitProvider())
    llm_gateway(
        "llm",
        FakeLLMProvider(
            text=plan_text([{"tool_name": MANUAL_TOOL, "arguments": {"query": ALARM_QUERY}}])
        ),
    )

    payload = client.post(INVOKE_PATH, json={"query": ALARM_QUERY}).json()

    document = [item for item in payload["evidence"] if item["source_type"] == "document"]
    assert len(document) == 1
    assert document[0]["document"] == "PowerFlex_520_User_Manual.pdf"
    assert document[0]["tool_name"] == MANUAL_TOOL


# --------------------------------------------------------------------------- #
# 9. Import order is not load-bearing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "module",
    [
        "app.agent.graph",
        "app.agent.planners",
        "app.services",
        "app.services.agent_service",
    ],
)
def test_module_imports_as_the_first_application_import(module: str) -> None:
    """Every entry point must be importable as the process's first app import.

    The agent layer imports ``app.services.device_catalog`` while the services
    package exposes an agent-dependent symbol. If that symbol were imported
    eagerly, the services package would be both upstream and downstream of the
    agent package, and whether a module loaded would depend on which module the
    process reached first. Each case therefore runs in a fresh interpreter, where
    the module under test is genuinely the first one loaded.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr

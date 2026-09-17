"""Tests for the V0.9.0 production observability layer.

The suite pins six properties:

1. ``GET /metrics`` serves a Prometheus exposition document, and answers 404 when
   metrics are disabled rather than serving an empty one;
2. every instrumented stage of the pipeline records what actually ran: the
   request, the planner, each tool, the retrieval call and the provider call;
3. a normal negative result stays distinct from a failure, so an unknown device
   is ``not_found`` and a broken retrieval is ``unavailable``;
4. no label, log field or span attribute can carry user text, a credential or a
   high-cardinality identifier, and the label vocabularies cannot drift from the
   code that produces them;
5. structured JSON output is well formed, correlates by ``request_id``, and is
   scrubbed of secrets;
6. tracing is inert when it is switched off: nothing is imported, no exporter is
   constructed and no socket is opened.

Every test runs without a network, without a RAG checkout, without a real
dataset, without a collector and without a provider credential. The provider and
the retrieval engine are doubles defined here or in ``tests/fake_llm.py``.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.parser import Intent
from app.agent.planners.dispatch import _FAILURE_REASONS, PLANNER_LLM, PLANNER_RULE
from app.agent.planners.llm import LLMPlanner
from app.agent.planners.schema import PlannerError, PlannerErrorCode
from app.config import Settings, get_settings
from app.database import session as db_session
from app.database.init_db import init_db
from app.integrations.llm import (
    LLMCompletionRequest,
    LLMCompletionResult,
    LLMProvider,
    LLMProviderError,
)
from app.integrations.rag import (
    VECTOR_COSINE_DISTANCE,
    RAGProvider,
    RAGProviderError,
    RAGSearchHit,
    RAGSearchResponse,
)
from app.logging_config import configure_logging
from app.main import app
from app.observability import instrumentation
from app.observability.labels import (
    INTENT_LABELS,
    OTHER,
    PLANNER_LABELS,
    PROVIDER_LABELS,
    REASON_LABELS,
    STATUS_LABELS,
    TOKEN_TYPE_LABELS,
    TOOL_LABELS,
    bounded,
)
from app.observability.logging import (
    EVENT_REQUEST_COMPLETED,
    FIELD_ORDER,
    JsonFormatter,
    event_fields,
    log_event,
)
from app.observability.metrics import (
    FORBIDDEN_LABEL_NAMES,
    METRIC_SPECS,
    get_metrics,
    reset_metrics,
)
from app.observability.tracing import (
    SPAN_ATTRIBUTE_NAMES,
    configure_tracing,
    reset_tracing,
    span,
    span_attributes,
)
from app.observability.tracing import (
    is_enabled as tracing_is_enabled,
)
from app.tools.names import ToolName
from app.tools.registry import registry

METRICS_PATH = "/metrics"
INVOKE_PATH = "/agent/invoke"

DEVICE_TOOL = ToolName.GET_DEVICE_STATUS.value
MANUAL_TOOL = ToolName.SEARCH_MAINTENANCE_MANUAL.value

DEVICE_QUERY = "PLC-001 现在什么状态"
UNKNOWN_DEVICE_QUERY = "XYZ-999 现在什么状态"
ALARM_QUERY = "包装线PLC报警F0045怎么办"

#: A test-only sentinel. Shaped like no real credential on purpose, so a leak
#: cannot be mistaken for a live key and no scanner has to triage it.
FAKE_SECRET = "TEST_SECRET_DO_NOT_LEAK_OBSERVABILITY_KEY_2c71f0ae55b9"

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures and doubles
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _fresh_observability() -> Iterator[None]:
    """Give every test an empty registry and a tracing layer that is switched off.

    A counter is a process-lifetime quantity, so without this the assertions would
    depend on which tests ran first.
    """
    reset_metrics()
    reset_tracing()
    try:
        yield
    finally:
        reset_metrics()
        reset_tracing()


@pytest.fixture()
def seeded_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[sessionmaker]:
    """Point the default session factory at a seeded in-memory database.

    ``StaticPool`` is required because the route is declared with ``def`` and
    Starlette therefore serves it from a worker thread; an in-memory SQLite
    database is otherwise private to the connection that created it.
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


class _StubProvider(RAGProvider):
    """Retrieval double that returns a fixed hit list, or raises."""

    provider_id = "local"

    def __init__(
        self,
        hits: list[RAGSearchHit] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._hits = hits if hits is not None else []
        self._error = error
        self.calls: list[str] = []

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        self.calls.append(query)
        if self._error is not None:
            raise self._error
        return RAGSearchResponse(query=query, hits=self._hits, retrieval_mode="hybrid")


def _use_rag_provider(monkeypatch: pytest.MonkeyPatch, provider: RAGProvider) -> None:
    """Route the tool layer to ``provider`` for the duration of a test."""
    monkeypatch.setattr("app.integrations.rag.factory.get_rag_provider", lambda: provider)


MANUAL_HIT = RAGSearchHit(
    content="F004 UnderVoltage: DC bus voltage fell below the minimum value.",
    document="PowerFlex_520_User_Manual.pdf",
    page=161,
    section="Chapter 4 Fault Codes",
    chunk_id="chunk-59aa84554891f3fb566fcc20",
    score=0.4988,
    score_semantics=VECTOR_COSINE_DISTANCE,
    higher_is_better=False,
)


class _StubLLMProvider(LLMProvider):
    """Completion double with a production provider identifier.

    The identifier is ``openai_compatible`` rather than a test-only string so the
    assertion checks a label a real deployment also produces.
    """

    provider_id = "openai_compatible"

    def __init__(
        self,
        *,
        text: str = '{"intent": "device_status", "tool_calls": []}',
        error: Exception | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self._text = text
        self._error = error
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.requests: list[LLMCompletionRequest] = []

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResult:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return LLMCompletionResult(
            text=self._text,
            provider=self.provider_id,
            model="stub-model",
            latency_ms=1.0,
            finish_reason="stop",
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )


def _sample(name: str, **labels: str) -> float | None:
    """Return one sample from the live registry, or ``None`` when absent."""
    return get_metrics().registry.get_sample_value(name, labels)


def _metrics_text() -> str:
    """Return the rendered exposition document."""
    response = client.get(METRICS_PATH)
    assert response.status_code == 200
    return response.text


def _label_values(body: str) -> set[str]:
    """Return every label value present in an exposition document."""
    import re

    values: set[str] = set()
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        head = line.split("}", 1)[0] if "{" in line else ""
        values.update(re.findall(r'\w+="([^"]*)"', head))
    return values


def _label_names(body: str) -> set[str]:
    """Return every label name present in an exposition document."""
    import re

    names: set[str] = set()
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        head = line.split("}", 1)[0] if "{" in line else ""
        names.update(re.findall(r'(\w+)="', head))
    return names


def _structured_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the records this application emitted as structured events."""
    return [record for record in caplog.records if _event_fields(record)]


def _event_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Return a record's structured fields, or an empty mapping.

    The fields travel as a dynamic attribute on the record, so an accessor keeps
    the assertions type-checked instead of reaching through ``getattr`` at every
    call site.
    """
    fields = getattr(record, "event_fields", None)
    return fields if isinstance(fields, dict) else {}


def _event_name(record: logging.LogRecord) -> str:
    """Return a record's event name, defaulting to the unstructured marker."""
    name = getattr(record, "event_name", None)
    return name if isinstance(name, str) else "log"


# --------------------------------------------------------------------------- #
# 1. The /metrics endpoint
# --------------------------------------------------------------------------- #


def test_metrics_endpoint_serves_the_prometheus_exposition_format() -> None:
    response = client.get(METRICS_PATH)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" in response.headers["content-type"]
    body = response.text
    assert "# TYPE industrial_agent_requests_total counter" in body
    assert "# TYPE industrial_agent_request_duration_seconds histogram" in body


def test_metrics_endpoint_is_reachable_before_any_invocation() -> None:
    """A scrape target must answer on a freshly started process."""
    body = _metrics_text()

    assert "# HELP industrial_agent_requests_total" in body
    assert _sample("industrial_agent_requests_total") is None


def test_metrics_endpoint_documents_every_declared_metric() -> None:
    body = _metrics_text()

    for spec in METRIC_SPECS:
        assert f"# TYPE {spec.name} {spec.kind}" in body, spec.name
        assert f"# HELP {spec.name}" in body, spec.name


# --------------------------------------------------------------------------- #
# 2. Request counters and histograms
# --------------------------------------------------------------------------- #


def test_request_counter_and_histogram_increase_on_a_successful_invocation(
    seeded_factory: sessionmaker,
) -> None:
    response = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})
    assert response.status_code == 200

    labels = {"planner": "rule", "intent": "device_status", "status": "success"}
    assert _sample("industrial_agent_requests_total", **labels) == 1.0
    assert (
        _sample(
            "industrial_agent_request_duration_seconds_count",
            planner="rule",
            status="success",
        )
        == 1.0
    )
    # The histogram is fed the same wall-clock measurement the response reports,
    # so its sum must be within one rounding step of that latency.
    assert _sample(
        "industrial_agent_request_duration_seconds_sum",
        planner="rule",
        status="success",
    ) == pytest.approx(response.json()["latency_ms"] / 1000.0, abs=0.01)


def test_request_counter_is_incremented_per_invocation(seeded_factory: sessionmaker) -> None:
    for _ in range(3):
        client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    labels = {"planner": "rule", "intent": "device_status", "status": "success"}
    assert _sample("industrial_agent_requests_total", **labels) == 3.0


def test_a_failed_invocation_is_counted_as_error_not_success() -> None:
    def _explode(payload: Any) -> Any:
        raise RuntimeError(f"upstream refused: token={FAKE_SECRET}")

    app.dependency_overrides.clear()
    from app.api.routes.agent import get_agent_service
    from app.services import AgentService

    app.dependency_overrides[get_agent_service] = lambda: AgentService(runner=_explode)
    try:
        response = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert (
        _sample(
            "industrial_agent_requests_total",
            planner=OTHER,
            intent=OTHER,
            status="error",
        )
        == 1.0
    )
    # The failing path records the error outcome, not a success.
    assert (
        _sample(
            "industrial_agent_requests_total",
            planner=OTHER,
            intent=OTHER,
            status="success",
        )
        is None
    )


# --------------------------------------------------------------------------- #
# 3. Planner metrics
# --------------------------------------------------------------------------- #


def test_rule_planner_metrics_are_recorded(seeded_factory: sessionmaker) -> None:
    client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    assert (
        _sample(
            "industrial_agent_planner_duration_seconds_count",
            planner="rule",
            status="success",
        )
        == 1.0
    )
    assert _sample("industrial_agent_planner_failures_total", planner="rule", reason=OTHER) is None


def test_llm_planner_metrics_are_recorded_with_a_mock_provider() -> None:
    provider = _StubLLMProvider()
    planner = LLMPlanner(provider, "stub-model")
    settings = Settings(planner_mode="llm", llm_model="stub-model")

    from app.agent.planners.dispatch import dispatch_plan

    state: dict[str, Any] = {"query": "PLC-001 状态", "intent": Intent.DEVICE_STATUS.value}
    update = dispatch_plan(
        state,  # type: ignore[arg-type]
        rule_planner=lambda _s: {"required_tools": []},
        settings=settings,
        planner=planner,
    )

    assert update["planner_used"] == PLANNER_LLM
    assert (
        _sample(
            "industrial_agent_planner_duration_seconds_count",
            planner="llm",
            status="success",
        )
        == 1.0
    )
    assert (
        _sample(
            "industrial_agent_planner_failures_total",
            planner="llm",
            reason="provider_error",
        )
        is None
    )


def test_planner_failure_metric_uses_a_closed_reason_enumeration() -> None:
    provider = _StubLLMProvider(error=LLMProviderError(f"HTTP 502 token={FAKE_SECRET}"))
    planner = LLMPlanner(provider, "stub-model")
    settings = Settings(planner_mode="llm", llm_model="stub-model")

    from app.agent.planners.dispatch import dispatch_plan

    with pytest.raises(PlannerError) as failure:
        dispatch_plan(
            {"query": "PLC-001 状态"},  # type: ignore[arg-type]
            rule_planner=lambda _s: {"required_tools": []},
            settings=settings,
            planner=planner,
        )

    assert failure.value.code is PlannerErrorCode.LLM_PROVIDER_ERROR
    assert (
        _sample(
            "industrial_agent_planner_failures_total",
            planner="llm",
            reason="provider_error",
        )
        == 1.0
    )
    # The provider message is never a label value.
    assert FAKE_SECRET not in _metrics_text()


def test_planner_failure_during_construction_is_classified_as_configuration() -> None:
    """A missing endpoint or model is a misconfiguration, not a provider outage.

    Both surface as ``LLM_PROVIDER_ERROR``, so the reason cannot be derived from
    the code alone. It is derived from the phase the failure happened in.
    """
    settings = Settings(planner_mode="llm", llm_model="")

    from app.agent.planners.dispatch import dispatch_plan

    with pytest.raises(PlannerError):
        dispatch_plan(
            {"query": "PLC-001 状态"},  # type: ignore[arg-type]
            rule_planner=lambda _s: {"required_tools": []},
            settings=settings,
        )

    assert (
        _sample(
            "industrial_agent_planner_failures_total",
            planner="llm",
            reason="configuration",
        )
        == 1.0
    )


def test_invalid_planner_output_maps_to_invalid_output_reason() -> None:
    provider = _StubLLMProvider(text="not json at all")
    planner = LLMPlanner(provider, "stub-model")
    settings = Settings(planner_mode="llm", llm_model="stub-model")

    from app.agent.planners.dispatch import dispatch_plan

    with pytest.raises(PlannerError):
        dispatch_plan(
            {"query": "PLC-001 状态"},  # type: ignore[arg-type]
            rule_planner=lambda _s: {"required_tools": []},
            settings=settings,
            planner=planner,
        )

    assert (
        _sample(
            "industrial_agent_planner_failures_total",
            planner="llm",
            reason="invalid_output",
        )
        == 1.0
    )


# --------------------------------------------------------------------------- #
# 4. Tool metrics
# --------------------------------------------------------------------------- #


def test_tool_metrics_record_the_executed_tool(seeded_factory: sessionmaker) -> None:
    client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    assert (
        _sample(
            "industrial_agent_tool_calls_total",
            tool=DEVICE_TOOL,
            status="success",
        )
        == 1.0
    )
    assert (
        _sample(
            "industrial_agent_tool_duration_seconds_count",
            tool=DEVICE_TOOL,
            status="success",
        )
        == 1.0
    )


def test_tool_error_metric_records_a_raising_tool(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**_: Any) -> Any:
        raise RuntimeError("device source exploded")

    monkeypatch.setattr(registry.get(DEVICE_TOOL), "func", _raise)

    response = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    assert response.status_code == 500
    assert (
        _sample(
            "industrial_agent_tool_calls_total",
            tool=DEVICE_TOOL,
            status="error",
        )
        == 1.0
    )
    # A tool that raised is not also counted as a success.
    assert _sample("industrial_agent_tool_calls_total", tool=DEVICE_TOOL, status="success") is None


def test_unknown_device_is_recorded_as_not_found_rather_than_error(
    seeded_factory: sessionmaker,
) -> None:
    """A missing device is a normal answer, and the metric must say so."""
    response = client.post(INVOKE_PATH, json={"query": UNKNOWN_DEVICE_QUERY})

    assert response.status_code == 200
    assert (
        _sample(
            "industrial_agent_tool_calls_total",
            tool=DEVICE_TOOL,
            status="not_found",
        )
        == 1.0
    )
    assert (
        _sample(
            "industrial_agent_tool_calls_total",
            tool=DEVICE_TOOL,
            status="error",
        )
        is None
    )
    # The invocation itself succeeded: nothing failed, the device is absent.
    assert (
        _sample(
            "industrial_agent_requests_total",
            planner="rule",
            intent="device_status",
            status="success",
        )
        == 1.0
    )


# --------------------------------------------------------------------------- #
# 5. RAG metrics
# --------------------------------------------------------------------------- #


def test_rag_metrics_record_a_successful_retrieval(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_rag_provider(monkeypatch, _StubProvider(hits=[MANUAL_HIT]))

    client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    assert _sample("industrial_agent_rag_requests_total", provider="local", status="success") == 1.0
    assert (
        _sample(
            "industrial_agent_rag_duration_seconds_count",
            provider="local",
            status="success",
        )
        == 1.0
    )
    assert (
        _sample(
            "industrial_agent_tool_calls_total",
            tool=MANUAL_TOOL,
            status="success",
        )
        == 1.0
    )


def test_rag_metrics_record_an_empty_retrieval_as_not_found(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A knowledge base with nothing on the subject is not a broken deployment."""
    _use_rag_provider(monkeypatch, _StubProvider(hits=[]))

    response = client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    assert response.status_code == 200
    assert (
        _sample(
            "industrial_agent_rag_requests_total",
            provider="local",
            status="not_found",
        )
        == 1.0
    )
    assert (
        _sample(
            "industrial_agent_rag_requests_total",
            provider="local",
            status="error",
        )
        is None
    )


def test_rag_unavailable_is_recorded_apart_from_an_empty_result(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_rag_provider(monkeypatch, _StubProvider(error=RAGProviderError("index missing")))

    client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    assert (
        _sample(
            "industrial_agent_rag_requests_total",
            provider="local",
            status="unavailable",
        )
        == 1.0
    )
    # A retrieval that could not run is not an empty retrieval.
    assert (
        _sample("industrial_agent_rag_requests_total", provider="local", status="not_found") is None
    )


def test_rag_unavailable_is_attributed_to_an_unknown_provider_when_none_can_be_built(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail() -> RAGProvider:
        raise RAGProviderError("RAG repo root is not configured")

    monkeypatch.setattr("app.integrations.rag.factory.get_rag_provider", _fail)

    client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    assert (
        _sample(
            "industrial_agent_rag_requests_total",
            provider="unknown",
            status="unavailable",
        )
        == 1.0
    )


# --------------------------------------------------------------------------- #
# 6. LLM metrics and token accounting
# --------------------------------------------------------------------------- #


def test_llm_metrics_record_a_provider_call() -> None:
    provider = _StubLLMProvider()
    planner = LLMPlanner(provider, "stub-model")

    from app.agent.planners.dispatch import dispatch_plan

    dispatch_plan(
        {"query": "PLC-001 状态"},  # type: ignore[arg-type]
        rule_planner=lambda _s: {"required_tools": []},
        settings=Settings(planner_mode="llm", llm_model="stub-model"),
        planner=planner,
    )

    assert (
        _sample(
            "industrial_agent_llm_requests_total",
            provider="openai_compatible",
            status="success",
        )
        == 1.0
    )
    assert (
        _sample(
            "industrial_agent_llm_duration_seconds_count",
            provider="openai_compatible",
            status="success",
        )
        == 1.0
    )


def test_llm_provider_failure_is_recorded_as_unavailable() -> None:
    provider = _StubLLMProvider(error=LLMProviderError("LLM endpoint returned HTTP 502"))
    planner = LLMPlanner(provider, "stub-model")

    from app.agent.planners.dispatch import dispatch_plan

    with pytest.raises(PlannerError):
        dispatch_plan(
            {"query": "PLC-001 状态"},  # type: ignore[arg-type]
            rule_planner=lambda _s: {"required_tools": []},
            settings=Settings(planner_mode="llm", llm_model="stub-model"),
            planner=planner,
        )

    assert (
        _sample(
            "industrial_agent_llm_requests_total",
            provider="openai_compatible",
            status="unavailable",
        )
        == 1.0
    )


def test_token_counter_records_endpoint_reported_usage() -> None:
    provider = _StubLLMProvider(prompt_tokens=1284, completion_tokens=57)
    planner = LLMPlanner(provider, "stub-model")

    from app.agent.planners.dispatch import dispatch_plan

    dispatch_plan(
        {"query": "PLC-001 状态"},  # type: ignore[arg-type]
        rule_planner=lambda _s: {"required_tools": []},
        settings=Settings(planner_mode="llm", llm_model="stub-model"),
        planner=planner,
    )

    assert (
        _sample(
            "industrial_agent_llm_tokens_total",
            provider="openai_compatible",
            type="prompt",
        )
        == 1284.0
    )
    assert (
        _sample(
            "industrial_agent_llm_tokens_total",
            provider="openai_compatible",
            type="completion",
        )
        == 57.0
    )


def test_token_counter_is_absent_when_the_provider_reports_no_usage() -> None:
    """A provider that sends no ``usage`` block contributes no token sample.

    Zero would be a measurement claim the endpoint never made.
    """
    provider = _StubLLMProvider(prompt_tokens=None, completion_tokens=None)
    planner = LLMPlanner(provider, "stub-model")

    from app.agent.planners.dispatch import dispatch_plan

    dispatch_plan(
        {"query": "PLC-001 状态"},  # type: ignore[arg-type]
        rule_planner=lambda _s: {"required_tools": []},
        settings=Settings(planner_mode="llm", llm_model="stub-model"),
        planner=planner,
    )

    assert (
        _sample("industrial_agent_llm_tokens_total", provider="openai_compatible", type="prompt")
        is None
    )
    assert (
        _sample(
            "industrial_agent_llm_tokens_total", provider="openai_compatible", type="completion"
        )
        is None
    )


def test_openai_compatible_provider_extracts_usage_from_the_envelope() -> None:
    from app.integrations.llm.openai_compatible import OpenAICompatibleProvider

    envelope = {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 41},
    }
    assert OpenAICompatibleProvider._extract_usage(envelope) == (900, 41)

    assert OpenAICompatibleProvider._extract_usage({"choices": []}) == (None, None)
    assert OpenAICompatibleProvider._extract_usage({"usage": None}) == (None, None)
    assert OpenAICompatibleProvider._extract_usage({"usage": {"prompt_tokens": "many"}}) == (
        None,
        None,
    )
    # ``True`` is an ``int`` in Python and must not be counted as one token.
    assert OpenAICompatibleProvider._extract_usage({"usage": {"prompt_tokens": True}}) == (
        None,
        None,
    )
    assert OpenAICompatibleProvider._extract_usage({"usage": {"prompt_tokens": -5}}) == (None, None)


# --------------------------------------------------------------------------- #
# 7. Configuration
# --------------------------------------------------------------------------- #


def test_metrics_disabled_returns_404_and_records_nothing(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METRICS_ENABLED", "false")
    get_settings.cache_clear()

    try:
        response = client.get(METRICS_PATH)
        assert response.status_code == 404
        assert "METRICS_ENABLED" in response.text

        client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

        # Recording is skipped, so the registry stays empty even though the agent
        # answered normally.
        assert (
            _sample(
                "industrial_agent_requests_total",
                planner="rule",
                intent="device_status",
                status="success",
            )
            is None
        )
        assert (
            _sample(
                "industrial_agent_tool_calls_total",
                tool=DEVICE_TOOL,
                status="success",
            )
            is None
        )
    finally:
        get_settings.cache_clear()


def test_metrics_default_is_enabled() -> None:
    assert Settings().metrics_enabled is True
    assert Settings().log_format == "text"
    assert Settings().otel_enabled is False
    assert Settings().otel_exporter_otlp_endpoint == ""


def test_the_agent_still_answers_with_metrics_disabled(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observability must not be able to break the service it observes."""
    monkeypatch.setenv("METRICS_ENABLED", "false")
    get_settings.cache_clear()
    try:
        response = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json()["tools_called"] == [DEVICE_TOOL]


# --------------------------------------------------------------------------- #
# 8. Structured logging
# --------------------------------------------------------------------------- #


def _record_for(event: str, **fields: Any) -> logging.LogRecord:
    """Build the record ``log_event`` would emit, without going through logging."""
    record = logging.LogRecord(
        name="app.observability",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="test event",
        args=(),
        exc_info=None,
    )
    record.event_fields = event_fields(event, **fields)
    record.event_name = event
    return record


def test_json_formatter_emits_one_object_per_line() -> None:
    formatter = JsonFormatter()
    payload = json.loads(
        formatter.format(
            _record_for(
                EVENT_REQUEST_COMPLETED,
                request_id="9c1e0f1a-5c4b-4b4c-9a1e-1a2b3c4d5e6f",
                planner="rule",
                intent="device_status",
                status="success",
                duration_ms=12.5,
            )
        )
    )

    assert payload["event"] == EVENT_REQUEST_COMPLETED
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "9c1e0f1a-5c4b-4b4c-9a1e-1a2b3c4d5e6f"
    assert payload["planner"] == "rule"
    assert payload["intent"] == "device_status"
    assert payload["status"] == "success"
    assert payload["duration_ms"] == 12.5
    assert payload["timestamp"]
    assert "\n" not in json.dumps(payload)


def test_json_fields_are_a_subset_of_the_declared_schema() -> None:
    payload = event_fields(
        EVENT_REQUEST_COMPLETED,
        request_id="r",
        planner="rule",
        undeclared_field="must not appear",
    )

    assert set(payload) <= set(FIELD_ORDER)
    assert "undeclared_field" not in payload


def test_field_order_covers_every_documented_field() -> None:
    assert set(FIELD_ORDER) >= {
        "timestamp",
        "level",
        "event",
        "request_id",
        "planner",
        "intent",
        "tool",
        "provider",
        "status",
        "duration_ms",
        "error_type",
    }


def test_json_output_keeps_every_event_name_this_application_emits() -> None:
    """The event vocabulary is closed and enumerable."""
    from app.observability import logging as events

    names = {
        value
        for name, value in vars(events).items()
        if name.startswith("EVENT_") and isinstance(value, str)
    }
    assert names == {
        "agent.request.started",
        "agent.request.completed",
        "planner.started",
        "planner.completed",
        "planner.failed",
        "tool.started",
        "tool.completed",
        "tool.failed",
        "rag.started",
        "rag.completed",
        "rag.unavailable",
        "llm.started",
        "llm.completed",
        "llm.failed",
    }


def test_text_format_is_unchanged_by_the_structured_logging_addition() -> None:
    """``LOG_FORMAT=text`` must still be the byte-identical historical line."""
    from app.observability.logging import TEXT_LOG_FORMAT

    assert TEXT_LOG_FORMAT == "%(asctime)s %(levelname)s %(name)s %(message)s"

    record = logging.LogRecord(
        name="app.services.agent_service",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="agent_invoke request_id=%s success=true",
        args=("abc",),
        exc_info=None,
    )
    rendered = logging.Formatter(TEXT_LOG_FORMAT).format(record)
    assert "agent_invoke request_id=abc success=true" in rendered
    assert not rendered.startswith("{")


def test_json_logging_can_be_selected_and_restored() -> None:
    app_logger = logging.getLogger("app")
    try:
        configure_logging("INFO", "json")
        assert isinstance(app_logger.handlers[0].formatter, JsonFormatter)

        configure_logging("INFO", "text")
        assert not isinstance(app_logger.handlers[0].formatter, JsonFormatter)
    finally:
        configure_logging(get_settings().log_level, "text")


def test_unknown_log_format_falls_back_to_text() -> None:
    from app.observability.logging import build_formatter

    assert not isinstance(build_formatter("yaml"), JsonFormatter)


# --------------------------------------------------------------------------- #
# 9. Privacy and redaction
# --------------------------------------------------------------------------- #


def test_secret_bearing_values_are_redacted_in_a_structured_event() -> None:
    formatter = JsonFormatter()
    record = _record_for(
        "tool.failed",
        error_type="RuntimeError",
        provider=f"Bearer {FAKE_SECRET}",
        reason=f"api_key={FAKE_SECRET}",
    )

    rendered = formatter.format(record)

    assert FAKE_SECRET not in rendered
    assert "<redacted>" in rendered


def test_json_output_scrubs_a_query_embedded_in_a_legacy_message() -> None:
    """A text-format line can carry the query; the JSON shape must not."""
    record = logging.LogRecord(
        name="app.services.agent_service",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="agent_invoke request_id=%s query=%r",
        args=("abc", "PLC-001 现在什么状态"),
        exc_info=None,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert "PLC-001" not in payload["message"]
    assert payload["message"].endswith("query=<redacted>")


def test_json_output_scrubs_a_multiline_query() -> None:
    record = logging.LogRecord(
        name="app.services.agent_service",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="planner_fallback reason=%s query=%r",
        args=("LLM_TIMEOUT", "PLC-001\n状态\t检查"),
        exc_info=None,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert "PLC-001" not in payload["message"]
    assert "状态" not in payload["message"]


def test_structured_fields_are_truncated_and_typed_safely() -> None:
    payload = event_fields("tool.failed", error_type="X" * 500, count={"nested": "mapping"})

    assert payload["error_type"].endswith("...<truncated>")
    assert len(payload["error_type"]) <= 200 + len("...<truncated>")
    # A nested structure becomes its type name rather than being serialized.
    assert payload["count"] == "dict"


def test_structured_events_never_carry_the_query_or_the_prompt(
    seeded_factory: sessionmaker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.observability"):
        client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    records = _structured_records(caplog)
    assert records, "expected structured events"

    for record in records:
        serialized = json.dumps(_event_fields(record), ensure_ascii=False)
        assert DEVICE_QUERY not in serialized
        assert "PLC-001" not in serialized
        assert "状态" not in serialized


def test_request_id_correlates_every_structured_event_of_an_invocation(
    seeded_factory: sessionmaker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.observability"):
        response = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    request_id = response.json()["request_id"]
    completed = [
        record
        for record in _structured_records(caplog)
        if _event_name(record) == "agent.request.completed"
    ]

    assert len(completed) == 1
    assert _event_fields(completed[0])["request_id"] == request_id


def test_request_id_is_never_a_metric_label(
    seeded_factory: sessionmaker,
) -> None:
    response = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})
    request_id = response.json()["request_id"]

    assert request_id not in _metrics_text()
    assert "request_id" not in _label_names(_metrics_text())


def test_the_raw_query_never_becomes_a_metric_label(
    seeded_factory: sessionmaker,
) -> None:
    client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    body = _metrics_text()

    assert DEVICE_QUERY not in body
    assert "PLC-001" not in body
    assert "状态" not in body


def test_the_device_identifier_never_becomes_a_metric_label(
    seeded_factory: sessionmaker,
) -> None:
    client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    body = _metrics_text()

    assert "PLC-001" not in body
    assert "device_id" not in _label_names(body)
    assert "equipment_id" not in _label_names(body)


def test_a_document_name_never_becomes_a_metric_label(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_rag_provider(monkeypatch, _StubProvider(hits=[MANUAL_HIT]))

    client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    body = _metrics_text()

    assert MANUAL_HIT.document is not None
    assert MANUAL_HIT.document not in body
    assert "PowerFlex" not in body
    assert MANUAL_HIT.chunk_id is not None
    assert MANUAL_HIT.chunk_id not in body


# --------------------------------------------------------------------------- #
# 10. Cardinality safety
# --------------------------------------------------------------------------- #


def test_no_declared_metric_uses_a_forbidden_label_name() -> None:
    for spec in METRIC_SPECS:
        overlap = set(spec.labelnames) & FORBIDDEN_LABEL_NAMES
        assert not overlap, f"{spec.name} uses forbidden label(s): {sorted(overlap)}"


def test_every_metric_label_has_a_frozen_vocabulary_of_matching_arity() -> None:
    """A label without a vocabulary is how a high-cardinality value gets in."""
    for spec in METRIC_SPECS:
        assert len(spec.vocabulary) == len(spec.labelnames), spec.name


def test_only_bounded_label_names_appear_in_a_rendered_registry(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every stage, then audit the label names actually in use."""
    _use_rag_provider(monkeypatch, _StubProvider(hits=[MANUAL_HIT]))

    client.post(INVOKE_PATH, json={"query": ALARM_QUERY})
    client.post(INVOKE_PATH, json={"query": UNKNOWN_DEVICE_QUERY})

    allowed = {name for spec in METRIC_SPECS for name in spec.labelnames} | {"le"}
    assert _label_names(_metrics_text()) <= allowed


def test_bounded_maps_anything_outside_the_vocabulary_to_other() -> None:
    assert bounded("rule", PLANNER_LABELS) == "rule"
    assert bounded("a-new-planner", PLANNER_LABELS) == OTHER
    assert bounded(None, PLANNER_LABELS) == OTHER
    assert bounded("", PLANNER_LABELS) == OTHER
    assert bounded(Intent.DEVICE_STATUS, INTENT_LABELS) == "device_status"


def test_every_vocabulary_closed_over_external_input_contains_the_fallback() -> None:
    """A vocabulary fed from application state must accept an unknown value."""
    for vocabulary in (
        PLANNER_LABELS,
        INTENT_LABELS,
        STATUS_LABELS,
        TOOL_LABELS,
        PROVIDER_LABELS,
        REASON_LABELS,
    ):
        assert OTHER in vocabulary
        assert bounded("definitely-not-a-member", vocabulary) == OTHER


def test_token_type_vocabulary_is_exactly_the_two_documented_directions() -> None:
    """No fallback here, and that is deliberate.

    Every other vocabulary absorbs an unexpected value as ``other`` because its
    input comes from state, a tool payload or an exception. The token direction is
    iterated from a literal pair in :func:`app.observability.instrumentation.llm_tokens`,
    so a value outside the pair is a code defect, not a new dimension. The
    vocabulary stays at exactly two members so the metric cannot acquire a third
    series, and the drift test above fails first if the literal changes.
    """
    assert TOKEN_TYPE_LABELS == {"prompt", "completion"}
    assert OTHER not in TOKEN_TYPE_LABELS


def test_vocabularies_cover_the_values_the_application_actually_produces() -> None:
    """Drift guard: a renamed enum member fails here rather than on a dashboard."""
    from app.tools.names import TOOL_NAMES

    assert set(TOOL_NAMES) <= TOOL_LABELS
    assert {member.value for member in Intent} <= INTENT_LABELS

    assert bounded(PLANNER_RULE, PLANNER_LABELS) == PLANNER_RULE
    assert bounded(PLANNER_LLM, PLANNER_LABELS) == PLANNER_LLM

    assert set(_FAILURE_REASONS.values()) <= REASON_LABELS
    assert {"configuration", "unknown"} <= REASON_LABELS

    assert {"local", "http", "openai_compatible"} <= PROVIDER_LABELS
    assert {"success", "not_found", "unavailable", "error"} <= STATUS_LABELS
    assert {"prompt", "completion"} <= TOKEN_TYPE_LABELS


def test_every_metric_the_instrumentation_writes_exists_in_the_registry() -> None:
    declared = {spec.name for spec in METRIC_SPECS}
    used = {
        value
        for name, value in vars(instrumentation).items()
        if name.startswith("M_") and isinstance(value, str)
    }

    assert (
        used == declared
    ), f"declared but unused: {declared - used}, used but undeclared: {used - declared}"


def test_recording_an_unknown_label_value_does_not_create_a_new_series(
    seeded_factory: sessionmaker,
) -> None:
    instrumentation.request_completed(
        planner="a-planner-that-does-not-exist",
        intent="an-intent-that-does-not-exist",
        status="success",
        duration_ms=1.0,
    )

    assert (
        _sample(
            "industrial_agent_requests_total",
            planner=OTHER,
            intent=OTHER,
            status="success",
        )
        == 1.0
    )


# --------------------------------------------------------------------------- #
# 11. Tracing
# --------------------------------------------------------------------------- #


def test_tracing_is_disabled_by_default_and_imports_no_sdk() -> None:
    """The decisive check runs in a fresh interpreter.

    Within the test session other tests import the SDK, so an in-process module
    check cannot prove that a deployment which never enables tracing never imports
    it.
    """
    program = (
        "import sys;"
        "from app.observability.tracing import configure_tracing, is_enabled;"
        "assert configure_tracing(enabled=False, service_name='x') is False;"
        "assert is_enabled() is False;"
        "print('opentelemetry' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_disabled_tracing_builds_no_exporter_and_opens_no_socket(
    seeded_factory: sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove a disabled tracer constructs no exporter and resolves no host.

    The instruments are deliberately narrow. Blocking ``socket.socket`` or
    ``socket.connect`` outright would also break the test client, whose event loop
    builds a local ``socketpair`` to run the ASGI application; that is not network
    activity and it is not what this test is about. What it checks is the decisive
    fact: with tracing disabled the exporter and its span processor are never
    constructed, and nothing resolves a host name.
    """

    def _refuse_exporter(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an OTLP exporter was built while tracing is disabled")

    def _refuse_processor(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a span processor was built while tracing is disabled")

    def _refuse_dns(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a host name was resolved while tracing is disabled")

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        _refuse_exporter,
    )
    monkeypatch.setattr("opentelemetry.sdk.trace.export.BatchSpanProcessor", _refuse_processor)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse_dns)

    configure_tracing(
        enabled=False,
        service_name="industrial-maintenance-agent",
        otlp_endpoint="http://collector.invalid:4318/v1/traces",
    )

    response = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    assert response.status_code == 200
    assert tracing_is_enabled() is False


def test_enabling_tracing_without_an_endpoint_builds_no_exporter() -> None:
    """An empty endpoint keeps spans in the process; no collector is assumed."""
    assert (
        configure_tracing(
            enabled=True,
            service_name="industrial-maintenance-agent",
            otlp_endpoint="",
        )
        is True
    )
    assert tracing_is_enabled() is True


def test_span_attributes_are_allow_listed() -> None:
    attributes = span_attributes(
        **{
            "tool.name": DEVICE_TOOL,
            "status": "success",
            "query": DEVICE_QUERY,
            "prompt": "the whole system prompt",
            "answer": "the whole answer",
            "document": MANUAL_HIT.document,
            "request_id": "abc",
        }
    )

    assert attributes == {"tool.name": DEVICE_TOOL, "status": "success"}
    assert set(attributes) <= SPAN_ATTRIBUTE_NAMES


def test_span_redacts_a_secret_that_reaches_an_allowed_attribute() -> None:
    attributes = span_attributes(status=f"Bearer {FAKE_SECRET}")

    assert FAKE_SECRET not in json.dumps(attributes)


def test_a_span_never_receives_the_query_or_the_prompt() -> None:
    """The span adapter drops anything outside the allow-list."""
    captured: dict[str, Any] = {}

    class _RecordingSpan:
        def set_attribute(self, name: str, value: Any) -> None:
            captured[name] = value

        def record_error(self, error_type: str, message: str | None = None) -> None:
            captured["error"] = error_type

        def set_status(self, status: str) -> None:
            captured["status"] = status

    from app.observability.tracing import _OtelSpanAdapter

    monkeypatch_target = _OtelSpanAdapter(_RecordingSpan())
    monkeypatch_target.set_attribute("query", DEVICE_QUERY)
    monkeypatch_target.set_attribute("prompt", "system prompt")
    monkeypatch_target.set_attribute("status", "success")
    monkeypatch_target.record_error("RuntimeError", f"token={FAKE_SECRET}")

    assert "query" not in captured
    assert "prompt" not in captured
    assert captured["status"] == "success"
    assert FAKE_SECRET not in json.dumps(captured)


def test_the_span_helper_is_usable_while_tracing_is_disabled() -> None:
    reset_tracing()

    with span("agent.invoke", **{"planner.type": "rule"}) as active:
        active.set_attribute("status", "success")
        active.record_error("RuntimeError")

    assert tracing_is_enabled() is False


def test_the_pipeline_still_answers_with_tracing_enabled(
    seeded_factory: sessionmaker,
) -> None:
    """Instrumentation must not change the response when tracing is on."""
    configure_tracing(enabled=True, service_name="industrial-maintenance-agent")

    response = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    assert response.status_code == 200
    assert response.json()["tools_called"] == [DEVICE_TOOL]


def test_truncated_and_redacted_before_leaving_the_process_is_pure() -> None:
    """``sanitize`` must not mutate its input or leak the original value."""
    from app.observability.logging import sanitize

    original = f"Authorization: Bearer {FAKE_SECRET}"
    result = sanitize(original)

    # The expectation is assembled from parts so the source does not carry a
    # credential-shaped literal, which would otherwise be reported by the
    # repository's own secret scan on every run.
    expected = "Authorization: " + "Bearer " + "<redacted>"

    assert result != original
    assert result == expected
    assert original.endswith(FAKE_SECRET)


def test_observability_events_are_logged_under_the_application_namespace(
    seeded_factory: sessionmaker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.observability"):
        client.post(INVOKE_PATH, json={"query": DEVICE_QUERY})

    assert all(record.name.startswith("app.") for record in _structured_records(caplog)), [
        record.name for record in _structured_records(caplog)
    ]


def test_log_event_emits_a_record_carrying_the_declared_fields() -> None:
    logger = logging.getLogger("app.observability.test")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        log_event(
            logger,
            logging.INFO,
            "tool.completed",
            "tool_completed tool=get_device_status",
            tool=DEVICE_TOOL,
            status="success",
            duration_ms=3.2,
        )
    finally:
        logger.removeHandler(handler)

    assert len(records) == 1
    assert _event_name(records[0]) == "tool.completed"
    assert _event_fields(records[0])["tool"] == DEVICE_TOOL
    assert _event_fields(records[0])["status"] == "success"

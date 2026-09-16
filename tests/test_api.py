"""Tests for the V0.4.2 HTTP surface.

The suite pins five things the API layer must guarantee:

1. the meta probes still answer as before, and the HTTP surface exposes nothing
   beyond the documented routes;
2. ``POST /agent/invoke`` returns the declared contract, with a fresh uuid
   ``request_id`` and a measured ``latency_ms``;
3. RAG retrieval surfaces as ``source_type=document`` evidence, score semantics
   included;
4. an out-of-range query is rejected with 422, and a blank query is rejected
   rather than passed to the parser;
5. a pipeline failure becomes a structured 500 whose body carries a stable code
   and the ``request_id``, and never a traceback, an exception class name or a
   secret. The same secret must not reach the log either.

The knowledge base is always stubbed here. These tests describe transport
behaviour, so they must not depend on a RAG checkout being present, on a built
index, or on any credential.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes.agent import get_agent_service
from app.api.schemas.agent import AGENT_QUERY_MAX_LENGTH, AGENT_QUERY_MIN_LENGTH
from app.database import session as db_session
from app.database.init_db import init_db
from app.integrations.rag import (
    VECTOR_COSINE_DISTANCE,
    RAGProvider,
    RAGProviderError,
    RAGSearchHit,
    RAGSearchResponse,
)
from app.logging_config import configure_logging
from app.main import app
from app.services import AGENT_INVOCATION_FAILED, AgentInvocationError, AgentService
from app.tools.names import ToolName

DEVICE_TOOL = ToolName.GET_DEVICE_STATUS.value
ALARM_TOOL = ToolName.QUERY_ALARM_CODE.value
MANUAL_TOOL = ToolName.SEARCH_MAINTENANCE_MANUAL.value

DEVICE_QUERY = "PLC-001 现在什么状态"
ALARM_QUERY = "包装线PLC报警F0045怎么办"

INVOKE_PATH = "/agent/invoke"

#: The nine fields the response contract requires.
REQUIRED_RESPONSE_FIELDS = {
    "request_id",
    "query",
    "intent",
    "equipment_id",
    "alarm_code",
    "tools_called",
    "answer",
    "evidence",
    "latency_ms",
}

#: A test-only sentinel. It must never appear in a body or a log. The value is
#: deliberately not shaped like any real provider credential.
FAKE_SECRET = "TEST_SECRET_DO_NOT_LEAK_HTTP_ERROR_TOKEN_5b8e13fa92c4"

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures and doubles
# --------------------------------------------------------------------------- #


@pytest.fixture()
def seeded_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[sessionmaker]:
    """Point the default session factory at a seeded in-memory database.

    ``StaticPool`` is required here and not in the graph-level tests. The route
    is declared with ``def``, so Starlette runs it in a worker thread, and an
    in-memory SQLite database is otherwise private to the connection that
    created it: the worker thread would see an empty schema. Pinning one shared
    connection keeps the database visible from whichever thread serves the
    request. Production uses a file-backed database and needs no such pin.
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
    """Provider double returning a fixed hit list."""

    provider_id = "stub"

    def __init__(self, hits: list[RAGSearchHit] | None = None) -> None:
        self._hits = hits if hits is not None else []

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        return RAGSearchResponse(query=query, hits=self._hits, retrieval_mode="hybrid")


def _use_provider(monkeypatch: pytest.MonkeyPatch, provider: RAGProvider) -> None:
    """Route the tool layer to ``provider`` for the duration of a test."""
    monkeypatch.setattr("app.integrations.rag.factory.get_rag_provider", lambda: provider)


@pytest.fixture(autouse=True)
def stub_knowledge_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every API test to a deterministic, empty knowledge base."""
    _use_provider(monkeypatch, _StubProvider())


MANUAL_HIT = RAGSearchHit(
    content="F004 UnderVoltage: DC bus voltage fell below the min value.",
    document="PowerFlex_520_User_Manual.pdf",
    page=161,
    section="Chapter 4 Fault Codes",
    chunk_id="chunk-59aa84554891f3fb566fcc20",
    score=0.4988,
    score_semantics=VECTOR_COSINE_DISTANCE,
    higher_is_better=False,
)


@pytest.fixture()
def graph_override() -> Iterator[Callable[[Any], None]]:
    """Install a graph runner double for the duration of one test."""

    def _install(runner: Any) -> None:
        app.dependency_overrides[get_agent_service] = lambda: AgentService(runner=runner)

    yield _install
    app.dependency_overrides.clear()


def _exploding_runner(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Runner that fails the way an upstream outage would."""
    raise RuntimeError(
        f"upstream refused the request: token={FAKE_SECRET} url=https://user:pw@example.invalid"
    )


# --------------------------------------------------------------------------- #
# 1. Meta endpoints and the shape of the HTTP surface
# --------------------------------------------------------------------------- #


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint_reports_the_current_version() -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.8.1"


def test_http_surface_exposes_only_the_documented_routes() -> None:
    """The workflow is reachable through ``/agent/invoke`` and nowhere else."""
    assert set(app.openapi()["paths"]) == {"/", "/health", INVOKE_PATH}


def _resolve_schema(node: dict[str, Any]) -> dict[str, Any]:
    """Resolve a ``$ref`` produced by FastAPI into its component schema."""
    reference = node.get("$ref")
    if not reference:
        return node
    name = reference.rsplit("/", 1)[-1]
    return app.openapi()["components"]["schemas"][name]


def test_openapi_documents_the_query_bounds_and_response_contract() -> None:
    post = app.openapi()["paths"][INVOKE_PATH]["post"]
    request_model = _resolve_schema(post["requestBody"]["content"]["application/json"]["schema"])

    query_field = request_model["properties"]["query"]
    assert query_field["minLength"] == AGENT_QUERY_MIN_LENGTH
    assert query_field["maxLength"] == AGENT_QUERY_MAX_LENGTH
    assert "query" in request_model["required"]

    response_schema = post["responses"]["200"]["content"]["application/json"]["schema"]
    response_model = _resolve_schema(response_schema)
    assert REQUIRED_RESPONSE_FIELDS <= set(response_model["properties"])
    assert "500" in post["responses"]


# --------------------------------------------------------------------------- #
# 2. A normal invocation
# --------------------------------------------------------------------------- #


def test_invoke_returns_the_declared_contract(seeded_factory: sessionmaker) -> None:
    response = client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    assert response.status_code == 200
    payload = response.json()
    assert REQUIRED_RESPONSE_FIELDS <= set(payload)
    assert payload["query"] == ALARM_QUERY
    assert payload["intent"] == "alarm_diagnosis"
    assert payload["equipment_id"] == "PLC-001"
    assert payload["alarm_code"] == "F0045"
    assert payload["tools_called"] == [DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL]
    assert "PLC-001" in payload["answer"]
    assert isinstance(payload["latency_ms"], int | float)
    assert payload["latency_ms"] >= 0


def test_request_id_is_a_fresh_uuid_per_call(seeded_factory: sessionmaker) -> None:
    first = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY}).json()
    second = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY}).json()

    assert uuid.UUID(first["request_id"]).version == 4
    assert uuid.UUID(second["request_id"]).version == 4
    assert first["request_id"] != second["request_id"]


def test_debug_block_is_opt_in(seeded_factory: sessionmaker) -> None:
    plain = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY}).json()
    debug = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY, "debug": True}).json()

    assert plain["debug_info"] is None
    assert debug["debug_info"] is not None
    assert debug["debug_info"]["required_tools"] == [DEVICE_TOOL]
    assert [item["tool"] for item in debug["debug_info"]["tool_status"]] == [DEVICE_TOOL]
    assert debug["debug_info"]["tool_status"][0]["found"] is True
    assert debug["debug_info"]["internal_error"] is None


# --------------------------------------------------------------------------- #
# 3. Document evidence
# --------------------------------------------------------------------------- #


def test_invoke_returns_document_evidence_from_rag(
    monkeypatch: pytest.MonkeyPatch,
    seeded_factory: sessionmaker,
) -> None:
    _use_provider(monkeypatch, _StubProvider([MANUAL_HIT]))

    response = client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    assert response.status_code == 200
    payload = response.json()
    assert MANUAL_TOOL in payload["tools_called"]

    document_evidence = [item for item in payload["evidence"] if item["source_type"] == "document"]
    assert len(document_evidence) == 1
    item = document_evidence[0]
    assert item["tool_name"] == MANUAL_TOOL
    assert item["document"] == "PowerFlex_520_User_Manual.pdf"
    assert item["page"] == 161
    assert item["score"] == pytest.approx(0.4988)
    assert item["score_semantics"] == VECTOR_COSINE_DISTANCE
    assert item["higher_is_better"] is False
    assert "【维护手册证据】" in payload["answer"]
    assert "PowerFlex_520_User_Manual.pdf" in payload["answer"]


def test_empty_knowledge_base_produces_no_document_evidence(
    seeded_factory: sessionmaker,
) -> None:
    response = client.post(INVOKE_PATH, json={"query": ALARM_QUERY})
    payload = response.json()

    # With a healthy but empty knowledge base the request still succeeds and
    # fabricates nothing: tool evidence survives, document evidence does not.
    assert response.status_code == 200
    assert MANUAL_TOOL in payload["tools_called"]
    assert all(item["source_type"] == "tool" for item in payload["evidence"])
    assert "未在维护手册中检索到相关片段" in payload["answer"]


def test_unavailable_knowledge_base_is_reported_not_faked(
    monkeypatch: pytest.MonkeyPatch,
    seeded_factory: sessionmaker,
) -> None:
    class _UnavailableProvider(RAGProvider):
        provider_id = "unavailable"

        def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
            raise RAGProviderError("knowledge base index is not built")

    _use_provider(monkeypatch, _UnavailableProvider())

    response = client.post(INVOKE_PATH, json={"query": ALARM_QUERY})
    payload = response.json()

    assert response.status_code == 200
    assert all(item["source_type"] == "tool" for item in payload["evidence"])
    assert "维护手册检索不可用" in payload["answer"]


# --------------------------------------------------------------------------- #
# 4. Request validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("body", [{}, {"query": ""}])
def test_empty_query_is_rejected_with_422(body: dict[str, Any]) -> None:
    response = client.post(INVOKE_PATH, json=body)

    assert response.status_code == 422
    assert response.json()["detail"]


def test_blank_query_is_rejected_with_422() -> None:
    """A run of spaces is not a question, so it never reaches the parser."""
    response = client.post(INVOKE_PATH, json={"query": "   \t  "})

    assert response.status_code == 422


def test_query_is_stripped_before_use(seeded_factory: sessionmaker) -> None:
    response = client.post(INVOKE_PATH, json={"query": f"  {DEVICE_QUERY}  "})

    assert response.status_code == 200
    assert response.json()["query"] == DEVICE_QUERY
    assert response.json()["equipment_id"] == "PLC-001"


def test_query_at_the_length_boundary_is_accepted() -> None:
    response = client.post(INVOKE_PATH, json={"query": "x" * AGENT_QUERY_MAX_LENGTH})

    assert response.status_code == 200
    assert len(response.json()["query"]) == AGENT_QUERY_MAX_LENGTH


def test_query_over_the_length_boundary_is_rejected() -> None:
    response = client.post(INVOKE_PATH, json={"query": "x" * (AGENT_QUERY_MAX_LENGTH + 1)})

    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 5. Failure handling
# --------------------------------------------------------------------------- #


def test_agent_failure_returns_a_structured_error(
    graph_override: Callable[[Any], None],
) -> None:
    graph_override(_exploding_runner)

    response = client.post(INVOKE_PATH, json={"query": ALARM_QUERY})

    assert response.status_code == 500
    payload = response.json()
    assert set(payload) == {"request_id", "error", "message", "latency_ms"}
    assert payload["error"] == AGENT_INVOCATION_FAILED
    assert uuid.UUID(payload["request_id"]).version == 4
    assert payload["latency_ms"] >= 0
    assert payload["message"]


def test_agent_failure_never_exposes_internals(
    graph_override: Callable[[Any], None],
) -> None:
    graph_override(_exploding_runner)

    body = client.post(INVOKE_PATH, json={"query": ALARM_QUERY}).text

    assert "Traceback" not in body
    assert "RuntimeError" not in body
    assert "app.services" not in body
    assert "example.invalid" not in body
    assert FAKE_SECRET not in body


def test_failure_log_carries_the_correlation_id_without_the_secret(
    graph_override: Callable[[Any], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph_override(_exploding_runner)

    with caplog.at_level(logging.INFO, logger="app.services.agent_service"):
        payload = client.post(INVOKE_PATH, json={"query": ALARM_QUERY}).json()

    records = [record for record in caplog.records if "agent_invoke_failed" in record.getMessage()]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    message = record.getMessage()
    assert payload["request_id"] in message
    assert "success=false" in message
    # The class name is logged; the message it carried is not, because it can
    # embed configuration values.
    assert "error_type=RuntimeError" in message
    assert FAKE_SECRET not in caplog.text
    assert "example.invalid" not in caplog.text
    assert "Traceback" not in caplog.text


def test_success_log_carries_the_required_fields(
    seeded_factory: sessionmaker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.services.agent_service"):
        payload = client.post(INVOKE_PATH, json={"query": DEVICE_QUERY}).json()

    records = [record for record in caplog.records if "agent_invoke " in record.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert payload["request_id"] in message
    assert "success=true" in message
    assert "latency_ms=" in message
    assert f"tools_called={DEVICE_TOOL}" in message
    # The query is rendered as a repr so a query containing spaces stays legible
    # to a log parser and cannot break the one-line-per-invocation contract.
    assert f"query={DEVICE_QUERY!r}" in message


def test_log_stays_on_one_line_for_a_multiline_query(
    seeded_factory: sessionmaker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tricky = "PLC-001\n状态\t检查"

    with caplog.at_level(logging.INFO, logger="app.services.agent_service"):
        payload = client.post(INVOKE_PATH, json={"query": tricky}).json()

    records = [record for record in caplog.records if "agent_invoke " in record.getMessage()]
    assert len(records) == 1
    assert "\n" not in records[0].getMessage()
    assert payload["query"] == "PLC-001\n状态\t检查"


# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #


def test_application_logging_surfaces_info_records() -> None:
    """An unconfigured logger would inherit WARNING and drop every INFO line."""
    app_logger = logging.getLogger("app")

    assert app_logger.handlers
    assert app_logger.level == logging.INFO
    assert logging.getLogger("app.services.agent_service").isEnabledFor(logging.INFO)
    # Propagation stays on so a host root handler, caplog included, still sees
    # these records.
    assert app_logger.propagate is True


def test_logging_configuration_is_idempotent() -> None:
    app_logger = logging.getLogger("app")
    before = len(app_logger.handlers)

    configure_logging("DEBUG")
    try:
        assert len(app_logger.handlers) == before
        assert app_logger.level == logging.DEBUG
    finally:
        configure_logging("INFO")

    assert app_logger.level == logging.INFO


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging("chatty")


# --------------------------------------------------------------------------- #
# Service-level contract
# --------------------------------------------------------------------------- #


def test_service_reports_execution_rather_than_the_plan() -> None:
    """``tools_called`` must not be a copy of ``required_tools``."""

    def runner(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "intent": "alarm_diagnosis",
            "required_tools": [DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL],
            "tool_results": [
                {"tool": DEVICE_TOOL, "result": {"device_id": "PLC-001", "found": True}}
            ],
            "error": f"missing state values for {ALARM_TOOL}: alarm_code",
            "final_answer": "degraded run",
            "evidence": [],
        }

    response = AgentService(runner=runner).invoke(ALARM_QUERY, debug=True)

    assert response.tools_called == [DEVICE_TOOL]
    assert response.debug_info is not None
    assert response.debug_info.required_tools == [DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL]
    assert response.debug_info.internal_error is not None
    assert response.answer == "degraded run"


def _junk_runner(payload: dict[str, Any]) -> Any:
    """Runner that violates the state contract on purpose.

    The return type is ``Any`` so the call type-checks while the runtime guard
    inside the service is what the test actually exercises.
    """
    return "not a state"


def test_service_raises_a_structured_error_when_the_pipeline_returns_junk() -> None:
    """A state that cannot be translated fails as a request, not as a crash."""
    service = AgentService(runner=_junk_runner)

    with pytest.raises(AgentInvocationError) as excinfo:
        service.invoke(DEVICE_QUERY)

    error = excinfo.value
    assert error.code == AGENT_INVOCATION_FAILED
    assert uuid.UUID(error.request_id).version == 4
    assert error.latency_ms >= 0

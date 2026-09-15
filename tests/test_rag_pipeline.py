"""Graph-level tests for the V0.4 RAG integration.

These tests pin three things the integration must guarantee:

1. the node order is ``route_query -> plan_actions -> execute_tools ->
   retrieve_context -> synthesize``;
2. ``retrieve_context`` only standardizes what the executor already produced and
   never contacts a knowledge base itself;
3. manual fragments become ``source_type=document`` evidence and are rendered in
   the answer, while an unavailable knowledge base is reported as unavailable
   rather than as "no evidence".
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.graph import (
    build_graph,
    get_graph,
    plan_actions,
    retrieve_context,
    synthesize,
)
from app.agent.state import MaintenanceState
from app.database import session as db_session
from app.database.init_db import init_db
from app.integrations.rag import RAGProvider, RAGProviderError, RAGSearchHit, RAGSearchResponse
from app.tools.names import ToolName

DEVICE_TOOL = ToolName.GET_DEVICE_STATUS.value
ALARM_TOOL = ToolName.QUERY_ALARM_CODE.value
MANUAL_TOOL = ToolName.SEARCH_MAINTENANCE_MANUAL.value


# --------------------------------------------------------------------------- #
# Fixtures and doubles
# --------------------------------------------------------------------------- #


@pytest.fixture()
def seeded_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[sessionmaker]:
    """Point the default session factory at a seeded in-memory database."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, future=True)
    factory = sessionmaker(bind=engine, future=True)
    init_db(seed=True, reset=True, engine=engine, session_factory=factory)
    monkeypatch.setattr(db_session, "SessionLocal", factory)
    try:
        yield factory
    finally:
        engine.dispose()


class _StubProvider(RAGProvider):
    provider_id = "stub"

    def __init__(self, hits: list[RAGSearchHit] | None = None) -> None:
        self._hits = hits if hits is not None else []
        self.calls = 0

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        self.calls += 1
        return RAGSearchResponse(query=query, hits=self._hits, retrieval_mode="hybrid")


class _FailingProvider(RAGProvider):
    provider_id = "failing"

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        raise RAGProviderError("Knowledge base is empty.")


class _ExplodingProvider(RAGProvider):
    provider_id = "exploding"

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:  # pragma: no cover
        raise AssertionError("retrieve_context must not call any provider")


def _use_provider(monkeypatch: pytest.MonkeyPatch, provider: RAGProvider) -> None:
    """Route the tool layer to ``provider`` for the duration of a test."""
    monkeypatch.setattr(
        "app.integrations.rag.factory.get_rag_provider",
        lambda: provider,
    )


def _manual_payload(
    *,
    found: bool,
    results: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "query": "F0045",
        "found": found,
        "results": results or [],
        "provider": "stub",
        "retrieval_mode": "hybrid",
        "error": error,
    }


def _hit_payload() -> dict[str, Any]:
    return {
        "content": "Drive overload trip is caused by excessive load.",
        "document": "ABB_ACS580_Firmware_Manual.pdf",
        "page": 42,
        "section": "4.3",
        "chunk_id": "chunk-00042",
        "score": 0.31,
    }


# --------------------------------------------------------------------------- #
# 1. Node order
# --------------------------------------------------------------------------- #


def _node_order() -> list[str]:
    edges = {
        (edge.source, edge.target)
        for edge in build_graph().get_graph().edges
        if not edge.conditional
    }
    order: list[str] = []
    current = "__start__"
    while True:
        nxt = next((target for source, target in edges if source == current), None)
        if nxt is None:
            break
        order.append(nxt)
        current = nxt
    return order


def test_graph_node_order_is_route_plan_execute_retrieve_synthesize() -> None:
    assert _node_order() == [
        "route_query",
        "plan_actions",
        "execute_tools",
        "retrieve_context",
        "synthesize",
        "__end__",
    ]


# --------------------------------------------------------------------------- #
# 2. Planner adds the RAG tool only for diagnosis / guidance intents
# --------------------------------------------------------------------------- #


def test_planner_requests_manual_search_for_alarm_diagnosis() -> None:
    update = plan_actions(
        MaintenanceState(equipment_id="PLC-001", alarm_code="F0045", intent="alarm_diagnosis")
    )

    assert update["required_tools"] == [DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL]


def test_planner_requests_manual_search_for_maintenance_advice() -> None:
    update = plan_actions(MaintenanceState(equipment_id="Robot-001", intent="maintenance_advice"))

    assert update["required_tools"] == [DEVICE_TOOL, MANUAL_TOOL]


def test_planner_skips_manual_search_for_device_status() -> None:
    update = plan_actions(MaintenanceState(equipment_id="PLC-001", intent="device_status"))

    assert update["required_tools"] == [DEVICE_TOOL]


def test_planner_skips_manual_search_for_unknown_intent() -> None:
    update = plan_actions(MaintenanceState(equipment_id="PLC-001"))

    assert update["required_tools"] == [DEVICE_TOOL]


# --------------------------------------------------------------------------- #
# 3. retrieve_context standardizes without calling out
# --------------------------------------------------------------------------- #


def test_retrieve_context_standardizes_tool_results_into_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_provider(monkeypatch, _ExplodingProvider())
    state = MaintenanceState(
        tool_results=[
            {
                "tool": DEVICE_TOOL,
                "result": {"device_id": "PLC-001", "found": False},
            },
            {
                "tool": ALARM_TOOL,
                "result": {"alarm_code": "F0045", "found": False},
            },
        ]
    )

    update = retrieve_context(state)

    evidence = update["retrieved_context"]
    assert [item["tool_name"] for item in evidence] == [DEVICE_TOOL, ALARM_TOOL]
    assert all(item["source_type"] == "tool" for item in evidence)


def test_retrieve_context_maps_manual_hits_to_document_evidence() -> None:
    manual_result = _manual_payload(found=True, results=[_hit_payload()])
    state = MaintenanceState(tool_results=[{"tool": MANUAL_TOOL, "result": manual_result}])

    evidence = retrieve_context(state)["retrieved_context"]

    assert len(evidence) == 1
    item = evidence[0]
    assert item["source_type"] == "document"
    assert item["tool_name"] == MANUAL_TOOL
    assert item["source"] == "ABB_ACS580_Firmware_Manual.pdf"
    assert item["document"] == "ABB_ACS580_Firmware_Manual.pdf"
    assert item["page"] == 42
    assert item["score"] == pytest.approx(0.31)
    assert item["content"] == "Drive overload trip is caused by excessive load."


def test_retrieve_context_emits_no_document_evidence_on_a_miss() -> None:
    state = MaintenanceState(
        tool_results=[{"tool": MANUAL_TOOL, "result": _manual_payload(found=False)}]
    )

    assert retrieve_context(state)["retrieved_context"] == []


def test_retrieve_context_emits_no_document_evidence_when_unavailable() -> None:
    state = MaintenanceState(
        tool_results=[
            {
                "tool": MANUAL_TOOL,
                "result": _manual_payload(found=False, error="Knowledge base is empty."),
            }
        ]
    )

    assert retrieve_context(state)["retrieved_context"] == []


def test_retrieve_context_ignores_unknown_tool_entries() -> None:
    state = MaintenanceState(tool_results=[{"tool": "no_such_tool", "result": {"found": True}}])

    assert retrieve_context(state)["retrieved_context"] == []


# --------------------------------------------------------------------------- #
# 4. Synthesis renders the maintenance manual section
# --------------------------------------------------------------------------- #


def test_synthesize_renders_manual_evidence_section() -> None:
    state = MaintenanceState(
        tool_results=[
            {"tool": MANUAL_TOOL, "result": _manual_payload(found=True, results=[_hit_payload()])}
        ]
    )
    state["retrieved_context"] = retrieve_context(state)["retrieved_context"]

    answer = synthesize(state)["final_answer"]

    assert "【维护手册证据】" in answer
    assert "ABB_ACS580_Firmware_Manual.pdf" in answer
    assert "第 42 页" in answer
    assert "Drive overload trip is caused by excessive load." in answer


def test_synthesize_marks_manual_section_as_not_requested() -> None:
    answer = synthesize(MaintenanceState())["final_answer"]

    assert "【维护手册证据】" in answer
    assert "本次未检索维护手册" in answer


def test_synthesize_reports_unavailable_manual_retrieval() -> None:
    state = MaintenanceState(
        tool_results=[
            {
                "tool": MANUAL_TOOL,
                "result": _manual_payload(found=False, error="Knowledge base is empty."),
            }
        ]
    )

    update = synthesize(state)

    assert "维护手册检索不可用" in update["final_answer"]
    assert "Knowledge base is empty." in update["final_answer"]
    # An unavailable retrieval produces no citable document evidence.
    assert update["evidence"] == []


def test_synthesize_distinguishes_empty_from_unavailable() -> None:
    empty = synthesize(
        MaintenanceState(
            tool_results=[{"tool": MANUAL_TOOL, "result": _manual_payload(found=False)}]
        )
    )["final_answer"]

    assert "未在维护手册中检索到相关片段" in empty
    assert "不可用" not in empty


# --------------------------------------------------------------------------- #
# 5. End to end
# --------------------------------------------------------------------------- #


def test_end_to_end_with_real_manual_hit(
    monkeypatch: pytest.MonkeyPatch,
    seeded_factory: sessionmaker,
) -> None:
    provider = _StubProvider([RAGSearchHit(**_hit_payload())])
    _use_provider(monkeypatch, provider)

    final = get_graph().invoke({"query": "包装线PLC报警F0045怎么办"})

    assert final["required_tools"] == [DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL]
    assert [entry["tool"] for entry in final["tool_results"]] == [
        DEVICE_TOOL,
        ALARM_TOOL,
        MANUAL_TOOL,
    ]
    document_evidence = [item for item in final["evidence"] if item["source_type"] == "document"]
    assert len(document_evidence) == 1
    assert document_evidence[0]["page"] == 42
    assert "【维护手册证据】" in final["final_answer"]
    assert provider.calls == 1


def test_end_to_end_with_unavailable_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
    seeded_factory: sessionmaker,
) -> None:
    _use_provider(monkeypatch, _FailingProvider())

    final = get_graph().invoke({"query": "包装线PLC报警F0045怎么办"})

    # The tool layer is still executed and still reports truthfully.
    manual_entry = next(entry for entry in final["tool_results"] if entry["tool"] == MANUAL_TOOL)
    assert manual_entry["result"]["found"] is False
    assert manual_entry["result"]["error"] == "Knowledge base is empty."
    # Tool evidence survives even though document evidence cannot be produced.
    assert all(item["source_type"] == "tool" for item in final["evidence"])
    assert "维护手册检索不可用" in final["final_answer"]


def test_end_to_end_device_query_never_touches_the_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
    seeded_factory: sessionmaker,
) -> None:
    _use_provider(monkeypatch, _ExplodingProvider())

    final = get_graph().invoke({"query": "PLC-001 现在什么状态"})

    assert final["required_tools"] == [DEVICE_TOOL]
    assert "本次未检索维护手册" in final["final_answer"]
    assert final["evidence"]

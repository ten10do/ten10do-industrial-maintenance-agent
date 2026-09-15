"""Tests for the Planner -> Executor -> Synthesis pipeline.

The executor tests deliberately set ``required_tools`` by hand: they must prove
the executor follows the plan rather than re-deciding which tools to call.

Since V0.4 the planner also schedules ``search_maintenance_manual`` for
diagnosis intents. These tests stub the RAG provider with a zero-hit response so
they stay hermetic and describe pipeline behaviour rather than knowledge base
contents; the manual tool's own contract is covered by
``tests/test_maintenance_manual_tool.py`` and ``tests/test_rag_pipeline.py``.
"""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.graph import execute_tools, get_graph, plan_actions, synthesize
from app.agent.state import MaintenanceState
from app.database import session as db_session
from app.database.init_db import init_db
from app.integrations.rag import RAGProvider, RAGSearchResponse
from app.schemas import Evidence, SourceType
from app.tools import registry
from app.tools.names import ToolName

DEVICE_TOOL = ToolName.GET_DEVICE_STATUS.value
ALARM_TOOL = ToolName.QUERY_ALARM_CODE.value
MANUAL_TOOL = ToolName.SEARCH_MAINTENANCE_MANUAL.value


class _EmptyManualProvider(RAGProvider):
    """Provider double that always reports a successful, empty retrieval."""

    provider_id = "test-empty"

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        return RAGSearchResponse(query=query, hits=[], retrieval_mode="hybrid")


@pytest.fixture(autouse=True)
def _stub_manual_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pipeline tests independent of any real knowledge base."""
    monkeypatch.setattr(
        "app.integrations.rag.factory.get_rag_provider",
        lambda: _EmptyManualProvider(),
    )


@pytest.fixture()
def seeded_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[sessionmaker]:
    """Point the default session factory at a seeded in-memory database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        future=True,
    )
    factory = sessionmaker(bind=engine, future=True)
    init_db(seed=True, reset=True, engine=engine, session_factory=factory)
    monkeypatch.setattr(db_session, "SessionLocal", factory)
    try:
        yield factory
    finally:
        engine.dispose()


# --------------------------------------------------------------------------- #
# 1. required_tools uses registry names
# --------------------------------------------------------------------------- #


def test_planner_only_emits_registry_names() -> None:
    update = plan_actions(MaintenanceState(equipment_id="PLC-001", alarm_code="F0045"))

    assert update["required_tools"] == [DEVICE_TOOL, ALARM_TOOL]
    assert set(update["required_tools"]).issubset(set(registry.names()))


# --------------------------------------------------------------------------- #
# 2. execute_tools follows required_tools exactly
# --------------------------------------------------------------------------- #


def test_executor_follows_required_tools_exactly(seeded_factory: sessionmaker) -> None:
    update = execute_tools(
        MaintenanceState(
            equipment_id="PLC-001",
            alarm_code="F0045",
            required_tools=[ALARM_TOOL],
        )
    )

    # equipment_id is present, yet the device tool is not called: only the plan
    # decides, the executor does not add tools of its own.
    assert [entry["tool"] for entry in update["tool_results"]] == [ALARM_TOOL]
    assert "error" not in update


def test_executor_reports_unknown_tool_without_substituting(
    seeded_factory: sessionmaker,
) -> None:
    update = execute_tools(MaintenanceState(equipment_id="PLC-001", required_tools=["device_tool"]))

    assert update["tool_results"] == []
    assert "device_tool" in update["error"]


def test_executor_reports_missing_arguments() -> None:
    update = execute_tools(MaintenanceState(required_tools=[DEVICE_TOOL]))

    assert update["tool_results"] == []
    assert "missing state values" in update["error"]


# --------------------------------------------------------------------------- #
# 3. device-only plan does not call the alarm tool
# --------------------------------------------------------------------------- #


def test_executor_skips_alarm_tool_when_not_requested(
    seeded_factory: sessionmaker,
) -> None:
    update = execute_tools(MaintenanceState(equipment_id="PLC-001", required_tools=[DEVICE_TOOL]))

    assert [entry["tool"] for entry in update["tool_results"]] == [DEVICE_TOOL]
    # The PLC-001 row does carry alarm_code F0045, but the planner did not ask
    # for the alarm tool, so the executor must not call it anyway.
    assert update["tool_results"][0]["result"]["alarm_code"] == "F0045"


# --------------------------------------------------------------------------- #
# 4. both tools keep the planned order
# --------------------------------------------------------------------------- #


def test_executor_keeps_required_tool_order(seeded_factory: sessionmaker) -> None:
    update = execute_tools(
        MaintenanceState(
            equipment_id="PLC-001",
            alarm_code="F0045",
            required_tools=[DEVICE_TOOL, ALARM_TOOL],
        )
    )

    assert [entry["tool"] for entry in update["tool_results"]] == [DEVICE_TOOL, ALARM_TOOL]
    assert update["tool_results"][0]["result"]["device_id"] == "PLC-001"
    assert update["tool_results"][1]["result"]["alarm_code"] == "F0045"
    assert "error" not in update


# --------------------------------------------------------------------------- #
# 5 & 6. synthesize produces evidence and a final answer
# --------------------------------------------------------------------------- #


def test_synthesize_produces_evidence(seeded_factory: sessionmaker) -> None:
    final = get_graph().invoke({"query": "包装线PLC报警F0045怎么办"})
    evidence = final["evidence"]

    assert [item["tool_name"] for item in evidence] == [DEVICE_TOOL, ALARM_TOOL]
    assert all(item["source_type"] == SourceType.TOOL.value for item in evidence)
    assert all(item["content"] for item in evidence)
    assert evidence[0]["source"] == "sqlite:devices"
    assert evidence[1]["source"] == "data/alarms.json"
    # Reserved RAG fields stay empty for tool-sourced evidence.
    assert all(item["document"] is None for item in evidence)
    assert all(item["page"] is None for item in evidence)
    assert all(item["score"] is None for item in evidence)


def test_synthesize_produces_final_answer(seeded_factory: sessionmaker) -> None:
    final = get_graph().invoke({"query": "包装线PLC报警F0045怎么办"})
    answer = final["final_answer"]

    assert "PLC-001" in answer
    assert "包装线PLC" in answer
    assert "running" in answer
    assert "温度 78.0" in answer
    assert "F0045" in answer
    assert "warning" in answer
    assert "可能原因" in answer
    assert "建议动作" in answer
    assert "热封刀加热管老化导致功率漂移" in answer
    assert answer == final["answer"]


# --------------------------------------------------------------------------- #
# 7. misses never fabricate
# --------------------------------------------------------------------------- #


def test_missing_device_is_reported_without_fabrication() -> None:
    update = synthesize(
        MaintenanceState(
            tool_results=[{"tool": DEVICE_TOOL, "result": {"device_id": "PLC-999", "found": False}}]
        )
    )
    answer = update["final_answer"]

    assert "未找到" in answer
    assert "可能原因" not in answer
    assert "建议动作" not in answer
    assert update["evidence"][0]["content"] == "设备 PLC-999 未找到：数据库中没有该设备记录。"


def test_missing_alarm_is_reported_without_fabrication() -> None:
    update = synthesize(
        MaintenanceState(
            tool_results=[{"tool": ALARM_TOOL, "result": {"alarm_code": "Z9999", "found": False}}]
        )
    )
    answer = update["final_answer"]

    assert "未找到" in answer
    assert "可能原因" not in answer
    assert "建议动作" not in answer


def test_synthesis_without_tool_results_does_not_fabricate() -> None:
    update = synthesize(MaintenanceState())
    answer = update["final_answer"]

    assert update["evidence"] == []
    assert "本次未查询设备状态" in answer
    assert "本次未查询报警信息" in answer
    assert "可能原因" not in answer


# --------------------------------------------------------------------------- #
# 8. full end-to-end runs
# --------------------------------------------------------------------------- #


def test_end_to_end_chinese_alarm_query(seeded_factory: sessionmaker) -> None:
    final = get_graph().invoke({"query": "包装线PLC报警F0045怎么办"})

    assert final["intent"] == "alarm_diagnosis"
    assert final["equipment_id"] == "PLC-001"
    assert final["alarm_code"] == "F0045"
    # alarm_diagnosis also schedules the maintenance manual lookup.
    assert final["required_tools"] == [DEVICE_TOOL, ALARM_TOOL, MANUAL_TOOL]
    assert [entry["tool"] for entry in final["tool_results"]] == [
        DEVICE_TOOL,
        ALARM_TOOL,
        MANUAL_TOOL,
    ]
    lookups = [entry for entry in final["tool_results"] if entry["tool"] != MANUAL_TOOL]
    assert all(entry["result"]["found"] is True for entry in lookups)
    assert final["final_answer"]
    assert final["evidence"]


def test_end_to_end_device_only_query(seeded_factory: sessionmaker) -> None:
    final = get_graph().invoke({"query": "PLC-001 现在什么状态"})

    assert final["required_tools"] == [DEVICE_TOOL]
    assert [entry["tool"] for entry in final["tool_results"]] == [DEVICE_TOOL]
    assert "本次未查询报警信息" in final["final_answer"]


# --------------------------------------------------------------------------- #
# Evidence model contract
# --------------------------------------------------------------------------- #


def test_evidence_model_supports_future_document_fields() -> None:
    item = Evidence(
        source_type=SourceType.DOCUMENT,
        source="manuals/plc-001.pdf",
        tool_name="search_manuals",
        content="热封单元温度上限为 80 摄氏度。",
        document="manuals/plc-001.pdf",
        page=12,
        score=0.87,
    )

    assert item.source_type is SourceType.DOCUMENT
    assert item.page == 12
    assert item.score == pytest.approx(0.87)
    assert item.tool_name == "search_manuals"

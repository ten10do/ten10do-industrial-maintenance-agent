"""Tests for the device tool, alarm tool and registry."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.init_db import init_db
from app.tools import get_device_status, query_alarm_code, registry
from app.tools.alarm_tool import DEFAULT_ALARMS_PATH
from app.tools.names import TOOL_NAMES, ToolName


@pytest.fixture()
def memory_session() -> Iterator[Session]:
    """A session bound to an isolated, seeded in-memory database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        future=True,
    )
    factory = sessionmaker(bind=engine, future=True)
    init_db(seed=True, reset=True, engine=engine, session_factory=factory)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_get_device_status_returns_plc_001(memory_session: Session) -> None:
    status = get_device_status("PLC-001", session=memory_session)

    assert status.found is True
    assert status.device_id == "PLC-001"
    assert status.device_name == "包装线PLC"
    assert status.device_type == "PLC"
    assert status.status == "running"
    assert status.temperature == 78.0
    assert status.alarm_code == "F0045"
    assert status.last_maintenance_time is not None


def test_get_device_status_unknown_device(memory_session: Session) -> None:
    status = get_device_status("NOPE-999", session=memory_session)

    assert status.found is False
    assert status.device_name is None
    assert status.temperature is None


def test_alarm_catalog_file_exists() -> None:
    assert DEFAULT_ALARMS_PATH.exists(), f"Missing alarm catalog: {DEFAULT_ALARMS_PATH}"


def test_query_alarm_code_f0045() -> None:
    alarm = query_alarm_code("F0045")

    assert alarm.found is True
    assert alarm.alarm_code == "F0045"
    assert alarm.name
    assert alarm.description
    assert alarm.severity == "warning"
    assert alarm.possible_causes
    assert alarm.recommended_actions


def test_query_alarm_code_is_case_insensitive() -> None:
    assert query_alarm_code(" f0045 ").alarm_code == "F0045"
    assert query_alarm_code("f0045").found is True


def test_query_alarm_code_unknown() -> None:
    alarm = query_alarm_code("Z9999")

    assert alarm.found is False
    assert alarm.possible_causes == []
    assert alarm.recommended_actions == []


def test_builtin_tools_are_registered_under_canonical_names() -> None:
    names = set(registry.names())

    assert set(TOOL_NAMES).issubset(names)
    assert registry.get(ToolName.GET_DEVICE_STATUS.value).func is get_device_status
    assert registry.get(ToolName.QUERY_ALARM_CODE.value).func is query_alarm_code

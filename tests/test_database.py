"""Tests for the Device ORM model, seed data and database initialization."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.init_db import DEFAULT_SEED_PATH, init_db, load_seed_records
from app.database.models import Device

REQUIRED_DEVICE_IDS = {"PLC-001", "PLC-002", "Robot-001"}

EXPECTED_COLUMNS = {
    "id",
    "device_id",
    "device_name",
    "device_type",
    "location",
    "status",
    "temperature",
    "pressure",
    "rpm",
    "alarm_code",
    "last_maintenance_time",
}


@pytest.fixture()
def memory_engine():
    """An isolated in-memory SQLite engine per test."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        future=True,
    )
    try:
        yield engine
    finally:
        engine.dispose()


def test_device_model_exposes_required_columns() -> None:
    columns = {column.name for column in Device.__table__.columns}
    assert EXPECTED_COLUMNS.issubset(columns)


def test_seed_file_exists_and_contains_required_devices() -> None:
    assert DEFAULT_SEED_PATH.exists(), f"Missing seed file: {DEFAULT_SEED_PATH}"
    records = load_seed_records()
    device_ids = {record["device_id"] for record in records}
    assert REQUIRED_DEVICE_IDS.issubset(device_ids)


def test_init_db_creates_and_seeds_devices(memory_engine) -> None:
    factory = sessionmaker(bind=memory_engine, future=True)

    result = init_db(seed=True, reset=True, engine=memory_engine, session_factory=factory)

    assert result.total >= len(REQUIRED_DEVICE_IDS)

    with factory() as session:
        device = session.query(Device).filter_by(device_id="PLC-001").one()
        assert device.device_name == "包装线PLC"
        assert device.status == "running"
        assert device.temperature == 78.0
        assert device.alarm_code == "F0045"
        assert device.last_maintenance_time is not None


def test_seeding_is_idempotent(memory_engine) -> None:
    factory = sessionmaker(bind=memory_engine, future=True)

    first = init_db(seed=True, reset=True, engine=memory_engine, session_factory=factory)
    second = init_db(seed=True, reset=False, engine=memory_engine, session_factory=factory)

    assert first.total == second.total


def test_reset_drops_existing_rows(memory_engine) -> None:
    factory = sessionmaker(bind=memory_engine, future=True)

    init_db(seed=True, reset=True, engine=memory_engine, session_factory=factory)

    with factory() as session:
        session.query(Device).delete()
        session.commit()

    result = init_db(seed=False, reset=False, engine=memory_engine, session_factory=factory)
    assert result.total == 0

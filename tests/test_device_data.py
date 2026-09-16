"""Tests for the external device data source layer and its tool dispatch.

Everything here runs against ``tests/fixtures/metropt3_sample.csv`` or against
temporary CSVs written during the test. Those inputs are **synthetic test
fixtures**, not measurements: they exist to pin parsing, status derivation and
error handling. The 208 MiB published dataset is never required, so the suite
stays hermetic and fast.

The published column layout is reproduced verbatim, including the dataset's own
``DV_eletric`` spelling and its unnamed index column, because the adapter is
expected to accept exactly the header the dataset ships.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.database.init_db import init_db
from app.database.models import Device
from app.integrations.device_data import (
    METROPT3_DEVICE_ID,
    METROPT3_SOURCE_ID,
    DeviceDataSourceError,
    MetroPT3Adapter,
)
from app.integrations.device_data.factory import (
    build_adapters,
    get_device_adapters,
    reset_device_adapters,
    resolve_adapter,
)
from app.tools import registry
from app.tools.device_tool import DeviceStatusInput, DeviceStatusOutput, get_device_status
from app.tools.names import TOOL_NAMES

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSV = FIXTURES / "metropt3_sample.csv"

#: Header of the published dataset. The leading empty field is the dataset's own
#: unnamed index column and is kept so the fixture matches byte for byte in shape.
HEADER = (
    ",timestamp,TP2,TP3,H1,DV_pressure,Reservoirs,Oil_temperature,Motor_current,"
    "COMP,DV_eletric,Towers,MPG,LPS,Pressure_switch,Oil_level,Caudal_impulses"
)

#: Statuses the sample fixture is engineered to reach, per row, in file order.
SAMPLE_STATUSES = [
    "stopped",
    "running_offloaded",
    "observed",
    "running_offloaded",
    "running_under_load",
]


def _row(
    index: int,
    timestamp: str,
    *,
    tp2: str = "1.0",
    tp3: str = "2.0",
    h1: str = "3.0",
    dv_pressure: str = "0.0",
    reservoirs: str = "2.0",
    oil_temperature: str = "50.0",
    motor_current: str = "0.0",
    comp: str = "1.0",
    dv_eletric: str = "0.0",
    towers: str = "0.0",
    mpg: str = "1.0",
    lps: str = "1.0",
    pressure_switch: str = "1.0",
    oil_level: str = "1.0",
    caudal_impulses: str = "1.0",
) -> str:
    """Build one CSV record with the published field order."""
    fields = [
        str(index),
        timestamp,
        tp2,
        tp3,
        h1,
        dv_pressure,
        reservoirs,
        oil_temperature,
        motor_current,
        comp,
        dv_eletric,
        towers,
        mpg,
        lps,
        pressure_switch,
        oil_level,
        caudal_impulses,
    ]
    return ",".join(fields)


def _write_csv(path: Path, rows: list[str]) -> Path:
    """Write a header plus records, no trailing blank line."""
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clear_adapter_cache() -> Iterator[None]:
    """Keep the process-wide adapter cache from leaking between tests."""
    reset_device_adapters()
    yield
    reset_device_adapters()


@pytest.fixture()
def sample_adapter() -> MetroPT3Adapter:
    """An adapter bound to the synthetic sample fixture."""
    return MetroPT3Adapter(SAMPLE_CSV)


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


# --------------------------------------------------------------------------- #
# Fixture hygiene
# --------------------------------------------------------------------------- #


def test_sample_fixture_is_present_and_declared_synthetic() -> None:
    """The CSV fixture must ship with its provenance note.

    Both files are synthetic test inputs. The note is asserted here so a fixture
    can never be mistaken for, or silently replaced by, real measurements.
    """
    assert SAMPLE_CSV.exists()
    note = (FIXTURES / "README.md").read_text(encoding="utf-8")
    assert "synthetic" in note.lower()
    assert "TEST FIXTURE" in note
    assert "not a sample of the real dataset" in note


def test_sample_fixture_header_matches_published_layout() -> None:
    first_line = SAMPLE_CSV.read_text(encoding="utf-8-sig").splitlines()[0]
    assert first_line == HEADER


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_snapshot_parses_last_record(sample_adapter: MetroPT3Adapter) -> None:
    snapshot = sample_adapter.get_latest_snapshot(METROPT3_DEVICE_ID)

    assert snapshot is not None
    assert snapshot.device_id == METROPT3_DEVICE_ID
    assert snapshot.device_name == "MetroPT-3 Air Production Unit"
    assert snapshot.device_type == "air_compressor"
    assert snapshot.data_source == METROPT3_SOURCE_ID == "uci_metropt3"
    assert snapshot.source_timestamp == datetime(2020, 1, 1, 0, 0, 40)
    assert snapshot.status == "running_under_load"


def test_numeric_conversion_yields_floats(sample_adapter: MetroPT3Adapter) -> None:
    snapshot = sample_adapter.get_latest_snapshot(METROPT3_DEVICE_ID)
    assert snapshot is not None

    assert snapshot.measurements["TP2_bar"] == 1.4
    assert snapshot.measurements["TP3_bar"] == 2.4
    assert snapshot.measurements["H1_bar"] == 3.4
    assert snapshot.measurements["reservoir_pressure_bar"] == 2.4
    assert snapshot.measurements["oil_temperature_c"] == 50.4
    assert snapshot.measurements["motor_current_a"] == 7.0
    for value in snapshot.measurements.values():
        assert value is None or isinstance(value, float)


def test_digital_signals_keep_published_column_names(sample_adapter: MetroPT3Adapter) -> None:
    """Contact signals are exposed under the dataset's own spelling."""
    snapshot = sample_adapter.get_latest_snapshot(METROPT3_DEVICE_ID)
    assert snapshot is not None

    assert set(snapshot.digital_signals) == {
        "COMP",
        "DV_eletric",
        "Towers",
        "MPG",
        "LPS",
        "Pressure_switch",
        "Oil_level",
        "Caudal_impulses",
    }
    assert snapshot.digital_signals["DV_eletric"] == 1.0
    assert snapshot.digital_signals["COMP"] == 0.0


def test_blank_numeric_field_becomes_none(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "blank.csv", [_row(0, "2020-01-01 00:00:00", tp2="")])
    snapshot = MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)

    assert snapshot is not None
    assert snapshot.measurements["TP2_bar"] is None


def test_status_label_always_carries_its_derivation(sample_adapter: MetroPT3Adapter) -> None:
    """A derived label must never travel without saying it is derived."""
    snapshot = sample_adapter.get_latest_snapshot(METROPT3_DEVICE_ID)
    assert snapshot is not None

    assert snapshot.status_derivation.startswith("derived:")
    lowered = snapshot.status_derivation.lower()
    assert "ground-truth" in lowered or "ground truth" in lowered


@pytest.mark.parametrize(
    ("row", "expected_status"),
    [
        (
            _row(0, "2020-01-01 00:00:00", comp="1.0", dv_eletric="0.0", motor_current="0.0"),
            "stopped",
        ),
        (
            _row(1, "2020-01-01 00:00:01", comp="1.0", dv_eletric="0.0", motor_current="4.0"),
            "running_offloaded",
        ),
        (
            _row(2, "2020-01-01 00:00:02", comp="0.0", dv_eletric="0.0", motor_current="0.0"),
            "observed",
        ),
        (
            _row(3, "2020-01-01 00:00:03", comp="0.0", dv_eletric="1.0", motor_current="7.0"),
            "running_under_load",
        ),
    ],
)
def test_status_derivation_branches(tmp_path: Path, row: str, expected_status: str) -> None:
    """Each documented contact combination maps to its stated operational state."""
    csv_path = _write_csv(tmp_path / "one.csv", [row])
    snapshot = MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)

    assert snapshot is not None
    assert snapshot.status == expected_status


def test_sample_fixture_covers_every_status_branch(tmp_path: Path) -> None:
    """The fixture is only useful if it exercises all four derived states."""
    observed = []
    for index, row in enumerate(SAMPLE_CSV.read_text(encoding="utf-8-sig").splitlines()[1:]):
        csv_path = _write_csv(tmp_path / f"row{index}.csv", [row])
        snapshot = MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)
        assert snapshot is not None
        observed.append(snapshot.status)

    assert observed == SAMPLE_STATUSES
    assert set(observed) == {"stopped", "running_offloaded", "observed", "running_under_load"}


def test_latest_record_is_the_last_one(tmp_path: Path) -> None:
    rows = [_row(i, f"2020-01-01 00:00:0{i}") for i in range(4)]
    csv_path = _write_csv(tmp_path / "many.csv", rows)
    snapshot = MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)

    assert snapshot is not None
    assert snapshot.source_timestamp == datetime(2020, 1, 1, 0, 0, 3)


def test_tail_read_works_beyond_one_window(tmp_path: Path) -> None:
    """A file larger than the 64 KiB tail window must still resolve correctly.

    This is the path that has to discard the partial first line of the window.
    """
    rows = [_row(i, f"2020-01-01 00:00:{i % 60:02d}", tp2=f"1.{i}") for i in range(4000)]
    csv_path = _write_csv(tmp_path / "big.csv", rows)
    assert csv_path.stat().st_size > 65536

    snapshot = MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)
    assert snapshot is not None
    assert snapshot.measurements["TP2_bar"] == 1.3999
    assert snapshot.device_id == METROPT3_DEVICE_ID


def test_repeated_reads_are_served_from_cache(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "cache.csv", [_row(0, "2020-01-01 00:00:00")])
    adapter = MetroPT3Adapter(csv_path)

    first = adapter.get_latest_snapshot(METROPT3_DEVICE_ID)
    second = adapter.get_latest_snapshot(METROPT3_DEVICE_ID)
    assert first is second


def test_appended_record_invalidates_the_cache(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "grow.csv", [_row(0, "2020-01-01 00:00:00")])
    adapter = MetroPT3Adapter(csv_path)
    first = adapter.get_latest_snapshot(METROPT3_DEVICE_ID)

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    _write_csv(csv_path, [*rows[1:], _row(1, "2020-01-01 00:00:01")])

    second = adapter.get_latest_snapshot(METROPT3_DEVICE_ID)
    assert first is not None and second is not None
    assert second.source_timestamp == datetime(2020, 1, 1, 0, 0, 1)
    assert second is not first


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


def test_unknown_device_falls_through_to_none(sample_adapter: MetroPT3Adapter) -> None:
    """A device this source does not own is not an error, it is a miss."""
    assert sample_adapter.get_latest_snapshot("PLC-001") is None
    assert resolve_adapter("PLC-001", (sample_adapter,)) is None


def test_identifier_lookup_is_case_insensitive(sample_adapter: MetroPT3Adapter) -> None:
    assert sample_adapter.get_latest_snapshot("metro-apu-001") is not None
    assert resolve_adapter("metro-apu-001", (sample_adapter,)) is sample_adapter


def test_empty_path_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MetroPT3Adapter("")


def test_missing_file_raises_source_error(tmp_path: Path) -> None:
    adapter = MetroPT3Adapter(tmp_path / "absent.csv")

    with pytest.raises(DeviceDataSourceError) as excinfo:
        adapter.get_latest_snapshot(METROPT3_DEVICE_ID)
    assert "METROPT3_CSV_PATH" in str(excinfo.value)


def test_empty_file_raises_source_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(DeviceDataSourceError, match="no header"):
        MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)


def test_header_without_records_raises_source_error(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "header_only.csv", [])

    with pytest.raises(DeviceDataSourceError, match="no data records"):
        MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)


def test_missing_column_raises_source_error(tmp_path: Path) -> None:
    header = HEADER.replace(",Oil_temperature", "")
    rows = [_row(0, "2020-01-01 00:00:00")]
    csv_path = tmp_path / "missing_column.csv"
    csv_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    with pytest.raises(DeviceDataSourceError, match="Oil_temperature"):
        MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)


def test_field_count_mismatch_raises_source_error(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ragged.csv", ["1,2020-01-01 00:00:00,1.0,2.0"])

    with pytest.raises(DeviceDataSourceError, match="does not match the header"):
        MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)


def test_malformed_timestamp_raises_source_error(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "bad_time.csv", [_row(0, "01/01/2020 00:00")])

    with pytest.raises(DeviceDataSourceError, match="timestamp"):
        MetroPT3Adapter(csv_path).get_latest_snapshot(METROPT3_DEVICE_ID)


def test_describe_exposes_metadata_but_no_local_path(sample_adapter: MetroPT3Adapter) -> None:
    described = sample_adapter.describe()

    assert described["source_id"] == METROPT3_SOURCE_ID
    assert described["dataset"] == "MetroPT-3"
    assert described["dataset_license"] == "CC BY 4.0"
    assert described["device_ids"] == [METROPT3_DEVICE_ID]
    assert str(SAMPLE_CSV) not in repr(described)


# --------------------------------------------------------------------------- #
# Tool dispatch
# --------------------------------------------------------------------------- #


def test_tool_answers_from_the_external_source(sample_adapter: MetroPT3Adapter) -> None:
    status = get_device_status(METROPT3_DEVICE_ID, adapters=(sample_adapter,))

    assert status.found is True
    assert status.device_id == METROPT3_DEVICE_ID
    assert status.data_source == METROPT3_SOURCE_ID
    assert status.source_timestamp == datetime(2020, 1, 1, 0, 0, 40)
    assert status.status == "running_under_load"
    assert status.status_derivation is not None
    assert status.measurements is not None
    assert status.measurements["oil_temperature_c"] == 50.4


def test_external_readings_are_not_forced_into_plc_fields(
    sample_adapter: MetroPT3Adapter,
) -> None:
    """Foreign measurements stay in ``measurements``; PLC-shaped fields stay empty."""
    status = get_device_status(METROPT3_DEVICE_ID, adapters=(sample_adapter,))

    assert status.temperature is None
    assert status.pressure is None
    assert status.rpm is None
    assert status.alarm_code is None
    assert status.last_maintenance_time is None


def test_tool_reports_a_broken_source_without_a_fallthrough(tmp_path: Path) -> None:
    """Configured but unreadable is its own outcome, distinct from unknown device."""
    adapter = MetroPT3Adapter(tmp_path / "absent.csv")
    status = get_device_status(METROPT3_DEVICE_ID, adapters=(adapter,))

    assert status.found is False
    assert status.error is not None
    assert "METROPT3_CSV_PATH" in status.error
    assert status.measurements is None


def test_tool_keeps_sqlite_behaviour_for_seeded_devices(
    memory_session: Session, sample_adapter: MetroPT3Adapter
) -> None:
    """Registering an external source must not change the demo devices."""
    status = get_device_status("PLC-001", session=memory_session, adapters=(sample_adapter,))

    assert status.found is True
    assert status.device_name == "包装线PLC"
    assert status.temperature == 78.0
    assert status.alarm_code == "F0045"
    assert status.data_source is None
    assert status.measurements is None
    assert status.source_timestamp is None
    assert status.status_derivation is None


def test_tool_still_reports_unknown_devices(memory_session: Session) -> None:
    status = get_device_status("NOPE-999", session=memory_session)

    assert status.found is False
    assert status.error is None
    assert status.device_name is None


def test_external_device_id_does_not_collide_with_seeded_ids(memory_session: Session) -> None:
    seeded = set(memory_session.scalars(select(Device.device_id)))
    assert METROPT3_DEVICE_ID not in seeded


def test_optional_fields_serialize_to_json(
    sample_adapter: MetroPT3Adapter,
) -> None:
    """The executor serializes with ``mode="json"``; the new fields must survive it."""
    status = get_device_status(METROPT3_DEVICE_ID, adapters=(sample_adapter,))
    payload = status.model_dump(mode="json")

    assert payload["source_timestamp"] == "2020-01-01T00:00:40"
    assert payload["data_source"] == METROPT3_SOURCE_ID
    assert payload["measurements"]["motor_current_a"] == 7.0
    assert payload["digital_signals"]["COMP"] == 0.0
    assert isinstance(payload["status_derivation"], str)


def test_seeded_output_leaves_optional_fields_unset() -> None:
    payload = DeviceStatusOutput(device_id="PLC-001").model_dump(mode="json")

    assert payload["data_source"] is None
    assert payload["source_timestamp"] is None
    assert payload["measurements"] is None
    assert payload["digital_signals"] is None


# --------------------------------------------------------------------------- #
# Configuration and contract stability
# --------------------------------------------------------------------------- #


def test_external_source_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("METROPT3_CSV_PATH", raising=False)
    get_settings.cache_clear()

    settings = Settings()
    assert settings.metropt3_csv_path == ""
    assert build_adapters(settings) == ()
    reset_device_adapters()
    assert get_device_adapters() == ()


def test_configured_path_enables_the_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv_path = _write_csv(tmp_path / "configured.csv", [_row(0, "2020-01-01 00:00:00")])
    monkeypatch.setenv("METROPT3_CSV_PATH", str(csv_path))
    get_settings.cache_clear()
    reset_device_adapters()

    adapters = get_device_adapters()
    assert len(adapters) == 1
    assert resolve_adapter(METROPT3_DEVICE_ID) is adapters[0]

    status = get_device_status(METROPT3_DEVICE_ID)
    assert status.found is True
    assert status.data_source == METROPT3_SOURCE_ID


def test_tool_contract_is_unchanged() -> None:
    """No new tool, and no new parameter: the planner sees exactly what it saw."""
    assert len(TOOL_NAMES) == 3
    assert set(registry.names()) == set(TOOL_NAMES)
    assert not any("metro" in name.lower() for name in registry.names())

    spec = registry.get("get_device_status")
    assert spec.input_model is DeviceStatusInput
    assert set(DeviceStatusInput.model_json_schema()["properties"]) == {"device_id"}

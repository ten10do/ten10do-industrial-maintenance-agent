"""MetroPT-3 real-world sensor adapter.

Source
------
UCI Machine Learning Repository, dataset id 791, "MetroPT-3".
DOI ``10.24432/C5VW3R``. Licensed CC BY 4.0.
Collected from the Air Production Unit (APU) compressor of a metro train in an
operational context, logged at 1 Hz from February to August 2020.

This module reads a *local* copy of the CSV. It never downloads anything, and
the repository does not redistribute the dataset: the extracted file is about
208 MiB and stays out of version control. Point ``METROPT3_CSV_PATH`` at your own
download. See ``docs/data/metropt3.md``.

Reading strategy
----------------
The file is ~208 MiB across ~1.5M rows, so it is never parsed whole to answer a
status query. The adapter:

* reads only the header line to validate the column set;
* seeks to the end of the file and reads a bounded tail to find the last
  complete record, growing the window only when the tail lands inside one record;
* caches the parsed snapshot against the file's ``(size, mtime_ns)`` stamp, so
  repeated queries are served from memory and a replaced file invalidates the
  cache by itself.

Peak memory is therefore one tail window, not the file.

Status is derived, never invented
---------------------------------
The dataset publishes sensor readings and no ground-truth equipment state, so
``status`` is an operational state derived from documented contact signals and
``status_derivation`` names the basis. No fault, degradation or remaining-useful-
life claim is made anywhere in this module. Fields the dataset does not carry,
such as rotational speed or an alarm code, stay ``None``.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.integrations.device_data.base import (
    DeviceDataAdapter,
    DeviceDataSourceError,
    DeviceSnapshot,
)

#: Identifier the agent exposes for this source. Chosen so it cannot collide
#: with the seeded demo identifiers (PLC-001, Robot-001, ...).
METROPT3_DEVICE_ID = "METRO-APU-001"
METROPT3_DEVICE_NAME = "MetroPT-3 Air Production Unit"
METROPT3_DEVICE_TYPE = "air_compressor"
METROPT3_LOCATION = "Metro train air production unit (per dataset description)"

#: Traceable name for this data source. It travels on the tool output and on the
#: evidence record, so a fact can be attributed to the dataset it came from.
METROPT3_SOURCE_ID = "uci_metropt3"

#: Columns that must be present for the file to be treated as MetroPT-3. Taken
#: from the header of the published CSV; a superset is accepted so a future
#: revision that adds a column still loads.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)

#: Analogue output key -> source column. The unit is the one the dataset
#: documentation states for that column and is carried in the key itself.
MEASUREMENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("TP2_bar", "TP2"),
    ("TP3_bar", "TP3"),
    ("H1_bar", "H1"),
    ("DV_pressure_bar", "DV_pressure"),
    ("reservoir_pressure_bar", "Reservoirs"),
    ("oil_temperature_c", "Oil_temperature"),
    ("motor_current_a", "Motor_current"),
)

#: Digital contact signals, kept under their published column names. Renaming
#: them would invent a vocabulary the dataset does not use.
DIGITAL_COLUMNS: tuple[str, ...] = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)

# Derived operational states. ``observed`` is the honest fallback used when the
# documented contacts are not in a state the documentation describes.
STATUS_RUNNING_UNDER_LOAD = "running_under_load"
STATUS_RUNNING_OFFLOADED = "running_offloaded"
STATUS_STOPPED = "stopped"
STATUS_OBSERVED = "observed"

#: Names the exact basis of the derived label so it is never read as ground truth.
STATUS_DERIVATION = (
    "derived: operational state from the documented control contacts "
    "DV_eletric (active while the compressor runs under load) and COMP "
    "(active while there is no air intake, i.e. off or offloaded), corroborated "
    "by the published motor-current bands. The dataset publishes no ground-truth "
    "equipment state and no fault label is inferred."
)

#: Motor current that separates "off" from "offloaded" in the published bands
#: (about 0 A when off, about 4 A when offloaded). Half an ampere sits between
#: the two and is far below either.
MOTOR_CURRENT_OFF_THRESHOLD_A = 0.5

_CONTACT_ACTIVE = 1.0

#: Bytes read from the end of the file when hunting for the last record. One row
#: is roughly 150 bytes, so this holds many rows and is grown only if a single
#: record is longer than the whole window.
TAIL_WINDOW_BYTES = 1 << 16

_TAIL_WINDOW_GROWTH = 4

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _decode(raw: bytes) -> str:
    """Decode one CSV line, dropping a BOM and any carriage return."""
    return raw.decode("utf-8-sig").strip()


def _read_columns(handle: io.BufferedReader) -> tuple[str, list[str]]:
    """Read and validate the header line.

    Returns the raw header text alongside the parsed columns, so the caller can
    recognise a file whose only line is the header.
    """
    handle.seek(0)
    header = _decode(handle.readline())
    if not header:
        raise DeviceDataSourceError("MetroPT-3 CSV is empty: no header line.")

    columns = next(csv.reader([header]))
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise DeviceDataSourceError(
            "MetroPT-3 CSV header is missing expected columns: " + ", ".join(missing) + "."
        )
    return header, columns


def _read_last_record(handle: io.BufferedReader, size: int) -> str | None:
    """Return the last complete line of the file without reading all of it."""
    window = min(TAIL_WINDOW_BYTES, size)
    while True:
        handle.seek(size - window)
        chunk = handle.read(window)
        lines = chunk.split(b"\n")
        # A window that does not start at byte 0 begins mid-record; drop that
        # fragment. The final element is empty whenever the file ends in a
        # newline, which is the normal case.
        if size - window > 0:
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            return _decode(lines[-1])
        if window >= size:
            return None
        window = min(window * _TAIL_WINDOW_GROWTH, size)


def _as_float(value: str) -> float | None:
    """Convert a CSV field to a float, or ``None`` when it is blank."""
    text = value.strip()
    if not text:
        return None
    return float(text)


def _derive_status(
    dv_electric: float | None,
    comp: float | None,
    motor_current: float | None,
) -> str:
    """Return the derived operational state for one record.

    The rule is the documented behaviour of two contact signals, cross-checked
    against the published motor-current bands. It classifies *operating state*
    only; it makes no claim about health or failure.
    """
    if dv_electric == _CONTACT_ACTIVE:
        return STATUS_RUNNING_UNDER_LOAD
    if comp == _CONTACT_ACTIVE:
        if motor_current is not None and motor_current < MOTOR_CURRENT_OFF_THRESHOLD_A:
            return STATUS_STOPPED
        return STATUS_RUNNING_OFFLOADED
    return STATUS_OBSERVED


def _snapshot_from_record(columns: list[str], record: str) -> DeviceSnapshot:
    """Build one snapshot from a CSV record line."""
    values = next(csv.reader([record]))
    if len(values) != len(columns):
        raise DeviceDataSourceError(
            "MetroPT-3 CSV record does not match the header: "
            f"{len(values)} fields against {len(columns)} columns."
        )
    row = dict(zip(columns, values, strict=False))

    timestamp_text = row.get("timestamp", "").strip()
    if not timestamp_text:
        raise DeviceDataSourceError("MetroPT-3 CSV record has no timestamp.")
    try:
        timestamp = datetime.strptime(timestamp_text, _TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise DeviceDataSourceError(
            f"MetroPT-3 CSV timestamp is not in {_TIMESTAMP_FORMAT} form: {timestamp_text!r}."
        ) from exc

    measurements: dict[str, float | None] = {
        key: _as_float(row.get(column, "")) for key, column in MEASUREMENT_COLUMNS
    }
    digital_signals: dict[str, float | None] = {
        column: _as_float(row.get(column, "")) for column in DIGITAL_COLUMNS
    }

    return DeviceSnapshot(
        device_id=METROPT3_DEVICE_ID,
        device_name=METROPT3_DEVICE_NAME,
        device_type=METROPT3_DEVICE_TYPE,
        location=METROPT3_LOCATION,
        status=_derive_status(
            digital_signals.get("DV_eletric"),
            digital_signals.get("COMP"),
            measurements.get("motor_current_a"),
        ),
        status_derivation=STATUS_DERIVATION,
        source_timestamp=timestamp,
        data_source=METROPT3_SOURCE_ID,
        measurements=MappingProxyType(measurements),
        digital_signals=MappingProxyType(digital_signals),
    )


def _read_snapshot(resolved_path: str, size: int, mtime_ns: int) -> DeviceSnapshot:
    """Parse the last record of ``resolved_path``.

    ``size`` and ``mtime_ns`` are part of the cache key rather than used inside
    the body, so a file that changes on disk produces a different key and is
    re-read.
    """
    del size, mtime_ns
    path = Path(resolved_path)
    try:
        with path.open("rb") as handle:
            total = path.stat().st_size
            header_line, columns = _read_columns(handle)
            record = _read_last_record(handle, total)
    except OSError as exc:
        raise DeviceDataSourceError(
            f"MetroPT-3 CSV could not be read: {type(exc).__name__} while opening the file."
        ) from exc

    # A file holding nothing but its header would otherwise be parsed as a record
    # and fail later with a confusing field error. ``_read_last_record`` has no
    # way to tell the header apart from data, so the check belongs here.
    if record is None or record == header_line:
        raise DeviceDataSourceError("MetroPT-3 CSV contains a header but no data records.")
    return _snapshot_from_record(columns, record)


@lru_cache(maxsize=8)
def _cached_snapshot(resolved_path: str, size: int, mtime_ns: int) -> DeviceSnapshot:
    """Return the parsed snapshot for a specific revision of the file."""
    return _read_snapshot(resolved_path, size, mtime_ns)


class MetroPT3Adapter(DeviceDataAdapter):
    """Device data source backed by a local MetroPT-3 CSV.

    Construction touches no filesystem state, so an incorrectly configured path
    only surfaces when a query actually needs the file.
    """

    source_id = METROPT3_SOURCE_ID

    def __init__(self, csv_path: str | Path) -> None:
        """Store the configured path.

        Args:
            csv_path: Location of the extracted ``MetroPT3(AirCompressor).csv``.
                Typically read from ``METROPT3_CSV_PATH``.

        Raises:
            ValueError: when the path is empty, so a caller cannot silently end
                up with an adapter that serves nothing.
        """
        text = str(csv_path).strip()
        if not text:
            raise ValueError("MetroPT3Adapter requires a non-empty csv_path.")
        self._csv_path = Path(text).expanduser()

    @property
    def device_ids(self) -> frozenset[str]:
        """The single identifier this source exposes."""
        return frozenset({METROPT3_DEVICE_ID})

    @property
    def csv_path(self) -> Path:
        """The configured CSV location. Not logged; it is a local path."""
        return self._csv_path

    def get_latest_snapshot(self, device_id: str) -> DeviceSnapshot | None:
        """Return the last record of the dataset for ``METRO-APU-001``.

        Raises:
            DeviceDataSourceError: when the file is missing, unreadable, not a
                MetroPT-3 CSV, or holds no data records.
        """
        if device_id.strip().upper() not in {value.upper() for value in self.device_ids}:
            return None

        path = self._csv_path
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise DeviceDataSourceError(
                "MetroPT-3 CSV not found at the configured METROPT3_CSV_PATH. "
                "Download the dataset from UCI and point the variable at the "
                "extracted CSV; see docs/data/metropt3.md."
            ) from exc
        except OSError as exc:
            raise DeviceDataSourceError(
                f"MetroPT-3 CSV could not be inspected: {type(exc).__name__}."
            ) from exc

        return _cached_snapshot(str(path.resolve()), stat.st_size, stat.st_mtime_ns)

    def describe(self) -> dict[str, Any]:
        """Return non-secret metadata, without the configured local path."""
        return {
            "source_id": self.source_id,
            "dataset": "MetroPT-3",
            "dataset_doi": "10.24432/C5VW3R",
            "dataset_license": "CC BY 4.0",
            "device_ids": sorted(self.device_ids),
        }

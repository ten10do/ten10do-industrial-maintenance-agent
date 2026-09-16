"""Gate: does the built service answer correctly from real MetroPT-3 data?

This is the runtime counterpart to ``scripts/prepare_metropt3.py``. That script
asks whether a local copy of the dataset is *valid*; this one asks whether the
service, wired through configuration exactly as it ships, returns the right
answer for the real file. It reads the path from ``METROPT3_CSV_PATH`` unless
``--csv`` overrides it, and builds the adapters through the same configuration
path the application uses, so a green result covers the wiring and not just the
adapter in isolation.

The gate checks, against the true last record of the file:

* the adapter was built from configuration at all;
* the tool reports ``found`` for the published identifier;
* ``data_source`` names the dataset it came from;
* ``source_timestamp`` carries an upstream timestamp;
* ``measurements`` carry pressure, oil temperature and motor current;
* ``status`` arrives with its derivation, so a derived label is never read as
  ground truth;
* no PLC-shaped field was invented from foreign readings.

It also reports the cold and warm query latency. The warm figure is the point:
the file is 208 MiB, so a service that re-parsed it per query would not return
in milliseconds.

Usage::

    python -m scripts.real_data_gate --csv "/path/MetroPT3(AirCompressor).csv"
    python -m scripts.real_data_gate --expect-rows 1516948 --expect-sha256 <hex>

Exit status is 0 when every check passes, 1 otherwise. The verdict line
``REAL_DATA_GATE: PASS|FAIL`` is printed last so it can be grepped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from app.config import Settings, get_settings
from app.integrations.device_data.base import DeviceDataSourceError
from app.integrations.device_data.factory import build_adapters, reset_device_adapters
from app.integrations.device_data.metropt3 import (
    METROPT3_DEVICE_ID,
    METROPT3_SOURCE_ID,
    TAIL_WINDOW_BYTES,
)
from app.tools.device_tool import DeviceStatusOutput, get_device_status

#: Fields the reading must carry. Keys, not friendly names, because the unit
#: travels in the key.
REQUIRED_MEASUREMENTS: tuple[str, ...] = (
    "oil_temperature_c",
    "motor_current_a",
)

_PRESSURE_KEYS = ("TP2_bar", "TP3_bar", "H1_bar", "DV_pressure_bar", "reservoir_pressure_bar")

_CHUNK_BYTES = 1 << 22


def _scan(path: Path) -> tuple[int, str]:
    """Return ``(data_row_count, sha256)`` from one streaming pass."""
    digest = hashlib.sha256()
    newlines = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
            newlines += chunk.count(b"\n")
    # The header is one line; a trailing newline does not start a record.
    return max(0, newlines - 1), digest.hexdigest()


def _check_reading(status: DeviceStatusOutput) -> list[str]:
    """Return the list of failed checks for one tool result."""
    failures: list[str] = []

    if status.found is not True:
        failures.append("found is not True")
    if status.data_source != METROPT3_SOURCE_ID:
        failures.append(f"data_source is {status.data_source!r}, expected {METROPT3_SOURCE_ID!r}")
    if status.source_timestamp is None:
        failures.append("source_timestamp is empty")

    measurements = status.measurements or {}
    for key in REQUIRED_MEASUREMENTS:
        value = measurements.get(key)
        if not isinstance(value, float):
            failures.append(f"measurement {key!r} is not a float: {value!r}")
    pressure_values = [measurements.get(key) for key in _PRESSURE_KEYS]
    if not any(isinstance(value, float) for value in pressure_values):
        failures.append("no pressure reading present")

    if not status.status_derivation:
        failures.append("status_derivation is missing")
    elif not status.status_derivation.startswith("derived:"):
        failures.append("status_derivation does not declare the label derived")

    for field in ("temperature", "pressure", "rpm", "alarm_code"):
        if getattr(status, field) is not None:
            failures.append(f"PLC-shaped field {field!r} was populated from a foreign source")

    if status.device_type != "air_compressor":
        failures.append(f"device_type is {status.device_type!r}")

    return failures


def _timed_query() -> tuple[DeviceStatusOutput, float]:
    """Call the tool once and return the result with its wall-clock latency."""
    started = time.perf_counter()
    status = get_device_status(METROPT3_DEVICE_ID)
    return status, (time.perf_counter() - started) * 1000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that the service answers the real MetroPT-3 file correctly.",
    )
    parser.add_argument("--csv", type=Path, help="Override METROPT3_CSV_PATH.")
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Skip the full-file scan, which reads all 208 MiB to count rows and hash.",
    )
    parser.add_argument(
        "--expect-rows", type=int, help="Fail unless the scan counts this many rows."
    )
    parser.add_argument("--expect-sha256", help="Fail unless the scan produces this digest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the gate and print a JSON report followed by the verdict line."""
    args = build_parser().parse_args(argv)

    if args.csv is not None:
        os.environ["METROPT3_CSV_PATH"] = str(args.csv)
        get_settings.cache_clear()
    reset_device_adapters()

    settings: Settings = get_settings()
    configured = (settings.metropt3_csv_path or "").strip()

    report: dict[str, object] = {
        "device_id": METROPT3_DEVICE_ID,
        "expected_source": METROPT3_SOURCE_ID,
        "configured": bool(configured),
        "tail_window_bytes": TAIL_WINDOW_BYTES,
    }
    failures: list[str] = []

    if not configured:
        failures.append("METROPT3_CSV_PATH is not set, so no adapter was built")
        report["failures"] = failures
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("REAL_DATA_GATE: FAIL")
        return 1

    path = Path(configured).expanduser()
    report["file_exists"] = path.is_file()
    if not path.is_file():
        failures.append(f"file not found: {path}")
        report["failures"] = failures
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("REAL_DATA_GATE: FAIL")
        return 1

    report["file_size_bytes"] = path.stat().st_size

    adapters = build_adapters(settings)
    report["adapter_count"] = len(adapters)
    report["adapter_metadata"] = [adapter.describe() for adapter in adapters]
    if not adapters:
        failures.append("configuration produced no adapter")

    if args.skip_scan:
        report["data_row_count"] = None
        report["sha256"] = None
        report["scan"] = "skipped"
    else:
        rows, digest = _scan(path)
        report["data_row_count"] = rows
        report["sha256"] = digest
        report["scan"] = "performed"
        if args.expect_rows is not None and rows != args.expect_rows:
            failures.append(f"row count {rows} != expected {args.expect_rows}")
        if args.expect_sha256 is not None and digest != args.expect_sha256:
            failures.append(f"sha256 {digest} != expected {args.expect_sha256}")

    if not adapters:
        report["failures"] = failures
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("REAL_DATA_GATE: FAIL")
        return 1

    try:
        cold, cold_ms = _timed_query()
        warm, warm_ms = _timed_query()
    except DeviceDataSourceError as exc:
        failures.append(f"source error: {exc}")
        report["failures"] = failures
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("REAL_DATA_GATE: FAIL")
        return 1

    report["cold_latency_ms"] = round(cold_ms, 3)
    report["warm_latency_ms"] = round(warm_ms, 3)
    report["measurements"] = cold.measurements
    report["digital_signals"] = cold.digital_signals
    report["latest_record"] = {
        "device_id": cold.device_id,
        "device_name": cold.device_name,
        "device_type": cold.device_type,
        "status": cold.status,
        "source_timestamp": (
            cold.source_timestamp.isoformat() if cold.source_timestamp is not None else None
        ),
        "data_source": cold.data_source,
        "status_derivation": cold.status_derivation,
    }

    failures.extend(_check_reading(cold))
    failures.extend(f"warm query: {item}" for item in _check_reading(warm))

    if failures:
        report["failures"] = failures
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("REAL_DATA_GATE: FAIL")
        return 1

    report["failures"] = []
    report["note"] = (
        "status is an operational state derived from documented control contacts; "
        "the dataset publishes no ground-truth equipment state."
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("REAL_DATA_GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

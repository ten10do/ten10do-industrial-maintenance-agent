"""Verify a local MetroPT-3 CSV before the service is pointed at it.

The dataset is downloaded by hand from UCI. Nothing here performs a network
request: ``--download-info`` prints where to get the archive and stops. The
running service reads only ``METROPT3_CSV_PATH``, so downloading and serving
stay decoupled by design.

What the check does:

* confirms the file exists and is readable;
* validates the header against the columns the adapter requires, by calling the
  adapter itself, so there is no second column list to drift;
* reports the file size, a streaming SHA-256, the data row count and the
  timestamp range;
* reads the last record through the adapter and reports what the service would
  answer, including the derived status and its basis.

Usage::

    python -m scripts.prepare_metropt3 --input "/path/MetroPT3(AirCompressor).csv"
    python -m scripts.prepare_metropt3 --download-info

The report is JSON on stdout. Exit status is 0 when the file is usable and 1
when it is not, so the command can gate a pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from app.integrations.device_data.base import DeviceDataSourceError
from app.integrations.device_data.metropt3 import (
    METROPT3_DEVICE_ID,
    REQUIRED_COLUMNS,
    MetroPT3Adapter,
)

#: Official location. Recorded here so the user does not have to trust a search
#: result, and so it is obvious the archive comes from the publisher.
DATASET_NAME = "MetroPT-3"
DATASET_UCI_ID = "791"
DATASET_DOI = "10.24432/C5VW3R"
DATASET_LICENSE = "CC BY 4.0"
DATASET_LANDING_PAGE = "https://archive.ics.uci.edu/dataset/791/metropt+3+dataset"
DATASET_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"

_CHUNK_BYTES = 1 << 22


def _provenance() -> dict[str, str]:
    """Return the publisher attribution as data, so every report carries it."""
    return {
        "dataset": DATASET_NAME,
        "publisher": "UCI Machine Learning Repository",
        "uci_dataset_id": DATASET_UCI_ID,
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "landing_page": DATASET_LANDING_PAGE,
        "archive_url": DATASET_ARCHIVE_URL,
        "redistributed_by_this_repository": "no",
    }


def _count_rows_and_hash(path: Path) -> tuple[int, str, int]:
    """Return ``(data_row_count, sha256, size_bytes)`` in one streaming pass."""
    digest = hashlib.sha256()
    size = 0
    newlines = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            newlines += chunk.count(b"\n")
    # One line is the header; a trailing newline does not start a new record.
    data_rows = max(0, newlines - 1)
    return data_rows, digest.hexdigest(), size


def _first_timestamp(path: Path) -> str | None:
    """Return the timestamp on the first data line."""
    with path.open("r", encoding="utf-8-sig") as handle:
        handle.readline()
        line = handle.readline()
    if not line:
        return None
    fields = line.rstrip("\n").split(",")
    return fields[1].strip() if len(fields) > 1 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a local MetroPT-3 CSV before pointing the service at it.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the extracted MetroPT3(AirCompressor).csv.",
    )
    parser.add_argument(
        "--download-info",
        action="store_true",
        help="Print the official UCI location and exit. Downloads nothing.",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip the SHA-256 pass, which reads the whole file once.",
    )
    return parser


def _describe_expectation() -> list[str]:
    return list(REQUIRED_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    """Run the check and print a JSON report."""
    args = build_parser().parse_args(argv)

    report: dict[str, object] = {
        "provenance": _provenance(),
        "expected_columns": _describe_expectation(),
        "device_id": METROPT3_DEVICE_ID,
    }

    if args.download_info:
        report["status"] = "DOWNLOAD_INFO"
        report["note"] = (
            "Download the archive from the publisher, extract it, then run this "
            "script again with --input. This script never downloads."
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.input is None:
        report["status"] = "NO_INPUT"
        report["note"] = "Pass --input, or --download-info to see where to get the data."
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    path = args.input.expanduser()
    if not path.is_file():
        report["status"] = "FILE_NOT_FOUND"
        report["input"] = str(path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    report["file_name"] = path.name
    report["file_size_bytes"] = path.stat().st_size

    if not args.skip_hash:
        rows, digest, size = _count_rows_and_hash(path)
        report["data_row_count"] = rows
        report["sha256"] = digest
        report["file_size_bytes"] = size
    else:
        with path.open("rb") as handle:
            rows = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""))
        report["data_row_count"] = max(0, rows - 1)
        report["sha256"] = None

    report["first_timestamp"] = _first_timestamp(path)

    try:
        snapshot = MetroPT3Adapter(csv_path=path).get_latest_snapshot(METROPT3_DEVICE_ID)
    except DeviceDataSourceError as exc:
        report["status"] = "INVALID_DATASET"
        report["error"] = str(exc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    except ValueError as exc:
        report["status"] = "INVALID_DATASET"
        report["error"] = str(exc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    if snapshot is None:
        report["status"] = "INVALID_DATASET"
        report["error"] = "The adapter did not accept the device identifier it owns."
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    report["status"] = "OK"
    report["latest_record"] = {
        "source_timestamp": snapshot.source_timestamp.isoformat(),
        "status": snapshot.status,
        "status_derivation": snapshot.status_derivation,
        "data_source": snapshot.data_source,
        "measurements": dict(snapshot.measurements),
        "digital_signals": dict(snapshot.digital_signals),
    }
    report["note"] = (
        "status is an operational state derived from documented control contacts; "
        "the dataset publishes no ground-truth equipment state."
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

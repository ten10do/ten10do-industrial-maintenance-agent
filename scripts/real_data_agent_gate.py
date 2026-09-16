"""Gate: does the running agent answer correctly for the real device?

Calls ``POST /agent/invoke`` against a live service and checks the response
against what the real MetroPT-3 record implies. The service is driven over HTTP
rather than in-process, so this exercises the whole path a user would take:
routing, the rule planner, the tool, evidence construction and answer rendering.

What it proves:

* the request succeeds and the parser resolved the full identifier, which a
  truncated match would not;
* exactly the device tool was planned and run;
* the answer attributes the reading to the dataset it came from;
* the readings and the upstream timestamp appear in the answer;
* the derived status travels with its derivation;
* nothing PLC-shaped was invented. The demo record shape has temperature,
  pressure, rpm and an alarm code; the real record must not borrow them.

Usage::

    python -m scripts.real_data_agent_gate --base-url http://127.0.0.1:8111

Exit status is 0 when every check passes, 1 otherwise, and the verdict line
``REAL_DATA_AGENT_E2E_GATE: PASS|FAIL`` is printed last so it can be grepped.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx

from app.integrations.device_data.metropt3 import (
    DIGITAL_COLUMNS,
    MEASUREMENT_COLUMNS,
    METROPT3_DEVICE_ID,
    METROPT3_SOURCE_ID,
)

DEFAULT_QUERY = f"{METROPT3_DEVICE_ID} 当前设备状态怎么样？"

#: Keys the answer must show. Pressure is checked as a group because any one of
#: the pressure channels satisfies the requirement.
_REQUIRED_MEASUREMENT_KEYS = ("oil_temperature_c", "motor_current_a")

_PRESSURE_KEYS = tuple(
    key for key, _ in MEASUREMENT_COLUMNS if "pressure" in key or key.endswith("_bar")
)


def _check(payload: dict[str, Any]) -> list[str]:
    """Return the list of failed checks for one invoke response."""
    failures: list[str] = []
    answer = payload.get("answer") or ""

    if payload.get("intent") != "device_status":
        failures.append(f"intent is {payload.get('intent')!r}, expected 'device_status'")

    if payload.get("equipment_id") != METROPT3_DEVICE_ID:
        failures.append(
            f"equipment_id is {payload.get('equipment_id')!r}, expected {METROPT3_DEVICE_ID!r}"
        )

    called = payload.get("tools_called") or []
    if called != ["get_device_status"]:
        failures.append(f"tools_called is {called!r}, expected ['get_device_status']")

    # The planner is a deployment choice, so both answers are acceptable. A
    # silent fallback is not: that would mean the plan came from a planner the
    # deployment did not ask for.
    if payload.get("planner_used") not in {"rule", "llm"}:
        failures.append(
            f"planner_used is {payload.get('planner_used')!r}, expected 'rule' or 'llm'"
        )
    if payload.get("planner_fallback"):
        failures.append("planner fell back, so the plan did not come from the requested planner")

    evidence = payload.get("evidence") or []
    sources = {item.get("source") for item in evidence if isinstance(item, dict)}
    if METROPT3_SOURCE_ID not in sources:
        failures.append(
            f"no evidence attributed to {METROPT3_SOURCE_ID!r}; sources={sorted(sources)}"
        )

    if METROPT3_DEVICE_ID not in answer:
        failures.append("answer does not name the device")
    if METROPT3_SOURCE_ID not in answer:
        failures.append("answer does not state the data source")
    if "derived:" not in answer:
        failures.append("answer does not carry the status derivation")

    for key in _REQUIRED_MEASUREMENT_KEYS:
        if key not in answer:
            failures.append(f"answer is missing measurement {key!r}")
    if not any(key in answer for key in _PRESSURE_KEYS):
        failures.append("answer is missing a pressure reading")

    if "数据时间戳" not in answer:
        failures.append("answer does not carry the upstream timestamp")

    # The demo record shape must not be borrowed by a foreign source.
    if "关键状态：" in answer:
        failures.append("answer invented PLC-shaped temperature/pressure/rpm fields")
    if "报警码：无" not in answer:
        failures.append("answer does not report the absent alarm code explicitly")
    for column in DIGITAL_COLUMNS:
        if column not in answer:
            failures.append(f"answer is missing digital signal {column!r}")

    if not isinstance(payload.get("latency_ms"), int | float):
        failures.append("latency_ms is missing")

    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a running agent against the real MetroPT-3 device.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running service.",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query to send.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the gate and print a JSON report followed by the verdict line."""
    args = build_parser().parse_args(argv)

    report: dict[str, object] = {
        "base_url": args.base_url,
        "query": args.query,
        "expected_intent": "device_status",
        "expected_tools": ["get_device_status"],
        "expected_equipment_id": METROPT3_DEVICE_ID,
    }
    failures: list[str] = []

    url = args.base_url.rstrip("/") + "/agent/invoke"
    try:
        response = httpx.post(
            url,
            json={"query": args.query, "debug": True},
            timeout=args.timeout,
        )
    except httpx.HTTPError as exc:
        failures.append(f"request failed: {type(exc).__name__}")
        report["failures"] = failures
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("REAL_DATA_AGENT_E2E_GATE: FAIL")
        return 1

    report["http_status"] = response.status_code
    if response.status_code != 200:
        failures.append(f"HTTP {response.status_code}")
        report["body"] = response.text[:2000]
        report["failures"] = failures
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("REAL_DATA_AGENT_E2E_GATE: FAIL")
        return 1

    payload: dict[str, Any] = response.json()
    report["request_id"] = payload.get("request_id")
    report["intent"] = payload.get("intent")
    report["equipment_id"] = payload.get("equipment_id")
    report["tools_called"] = payload.get("tools_called")
    report["planner_used"] = payload.get("planner_used")
    report["planner_fallback"] = payload.get("planner_fallback")
    report["latency_ms"] = payload.get("latency_ms")
    report["evidence_sources"] = [
        item.get("source") for item in (payload.get("evidence") or []) if isinstance(item, dict)
    ]
    report["answer"] = payload.get("answer")
    debug_info = payload.get("debug_info") or {}
    report["debug_required_tools"] = debug_info.get("required_tools")
    report["debug_tool_status"] = debug_info.get("tool_status")

    tool_status = debug_info.get("tool_status") or []
    if not any(
        isinstance(item, dict) and item.get("tool") == "get_device_status" and item.get("found")
        for item in tool_status
    ):
        failures.append("debug tool_status does not report get_device_status as found")

    failures.extend(_check(payload))

    if failures:
        report["failures"] = failures
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("REAL_DATA_AGENT_E2E_GATE: FAIL")
        return 1

    report["failures"] = []
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("REAL_DATA_AGENT_E2E_GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

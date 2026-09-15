"""Rule-based natural language query parser.

Deterministic keyword and pattern matching only. No LLM, no network access, no
embeddings.

The parser owns text parsing. Everything about device identifiers and device
name aliases lives in :mod:`app.services.device_catalog`, so this module never
touches the catalog file directly.

Extraction strategy:

1. A ``PREFIX-123`` pattern is matched and canonicalized through the device
   catalog, so ``plc-001`` and ``robot-001`` map back to ``PLC-001`` and
   ``Robot-001``.
2. If no identifier pattern is present, device names are resolved as aliases, so
   ``包装线PLC`` resolves to ``PLC-001``.
3. Alarm codes are matched by the ``LETTER1234`` pattern and upper-cased.

When several candidates are present the first match wins.
"""

import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from app.services.device_catalog import canonicalize_device_id, resolve_device_alias

# ASCII-aware boundaries. ``\b`` cannot be used because Python treats CJK
# characters as word characters, so "F0045怎么办" would not match at all.
_DEVICE_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,}-\d{3,})(?![A-Za-z0-9])")
_ALARM_CODE_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]\d{4})(?![A-Za-z0-9])")

ALARM_KEYWORDS = ("报警", "告警", "警报", "故障", "alarm", "alert", "fault")
MAINTENANCE_KEYWORDS = (
    "怎么办",
    "怎么处理",
    "如何",
    "处理",
    "维修",
    "检修",
    "保养",
    "维护",
    "排查",
    "诊断",
    "原因",
    "how",
    "fix",
    "repair",
    "maintain",
)
STATUS_KEYWORDS = (
    "状态",
    "运行",
    "温度",
    "压力",
    "转速",
    "参数",
    "status",
    "temperature",
    "pressure",
    "rpm",
)


class Intent(str, Enum):
    """Coarse request classification produced by the parser."""

    ALARM_DIAGNOSIS = "alarm_diagnosis"
    MAINTENANCE_ADVICE = "maintenance_advice"
    DEVICE_STATUS = "device_status"
    UNKNOWN = "unknown"


class ParsedQuery(BaseModel):
    """Structured parser output."""

    equipment_id: str | None = Field(
        default=None,
        description="Canonical device identifier, for example PLC-001.",
    )
    alarm_code: str | None = Field(
        default=None,
        description="Upper-cased alarm code, for example F0045.",
    )
    intent: Intent = Field(
        default=Intent.UNKNOWN,
        description="Coarse classification of the request.",
    )


def _extract_equipment_id(text: str, devices_path: Path | str | None) -> str | None:
    """Return the first device identifier or device name found in ``text``."""
    match = _DEVICE_ID_PATTERN.search(text)
    if match:
        return canonicalize_device_id(match.group(1), devices_path)
    return resolve_device_alias(text, devices_path)


def _extract_alarm_code(text: str) -> str | None:
    """Return the first alarm code found in ``text``, upper-cased."""
    match = _ALARM_CODE_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).upper()


def _detect_intent(
    text: str,
    equipment_id: str | None,
    alarm_code: str | None,
) -> Intent:
    """Classify the request using a fixed precedence order."""
    lowered = text.lower()

    if alarm_code or any(keyword in lowered for keyword in ALARM_KEYWORDS):
        return Intent.ALARM_DIAGNOSIS
    if any(keyword in lowered for keyword in MAINTENANCE_KEYWORDS):
        return Intent.MAINTENANCE_ADVICE
    if equipment_id or any(keyword in lowered for keyword in STATUS_KEYWORDS):
        return Intent.DEVICE_STATUS
    return Intent.UNKNOWN


def parse_query(text: str, devices_path: Path | str | None = None) -> ParsedQuery:
    """Parse a natural language maintenance query.

    Args:
        text: Raw user input, for example ``"包装线PLC报警F0045怎么办"``.
        devices_path: Optional device catalog override, mainly for tests.

    Returns:
        A :class:`ParsedQuery` with ``equipment_id``, ``alarm_code`` and
        ``intent``. Any field the rules cannot resolve stays ``None``.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return ParsedQuery(intent=Intent.UNKNOWN)

    equipment_id = _extract_equipment_id(normalized, devices_path)
    alarm_code = _extract_alarm_code(normalized)

    return ParsedQuery(
        equipment_id=equipment_id,
        alarm_code=alarm_code,
        intent=_detect_intent(normalized, equipment_id, alarm_code),
    )

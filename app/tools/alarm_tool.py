"""Alarm code lookup tool backed by the local ``data/alarms.json`` catalog.

Exposes :func:`query_alarm_code` with explicit Pydantic input and output
models. The catalog is a plain JSON file so it can be edited without a code
change. Codes are matched case-insensitively.

This tool performs no LLM call.
"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALARMS_PATH = PROJECT_ROOT / "data" / "alarms.json"


class AlarmQueryInput(BaseModel):
    """Input model for :func:`query_alarm_code`."""

    alarm_code: str = Field(
        ...,
        min_length=1,
        description="Alarm code to look up, for example F0045.",
    )


class AlarmQueryOutput(BaseModel):
    """Structured alarm description.

    ``found`` is ``False`` when the code is absent from the catalog, in which
    case every descriptive field stays empty.
    """

    alarm_code: str
    found: bool = True
    name: str | None = None
    description: str | None = None
    severity: str | None = None
    category: str | None = None
    possible_causes: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


def normalize_alarm_code(alarm_code: str) -> str:
    """Normalize an alarm code for catalog lookup."""
    return alarm_code.strip().upper()


@lru_cache(maxsize=4)
def _load_index(resolved_path: str) -> dict[str, AlarmQueryOutput]:
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Alarm catalog not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Alarm catalog must contain a JSON list: {path}")

    index: dict[str, AlarmQueryOutput] = {}
    for record in payload:
        code = normalize_alarm_code(str(record.get("alarm_code", "")))
        if not code:
            continue
        index[code] = AlarmQueryOutput(
            alarm_code=code,
            found=True,
            name=record.get("name"),
            description=record.get("description"),
            severity=record.get("severity"),
            category=record.get("category"),
            possible_causes=list(record.get("possible_causes") or []),
            recommended_actions=list(record.get("recommended_actions") or []),
        )
    return index


def load_alarm_catalog(
    alarms_path: Path | str = DEFAULT_ALARMS_PATH,
) -> dict[str, AlarmQueryOutput]:
    """Return the alarm catalog indexed by normalized alarm code."""
    return _load_index(str(Path(alarms_path).resolve()))


def query_alarm_code(
    alarm_code: str,
    alarms_path: Path | str | None = None,
) -> AlarmQueryOutput:
    """Look up a single alarm code in the local catalog.

    Args:
        alarm_code: Code to look up, for example ``F0045``.
        alarms_path: Optional catalog path override, mainly for tests.
    """
    payload = AlarmQueryInput(alarm_code=alarm_code)
    normalized = normalize_alarm_code(payload.alarm_code)
    catalog = load_alarm_catalog(alarms_path or DEFAULT_ALARMS_PATH)
    return catalog.get(normalized, AlarmQueryOutput(alarm_code=normalized, found=False))

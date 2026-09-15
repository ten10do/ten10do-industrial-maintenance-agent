"""Device catalog service.

Owns every piece of knowledge about the device catalog: where the file lives,
how identifiers are canonicalized and how device names map to identifiers. The
query parser delegates here, so the parser only has to deal with text.

Kept free of agent-layer imports so it can be reused by API endpoints and tools.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEVICES_PATH = PROJECT_ROOT / "data" / "devices.json"


@dataclass(frozen=True)
class DeviceCatalog:
    """Lookup tables built from the device catalog file."""

    by_id: dict[str, str]
    by_name: dict[str, str]
    names_by_length: list[str]
    names_by_id: dict[str, str]


@lru_cache(maxsize=4)
def _load_catalog(resolved_path: str) -> DeviceCatalog:
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Device catalog not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Device catalog must contain a JSON list: {path}")

    by_id: dict[str, str] = {}
    by_name: dict[str, str] = {}
    names_by_id: dict[str, str] = {}
    for record in payload:
        device_id = str(record.get("device_id", "")).strip()
        if not device_id:
            continue
        by_id[device_id.lower()] = device_id
        device_name = record.get("device_name")
        if device_name:
            display = str(device_name).strip()
            by_name[display.lower()] = device_id
            names_by_id[device_id] = display

    return DeviceCatalog(
        by_id=by_id,
        by_name=by_name,
        names_by_length=sorted(by_name, key=len, reverse=True),
        names_by_id=names_by_id,
    )


def load_device_catalog(devices_path: Path | str = DEFAULT_DEVICES_PATH) -> DeviceCatalog:
    """Return the indexed device catalog."""
    return _load_catalog(str(Path(devices_path).resolve()))


def canonicalize_device_id(candidate: str, devices_path: Path | str | None = None) -> str:
    """Map a pattern-matched candidate to its canonical casing.

    ``plc-001`` becomes ``PLC-001`` and ``robot-001`` becomes ``Robot-001``.
    Unknown identifiers are returned unchanged so a caller can still report the
    device as missing rather than losing the user's input.
    """
    catalog = load_device_catalog(devices_path or DEFAULT_DEVICES_PATH)
    normalized = candidate.strip()
    return catalog.by_id.get(normalized.lower(), normalized)


def resolve_device_alias(text: str, devices_path: Path | str | None = None) -> str | None:
    """Return the equipment id whose device name appears in ``text``.

    Names are matched longest first so a shorter name cannot win over a longer
    one that overlaps it.
    """
    catalog = load_device_catalog(devices_path or DEFAULT_DEVICES_PATH)
    lowered = text.lower()
    for name in catalog.names_by_length:
        if name and name in lowered:
            return catalog.by_name[name]
    return None


def device_vocabulary(devices_path: Path | str | None = None) -> list[tuple[str, str]]:
    """Return ``(device_id, display_name)`` pairs, ordered by identifier.

    This is the identifier space a caller may pick from. The LLM planner uses it
    as prompt context so an argument means the canonical identifier rather than
    whatever the user typed. It is vocabulary, not evidence: nothing here answers
    a question or replaces a tool result.
    """
    catalog = load_device_catalog(devices_path or DEFAULT_DEVICES_PATH)
    identifiers = sorted(set(catalog.by_id.values()))
    return [(device_id, catalog.names_by_id.get(device_id, "")) for device_id in identifiers]

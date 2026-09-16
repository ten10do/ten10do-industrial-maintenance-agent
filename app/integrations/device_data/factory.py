"""Source selection for device status reads.

Adapters are chosen by configuration and built lazily. No adapter opens a file
at import time, so the service starts, and the seeded SQLite devices keep
working, whether or not an external dataset is present.

An unconfigured external source is simply absent from the adapter list. That
keeps "this deployment does not carry that device" separate from "the device is
unknown", and it means the default installation behaves exactly as before.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import Settings, get_settings
from app.integrations.device_data.base import DeviceDataAdapter
from app.integrations.device_data.metropt3 import MetroPT3Adapter

if TYPE_CHECKING:
    from collections.abc import Iterable

#: External sources this build can serve, in configuration order.
SUPPORTED_SOURCES: tuple[str, ...] = ("metropt3",)


def build_adapters(settings: Settings) -> tuple[DeviceDataAdapter, ...]:
    """Construct the external adapters the settings enable.

    A source with no configured location is skipped rather than added in a
    half-configured state, so it can neither answer nor fail.
    """
    adapters: list[DeviceDataAdapter] = []

    csv_path = (settings.metropt3_csv_path or "").strip()
    if csv_path:
        adapters.append(MetroPT3Adapter(csv_path=csv_path))

    return tuple(adapters)


@lru_cache(maxsize=1)
def get_device_adapters() -> tuple[DeviceDataAdapter, ...]:
    """Return the configured external adapters, built once per process."""
    return build_adapters(get_settings())


def reset_device_adapters() -> None:
    """Drop the cached adapters so the next call rebuilds them."""
    get_device_adapters.cache_clear()


def resolve_adapter(
    device_id: str,
    adapters: Iterable[DeviceDataAdapter] | None = None,
) -> DeviceDataAdapter | None:
    """Return the external adapter that claims ``device_id``.

    Returns ``None`` when no external source claims it, which is the signal for
    the caller to use the built-in SQLite source. Comparison is
    case-insensitive.
    """
    candidate = device_id.strip().upper()
    for adapter in adapters if adapters is not None else get_device_adapters():
        if candidate in {value.upper() for value in adapter.device_ids}:
            return adapter
    return None

"""External device data sources.

The built-in device source is the seeded SQLite table. This package adds
optional external sources that answer for their own identifiers, behind the
:class:`~app.integrations.device_data.base.DeviceDataAdapter` boundary.

Importing this package opens nothing and reads no configuration: adapters are
built on first use through :func:`get_device_adapters`.
"""

from app.integrations.device_data.base import (
    DeviceDataAdapter,
    DeviceDataSourceError,
    DeviceSnapshot,
)
from app.integrations.device_data.factory import (
    SUPPORTED_SOURCES,
    build_adapters,
    get_device_adapters,
    reset_device_adapters,
    resolve_adapter,
)
from app.integrations.device_data.metropt3 import (
    METROPT3_DEVICE_ID,
    METROPT3_SOURCE_ID,
    MetroPT3Adapter,
)

__all__ = [
    "METROPT3_DEVICE_ID",
    "METROPT3_SOURCE_ID",
    "SUPPORTED_SOURCES",
    "DeviceDataAdapter",
    "DeviceDataSourceError",
    "DeviceSnapshot",
    "MetroPT3Adapter",
    "build_adapters",
    "get_device_adapters",
    "reset_device_adapters",
    "resolve_adapter",
]

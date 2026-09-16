"""Device status tool.

Exposes :func:`get_device_status` together with explicit Pydantic input and
output models, so the tool can be bound to a LangGraph node or to an LLM
function-calling interface without leaking ORM objects across the boundary.

Two sources answer this one tool, selected by identifier:

* the seeded SQLite ``Device`` table, which holds the deterministic demo
  records and is the default for every identifier it owns;
* any configured external adapter, such as the MetroPT-3 real-world sensor
  source, which answers for its own identifiers only.

The input schema, the registered tool name and the planner are untouched. A
device that no external source claims keeps the exact behaviour it had before
this module learned about external sources.

Three failure modes stay distinguishable, following the manual-retrieval tool:

* the device is unknown -> ``found`` is ``False`` and ``error`` is ``None``;
* an external source is configured for the device but unreadable -> ``found`` is
  ``False`` and ``error`` carries the source diagnostic;
* the device is found -> ``found`` is ``True``.

Fields the source does not carry are left ``None``. Nothing is invented to make
a real dataset look like a PLC record.

This tool performs no LLM call.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import session as db_session
from app.database.models import Device
from app.integrations.device_data.base import DeviceDataSourceError, DeviceSnapshot
from app.integrations.device_data.factory import resolve_adapter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from app.integrations.device_data.base import DeviceDataAdapter


class DeviceStatusInput(BaseModel):
    """Input model for :func:`get_device_status`."""

    device_id: str = Field(
        ...,
        min_length=1,
        description="Business identifier of the device, for example PLC-001.",
    )


class DeviceStatusOutput(BaseModel):
    """Structured device status.

    ``found`` is ``False`` when the identifier is not present in any source, in
    which case every descriptive field stays ``None``.

    The fields below ``last_maintenance_time`` are optional and were added for
    external real-world sources. They are ``None`` for the seeded SQLite
    devices, whose semantics are unchanged.
    """

    device_id: str
    found: bool = True
    device_name: str | None = None
    device_type: str | None = None
    location: str | None = None
    status: str | None = None
    temperature: float | None = None
    pressure: float | None = None
    rpm: float | None = None
    alarm_code: str | None = None
    last_maintenance_time: datetime | None = None

    # Trailing optional block for external sources. ``temperature``, ``pressure``,
    # ``rpm`` and ``alarm_code`` above are deliberately left unset for such a
    # source: filling them would fold foreign measurements into a PLC-shaped
    # record. The readings travel in ``measurements`` under their own names.
    data_source: str | None = Field(
        default=None,
        description="Traceable name of the source, for example uci_metropt3.",
    )
    source_timestamp: datetime | None = Field(
        default=None,
        description="Timestamp the reading carries upstream, not the request time.",
    )
    measurements: dict[str, float | None] | None = Field(
        default=None,
        description="Analogue readings keyed by measurement name, unit in the key.",
    )
    digital_signals: dict[str, float | None] | None = Field(
        default=None,
        description="Digital contact signals under their published column names.",
    )
    status_derivation: str | None = Field(
        default=None,
        description=(
            "How ``status`` was obtained. Present whenever the label is derived "
            "rather than published, so it is never read as ground truth."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Diagnostic from a configured source that could not be read.",
    )

    @classmethod
    def from_device(cls, device: Device) -> DeviceStatusOutput:
        """Build an output model from a ``Device`` ORM instance."""
        return cls(
            device_id=device.device_id,
            found=True,
            device_name=device.device_name,
            device_type=device.device_type,
            location=device.location,
            status=device.status,
            temperature=device.temperature,
            pressure=device.pressure,
            rpm=device.rpm,
            alarm_code=device.alarm_code,
            last_maintenance_time=device.last_maintenance_time,
        )

    @classmethod
    def from_snapshot(cls, snapshot: DeviceSnapshot) -> DeviceStatusOutput:
        """Build an output model from an external source snapshot.

        Only fields the source actually carries are copied. The readings are
        placed in ``measurements`` / ``digital_signals`` rather than being
        projected onto the PLC-shaped numeric fields.
        """
        return cls(
            device_id=snapshot.device_id,
            found=True,
            device_name=snapshot.device_name,
            device_type=snapshot.device_type,
            location=snapshot.location,
            status=snapshot.status,
            status_derivation=snapshot.status_derivation,
            source_timestamp=snapshot.source_timestamp,
            data_source=snapshot.data_source,
            measurements=dict(snapshot.measurements),
            digital_signals=dict(snapshot.digital_signals),
        )


def _from_external_source(
    device_id: str,
    adapters: Iterable[DeviceDataAdapter] | None,
) -> DeviceStatusOutput | None:
    """Answer from an external source, or ``None`` to fall back to SQLite."""
    adapter = resolve_adapter(device_id, adapters)
    if adapter is None:
        return None

    try:
        snapshot = adapter.get_latest_snapshot(device_id)
    except DeviceDataSourceError as exc:
        return DeviceStatusOutput(device_id=device_id, found=False, error=str(exc))

    if snapshot is None:
        return None
    return DeviceStatusOutput.from_snapshot(snapshot)


def get_device_status(
    device_id: str,
    session: Session | None = None,
    adapters: Iterable[DeviceDataAdapter] | None = None,
) -> DeviceStatusOutput:
    """Return the current status of a device.

    Args:
        device_id: Business identifier such as ``PLC-001`` or ``METRO-APU-001``.
        session: Optional SQLAlchemy session for dependency injection. When
            omitted, a transient session is created and closed automatically.
        adapters: Optional external adapters for dependency injection. When
            omitted, the adapters enabled by configuration are used.
    """
    payload = DeviceStatusInput(device_id=device_id)
    normalized = payload.device_id.strip()

    external = _from_external_source(normalized, adapters)
    if external is not None:
        return external

    owns_session = session is None
    active = session if session is not None else db_session.SessionLocal()
    try:
        device = active.scalar(select(Device).where(Device.device_id == normalized))
        if device is None:
            return DeviceStatusOutput(device_id=normalized, found=False)
        return DeviceStatusOutput.from_device(device)
    finally:
        if owns_session:
            active.close()

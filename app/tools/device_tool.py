"""Device status tool backed by the ``Device`` ORM model.

Exposes :func:`get_device_status` together with explicit Pydantic input and
output models, so the tool can be bound to a LangGraph node or (later) to an
LLM function-calling interface without leaking ORM objects across the boundary.

This tool performs no LLM call.
"""

from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import session as db_session
from app.database.models import Device


class DeviceStatusInput(BaseModel):
    """Input model for :func:`get_device_status`."""

    device_id: str = Field(
        ...,
        min_length=1,
        description="Business identifier of the device, for example PLC-001.",
    )


class DeviceStatusOutput(BaseModel):
    """Structured device status.

    ``found`` is ``False`` when the identifier is not present in the database,
    in which case every descriptive field stays ``None``.
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

    @classmethod
    def from_device(cls, device: Device) -> "DeviceStatusOutput":
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


def get_device_status(device_id: str, session: Session | None = None) -> DeviceStatusOutput:
    """Return the current status of a device.

    Args:
        device_id: Business identifier such as ``PLC-001``.
        session: Optional SQLAlchemy session for dependency injection. When
            omitted, a transient session is created and closed automatically.
    """
    payload = DeviceStatusInput(device_id=device_id)
    normalized = payload.device_id.strip()

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

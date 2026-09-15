"""SQLAlchemy ORM models for the industrial maintenance domain."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Device(Base):
    """A monitored industrial device.

    ``device_id`` is the stable business identifier. The surrogate ``id``
    primary key keeps relationships and migrations simple.
    """

    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    device_name: Mapped[str] = mapped_column(String(128), nullable=False)
    device_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    location: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure: Mapped[float | None] = mapped_column(Float, nullable=True)
    rpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    alarm_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_maintenance_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Device device_id={self.device_id!r} status={self.status!r}>"

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "location": self.location,
            "status": self.status,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "rpm": self.rpm,
            "alarm_code": self.alarm_code,
            "last_maintenance_time": (
                self.last_maintenance_time.isoformat() if self.last_maintenance_time else None
            ),
        }

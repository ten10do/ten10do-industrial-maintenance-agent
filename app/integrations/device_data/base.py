"""Data-source abstraction for device status reads.

The device tool answers from a *source*. The built-in source is the seeded
SQLite table, which holds deterministic demo records. This module defines the
boundary that lets a second, external source answer for its own identifiers
without touching the tool contract, the planner, the prompt or the database
layer.

An adapter is side-effect free with respect to the agent: it reads and returns
an immutable snapshot and never mutates agent state, and it never writes to the
data it reads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class DeviceDataSourceError(RuntimeError):
    """Raised when a configured device data source cannot be read.

    Kept distinct from "the device is unknown" on purpose. A source that is
    configured but unreadable and a device that genuinely does not exist are
    different situations, and collapsing them would let a broken deployment look
    like a missing device.
    """


@dataclass(frozen=True)
class DeviceSnapshot:
    """One immutable, source-labelled reading for a single device.

    ``status`` is never presented as ground truth on its own.
    ``status_derivation`` states where that label came from and is part of the
    contract, so a consumer can tell an operational state derived from
    documented control signals apart from a label the dataset actually
    publishes.
    """

    device_id: str
    device_name: str
    device_type: str
    location: str | None
    status: str
    status_derivation: str
    source_timestamp: datetime
    data_source: str
    measurements: Mapping[str, float | None] = field(default_factory=dict)
    digital_signals: Mapping[str, float | None] = field(default_factory=dict)


class DeviceDataAdapter(ABC):
    """Answers device status queries for a fixed set of identifiers."""

    source_id: str = "device_data"

    @property
    @abstractmethod
    def device_ids(self) -> frozenset[str]:
        """Canonical identifiers this adapter serves.

        Comparison is case-insensitive, so the stored casing is free to be the
        one the source itself uses.
        """

    @abstractmethod
    def get_latest_snapshot(self, device_id: str) -> DeviceSnapshot | None:
        """Return the most recent snapshot for ``device_id``.

        Returns ``None`` when the identifier is not one this adapter serves, so
        a caller can fall through to another source.

        Raises:
            DeviceDataSourceError: when this adapter does serve the identifier
                but the source cannot be read.
        """

    def describe(self) -> dict[str, Any]:
        """Return non-secret source metadata, useful for diagnostics."""
        return {"source_id": self.source_id}

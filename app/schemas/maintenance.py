"""Core Pydantic models for maintenance requests and responses."""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Severity(str, Enum):
    """Severity level assigned to a diagnosis."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EquipmentRef(BaseModel):
    """Reference to a piece of industrial equipment."""

    equipment_id: str = Field(..., description="Unique equipment identifier")
    name: str | None = Field(default=None, description="Human readable name")
    line: str | None = Field(default=None, description="Production line identifier")


class MaintenanceQuery(BaseModel):
    """Incoming maintenance question or fault report."""

    query: str = Field(..., min_length=1, description="Natural language request")
    equipment: EquipmentRef | None = None
    session_id: str | None = None


class Recommendation(BaseModel):
    """A single recommended maintenance action."""

    action: str = Field(..., description="Action to perform")
    rationale: str | None = Field(default=None, description="Why this action applies")
    priority: int = Field(default=3, ge=1, le=5, description="1 is highest priority")


class Diagnosis(BaseModel):
    """Structured diagnosis produced by the agent."""

    summary: str = Field(..., description="Short diagnosis summary")
    severity: Severity = Severity.INFO
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    possible_causes: list[str] = Field(default_factory=list)


class MaintenanceResponse(BaseModel):
    """Agent response returned to the caller."""

    session_id: str | None = None
    diagnosis: Diagnosis | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    answer: str = ""
    created_at: datetime = Field(default_factory=_utcnow)

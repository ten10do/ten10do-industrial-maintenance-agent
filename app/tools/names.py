"""Canonical tool names.

The planner emits ``required_tools`` and the registry stores the callables, so
both sides must agree on identifiers. Defining them once here removes the risk
of a typo silently diverging the plan from what the executor can actually run.
"""

from enum import Enum


class ToolName(str, Enum):
    """Registry identifiers for the built-in tools."""

    GET_DEVICE_STATUS = "get_device_status"
    QUERY_ALARM_CODE = "query_alarm_code"
    SEARCH_MAINTENANCE_MANUAL = "search_maintenance_manual"


TOOL_NAMES: tuple[str, ...] = tuple(member.value for member in ToolName)

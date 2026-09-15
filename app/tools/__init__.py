"""Tool package: registry, canonical names and the built-in maintenance tools."""

from app.tools.alarm_tool import (
    AlarmQueryInput,
    AlarmQueryOutput,
    load_alarm_catalog,
    query_alarm_code,
)
from app.tools.device_tool import (
    DeviceStatusInput,
    DeviceStatusOutput,
    get_device_status,
)
from app.tools.maintenance_manual_tool import (
    MANUAL_TOOL_SOURCE,
    ManualSearchInput,
    ManualSearchOutput,
    ManualSearchResult,
    search_maintenance_manual,
)
from app.tools.names import TOOL_NAMES, ToolName
from app.tools.registry import ToolRegistry, ToolSpec, registry

__all__ = [
    "MANUAL_TOOL_SOURCE",
    "TOOL_NAMES",
    "AlarmQueryInput",
    "AlarmQueryOutput",
    "DeviceStatusInput",
    "DeviceStatusOutput",
    "ManualSearchInput",
    "ManualSearchOutput",
    "ManualSearchResult",
    "ToolName",
    "ToolRegistry",
    "ToolSpec",
    "get_device_status",
    "load_alarm_catalog",
    "query_alarm_code",
    "register_default_tools",
    "registry",
    "search_maintenance_manual",
]


def register_default_tools(target: ToolRegistry | None = None) -> ToolRegistry:
    """Register the built-in tools on ``target`` (defaults to the shared one).

    Idempotent: already-registered names are skipped, so importing this package
    more than once is safe.
    """
    target_registry = target if target is not None else registry
    existing = set(target_registry.names())

    if ToolName.GET_DEVICE_STATUS.value not in existing:
        target_registry.register(
            name=ToolName.GET_DEVICE_STATUS.value,
            description="Return the current status of a device by its device_id.",
            func=get_device_status,
            tags=["device", "status", "read"],
            input_model=DeviceStatusInput,
        )

    if ToolName.QUERY_ALARM_CODE.value not in existing:
        target_registry.register(
            name=ToolName.QUERY_ALARM_CODE.value,
            description="Look up an alarm code in the local alarm catalog.",
            func=query_alarm_code,
            tags=["alarm", "read"],
            input_model=AlarmQueryInput,
        )

    if ToolName.SEARCH_MAINTENANCE_MANUAL.value not in existing:
        target_registry.register(
            name=ToolName.SEARCH_MAINTENANCE_MANUAL.value,
            description="Retrieve maintenance manual fragments from the RAG knowledge base.",
            func=search_maintenance_manual,
            tags=["manual", "rag", "read"],
            input_model=ManualSearchInput,
        )

    return target_registry


# Register on import so the shared registry is ready for the agent layer.
register_default_tools()

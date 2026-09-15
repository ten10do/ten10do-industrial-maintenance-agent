"""Tool schemas generated from the registry's own Pydantic input models.

Requirement this module exists to satisfy: there is exactly one definition of a
tool's parameters. The registry holds the callable and its input model; the
schema handed to an LLM is produced by ``model_json_schema()`` from that same
model. Adding a parameter to a tool therefore changes the model, the tool's own
validation and the LLM schema at once, and no second list can drift.

The generated schema is passed through unmodified apart from the wrapper keys
(``name``, ``description``, ``parameters``), so what a model sees is what Pydantic
actually enforces.
"""

from typing import Any

from app.tools.registry import ToolRegistry, ToolSpec, registry


def tool_parameters(spec: ToolSpec) -> dict[str, Any]:
    """Return the JSON Schema for a tool's arguments.

    A tool that declares no input model yields an object schema with no
    properties. That is not treated as "any arguments are fine": the strict
    resolver refuses such a tool outright.
    """
    if spec.input_model is None:
        return {"type": "object", "properties": {}}
    return spec.input_model.model_json_schema()


def tool_definition(spec: ToolSpec) -> dict[str, Any]:
    """Return the planner-facing definition of one tool."""
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": tool_parameters(spec),
    }


def available_tool_definitions(target: ToolRegistry | None = None) -> list[dict[str, Any]]:
    """Return a definition for every registered tool, in registration order."""
    active = target if target is not None else registry
    return [tool_definition(spec) for spec in active.list_tools()]


def available_tool_names(target: ToolRegistry | None = None) -> list[str]:
    """Return the registered tool names, which is the planner's closed vocabulary."""
    active = target if target is not None else registry
    return active.names()

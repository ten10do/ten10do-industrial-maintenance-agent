"""Strict tool-argument resolution shared by the planner and the executor.

Both layers gate calls through this module, so a plan cannot reach a tool with a
parameter the tool does not declare, or with a required parameter missing.

Strictness is deliberate. Pydantic's default ``extra="ignore"`` silently drops an
undeclared argument, which would turn a hallucinated parameter into a call that
looks valid. Here an undeclared argument is an error and a missing required
argument is an error. Nothing is defaulted on the model's behalf, because
defaulting a missing argument is exactly the guessing the executor is forbidden
from doing.

Error text is built from the validation error's field path and message only. The
offending value is never echoed.
"""

from typing import Any

from pydantic import BaseModel, ValidationError

from app.tools.registry import registry


class ToolArgumentError(ValueError):
    """Raised when arguments cannot be validated for a tool."""


def resolve_input_model(name: str) -> type[BaseModel]:
    """Return the input model registered for ``name``.

    Raises:
        ToolArgumentError: The tool is unknown, or it declares no input model, in
            which case its parameters cannot be checked at all.
    """
    try:
        spec = registry.get(name)
    except KeyError as exc:
        raise ToolArgumentError(f"unknown tool: {name}") from exc
    if spec.input_model is None:
        raise ToolArgumentError(f"tool declares no input model: {name}")
    return spec.input_model


def describe_validation_error(error: ValidationError) -> str:
    """Summarize a validation error without echoing the offending value.

    Shared with the planner so a malformed plan and malformed arguments are
    reported in the same shape.
    """
    parts: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
        parts.append(f"{location}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate ``arguments`` against the tool's own input model.

    Returns:
        The declared arguments only, coerced by the model. Arguments the caller
        did not supply stay absent, so the tool's own defaults still apply.

    Raises:
        ToolArgumentError: A tool is unknown, the payload is not an object, an
            argument is not declared by the model, or a required argument is
            missing or of the wrong type.
    """
    model = resolve_input_model(name)

    if not isinstance(arguments, dict):
        raise ToolArgumentError(f"arguments for {name} must be a JSON object")

    undeclared = sorted(set(arguments) - set(model.model_fields))
    if undeclared:
        raise ToolArgumentError(f"undeclared argument(s) for {name}: {', '.join(undeclared)}")

    try:
        validated = model.model_validate(arguments)
    except ValidationError as exc:
        raise ToolArgumentError(
            f"invalid arguments for {name}: {describe_validation_error(exc)}"
        ) from exc

    return validated.model_dump(exclude_unset=True)


def argument_names(name: str) -> list[str]:
    """Return the declared argument names for a tool, in model order."""
    return list(resolve_input_model(name).model_fields)

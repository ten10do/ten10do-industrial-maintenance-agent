"""Minimal tool registry used by the agent workflow.

Tools are plain callables decorated with metadata so they can later be bound
to LangGraph nodes or exposed to an LLM as callable functions.

``input_model`` carries the tool's existing Pydantic input model. It is what the
LLM tool schema is generated from, so the registry, the tool's own validation and
the schema handed to a model all describe the same parameters. There is no second
hand-written parameter list to drift.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass
class ToolSpec:
    """Metadata wrapper around a callable tool."""

    name: str
    description: str
    func: Callable[..., Any]
    tags: list[str] = field(default_factory=list)
    input_model: type[BaseModel] | None = None


class ToolRegistry:
    """In-memory registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        tags: list[str] | None = None,
        input_model: type[BaseModel] | None = None,
    ) -> ToolSpec:
        """Register a tool, raising if the name is already taken."""
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        spec = ToolSpec(
            name=name,
            description=description,
            func=func,
            tags=tags or [],
            input_model=input_model,
        )
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> ToolSpec:
        """Look up a tool by name."""
        return self._tools[name]

    def list_tools(self) -> list[ToolSpec]:
        """Return all registered tools."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return registered tool names."""
        return list(self._tools.keys())


# Shared application-wide registry.
registry = ToolRegistry()

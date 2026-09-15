"""Application services that sit between the agent layer and data sources.

The package exposes two families of symbols:

* the device catalog, a leaf utility that depends on nothing in the agent layer,
  imported eagerly;
* the invocation service, which depends on the compiled agent graph and is
  therefore resolved lazily through :func:`__getattr__`.

The lazy half is load-bearing. The agent layer imports
``app.services.device_catalog``, and importing any submodule executes this file
first. An eager import of ``agent_service`` here would drag the whole agent
package into that import while the agent package is still initialising itself.
The cycle does not fail deterministically: it surfaces as an ``ImportError`` on
whichever module happens to be imported first, so a module importable on its own
can fail when another module reaches it first. Deferring the attribute keeps both
directions importable in any order.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from app.services.device_catalog import (
    DeviceCatalog,
    canonicalize_device_id,
    device_vocabulary,
    load_device_catalog,
    resolve_device_alias,
)

if TYPE_CHECKING:  # pragma: no cover - seen by type checkers, not executed
    from app.services.agent_service import (
        AGENT_INVOCATION_FAILED,
        AgentInvocationError,
        AgentService,
    )

__all__ = [
    "AGENT_INVOCATION_FAILED",
    "AgentInvocationError",
    "AgentService",
    "DeviceCatalog",
    "canonicalize_device_id",
    "device_vocabulary",
    "load_device_catalog",
    "resolve_device_alias",
]

#: Symbols owned by the agent-dependent module, resolved on first access.
_LAZY_IMPORTS: dict[str, str] = {
    "AGENT_INVOCATION_FAILED": "app.services.agent_service",
    "AgentInvocationError": "app.services.agent_service",
    "AgentService": "app.services.agent_service",
}


def __getattr__(name: str) -> Any:
    """Resolve an agent-dependent symbol on first access (PEP 562).

    The resolved value is cached in the module globals so the second access is a
    plain attribute lookup.

    Raises:
        AttributeError: The name is not part of the public surface.
    """
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public names, lazily exported ones included."""
    return sorted(__all__)

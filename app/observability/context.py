"""Request correlation context.

``request_id`` is created once per invocation and must be readable by the
planner, the executor, the RAG provider call and the LLM provider call without
being threaded through every function signature. A :class:`~contextvars.ContextVar`
does that, and it keeps the graph's state schema unchanged: no node signature
grows a correlation argument, so the workflow contract stays exactly as it was.

The context is intentionally not a metric label. It is unique per request, so
labelling a metric with it would create one series per invocation. It exists for
log correlation and span attributes only.

Propagation note: FastAPI runs the synchronous agent pipeline in a worker thread,
and the context is set and read inside that same thread for the whole
invocation. A context created in one thread is not visible in another, which is
the correct behaviour here: an invocation never spans threads.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

#: Correlation fields for the invocation currently running on this thread.
_current: ContextVar[ObservationContext | None] = ContextVar(
    "app_observability_context", default=None
)


@dataclass(frozen=True)
class ObservationContext:
    """Correlation state for one invocation.

    Frozen so a nested helper cannot rewrite the identity of the request it is
    reporting on.
    """

    request_id: str

    def fields(self) -> dict[str, str]:
        """Return the correlation fields as a plain mapping for log records."""
        return {"request_id": self.request_id}


def current_context() -> ObservationContext | None:
    """Return the active context, or ``None`` when nothing is being observed.

    ``None`` is a normal answer. A tool, a provider or a planner can be exercised
    directly by a caller or a test with no request around it, and instrumentation
    must stay silent rather than invent an identifier.
    """
    return _current.get()


def current_request_id() -> str | None:
    """Return the active ``request_id``, or ``None``."""
    context = _current.get()
    return context.request_id if context is not None else None


@contextmanager
def request_context(request_id: str) -> Iterator[ObservationContext]:
    """Bind ``request_id`` for the duration of the block.

    The previous binding is restored on exit, so nesting works and a worker
    thread that serves a later request never inherits the previous request's
    identifier.
    """
    token = _current.set(ObservationContext(request_id=request_id))
    try:
        yield ObservationContext(request_id=request_id)
    finally:
        _current.reset(token)

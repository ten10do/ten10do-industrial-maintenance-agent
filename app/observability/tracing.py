"""Optional OpenTelemetry tracing.

Tracing is off unless it is asked for. The three properties that matter:

**No import when disabled.** Neither ``opentelemetry-api`` nor
``opentelemetry-sdk`` is imported at module import time. A deployment that does
not trace therefore does not need the dependency installed, and one that has it
installed but disabled never constructs a tracer provider, never builds an
exporter and never opens a socket. ``tests/test_observability.py`` asserts this by
failing the test if the exporter constructor is reached.

**No implied collector.** The OTLP endpoint comes from configuration and has no
default. Pointing a default at ``localhost:4317`` would turn an optional feature
into a runtime dependency on a process the operator may not have started, and the
SDK's retry behaviour would turn that into latency in the request path.

**Attributes are allow-listed.** :func:`span_attributes` drops anything not in
:data:`SPAN_ATTRIBUTE_NAMES` and redacts through the same policy the JSON logger
uses. ``record_exception`` is available but is only ever reached through
:func:`record_span_error`, which records the class name and a redacted message
rather than the exception object, so a provider error's URL and headers cannot
become span attributes.

The span hierarchy is the execution hierarchy: ``agent.invoke`` contains
``planner.plan``, each ``tool.execute`` with the tool's own name as an attribute,
``rag.retrieve``, ``llm.request`` and ``synthesize``. No span is invented for a
step that did not run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

from app.observability.logging import sanitize

logger = logging.getLogger(__name__)

#: Root span name. Every other span is a descendant of it.
SPAN_AGENT_INVOKE: Final = "agent.invoke"
SPAN_PLANNER_PLAN: Final = "planner.plan"
SPAN_TOOL_EXECUTE: Final = "tool.execute"
SPAN_RAG_RETRIEVE: Final = "rag.retrieve"
SPAN_LLM_REQUEST: Final = "llm.request"
SPAN_SYNTHESIZE: Final = "synthesize"

#: Attributes a span may carry. This is the whole allow-list.
SPAN_ATTRIBUTE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "planner.type",
        "intent",
        "tool.name",
        "provider",
        "status",
        "evidence.count",
    }
)


def span_attributes(**attributes: Any) -> dict[str, Any]:
    """Return the subset of ``attributes`` that is allowed onto a span.

    Values pass through the logger's redaction so a provider message that reached
    an attribute by mistake is still stripped of a bearer token or a key.
    """
    return {
        name: sanitize(value)
        for name, value in attributes.items()
        if name in SPAN_ATTRIBUTE_NAMES and value is not None
    }


class _NullSpan:
    """Stand-in for a span when tracing is disabled.

    Every method is a no-op, so an instrumented call site needs no branch: it
    always calls the same methods whether or not tracing is on.
    """

    def set_attribute(self, name: str, value: Any) -> None:
        """Ignore an attribute."""

    def record_error(self, error_type: str, message: str | None = None) -> None:
        """Ignore an error."""

    def set_status(self, status: str) -> None:
        """Ignore a status."""


_null_span = _NullSpan()

#: Set by :func:`configure_tracing`. ``None`` means tracing is off.
_tracer: Any | None = None


def is_enabled() -> bool:
    """Return ``True`` when a tracer is installed."""
    return _tracer is not None


def configure_tracing(
    *,
    enabled: bool,
    service_name: str,
    otlp_endpoint: str = "",
) -> bool:
    """Install a tracer when tracing is enabled and the SDK is importable.

    Args:
        enabled: Value of ``OTEL_ENABLED``. ``False`` leaves tracing off and
            imports nothing.
        service_name: Value of ``OTEL_SERVICE_NAME``.
        otlp_endpoint: Value of ``OTEL_EXPORTER_OTLP_ENDPOINT``. Empty means no
            exporter is built, which keeps spans in the process. An endpoint is
            never invented.

    Returns:
        ``True`` when a tracer was installed.

    A missing SDK or a bad endpoint logs a warning and leaves tracing off. An
    observability feature must not be able to stop the service from starting.
    """
    global _tracer

    if not enabled:
        _tracer = None
        return False

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # pragma: no cover - depends on the installed extras
        logger.warning("otel_sdk_missing tracing_disabled=true")
        _tracer = None
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

    if otlp_endpoint.strip():
        try:  # pragma: no cover - requires a collector to be meaningful
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
        except ImportError:
            logger.warning("otel_otlp_exporter_missing endpoint_ignored=true")

    _tracer = provider.get_tracer(service_name)
    return True


def reset_tracing() -> None:
    """Drop the installed tracer. For tests and for a process that reconfigures."""
    global _tracer
    _tracer = None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a span, or a null span when tracing is off.

    Yields:
        An object exposing ``set_attribute``, ``record_error`` and
        ``set_status``. The two implementations behave identically from the
        caller's point of view, which is what keeps the call sites free of a
        tracing-enabled branch.
    """
    if _tracer is None:
        yield _null_span
        return

    with _tracer.start_as_current_span(name) as active:
        for key, value in span_attributes(**attributes).items():
            active.set_attribute(key, value)
        yield _OtelSpanAdapter(active)


class _OtelSpanAdapter:
    """Adapts a real span to the small surface the call sites use."""

    __slots__ = ("_span",)

    def __init__(self, active: Any) -> None:
        self._span = active

    def set_attribute(self, name: str, value: Any) -> None:
        """Set an attribute when it is on the allow-list."""
        allowed = span_attributes(**{name: value})
        for key, sanitized in allowed.items():
            self._span.set_attribute(key, sanitized)

    def record_error(self, error_type: str, message: str | None = None) -> None:
        """Record an error as a type name plus a redacted message.

        The exception object is not passed to ``record_exception``. That call
        serializes the exception's arguments, and an LLM provider error carries a
        URL and, on some transports, the headers, which carry the credential.
        """
        self._span.set_attribute("error.type", sanitize(error_type))
        if message:
            self._span.set_attribute("error.message", sanitize(message))

    def set_status(self, status: str) -> None:
        """Set the span status from a bounded vocabulary value."""
        self._span.set_attribute("status", sanitize(status))

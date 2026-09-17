"""Production observability for the maintenance agent.

Three capabilities, each of which can be switched off without affecting the
agent's behaviour:

* **structured logging**, ``text`` by default and ``json`` on request;
* **Prometheus metrics**, exposed on ``GET /metrics`` when enabled;
* **optional OpenTelemetry tracing**, off unless configured.

The package is deliberately decoupled from the domain. Nothing in
``app.agent``, ``app.tools`` or ``app.integrations`` imports a metrics library or
an OpenTelemetry symbol: those layers call the helpers in
:mod:`app.observability.instrumentation`, which own the mapping from an
application event onto a metric, a span and a log line.

What observability must never do, enforced here rather than by convention:

* change an agent decision, a tool schema, an answer or a benchmark result;
* put the query, the prompt, the completion, a document body, a header or a
  credential into a log, a metric label or a span attribute;
* use a high-cardinality value, ``request_id`` above all, as a metric label.

Usage:

    from app.observability import configure_observability, metrics_enabled

    configure_observability(settings)
"""

from __future__ import annotations

from app.observability.context import (
    ObservationContext,
    current_context,
    current_request_id,
    request_context,
)
from app.observability.instrumentation import metrics_enabled
from app.observability.metrics import (
    CONTENT_TYPE_LATEST,
    FORBIDDEN_LABEL_NAMES,
    METRIC_SPECS,
    Metrics,
    get_metrics,
    reset_metrics,
)
from app.observability.tracing import (
    SPAN_AGENT_INVOKE,
    SPAN_ATTRIBUTE_NAMES,
    configure_tracing,
    reset_tracing,
    span_attributes,
)
from app.observability.tracing import (
    is_enabled as tracing_enabled,
)

__all__ = [
    "CONTENT_TYPE_LATEST",
    "FORBIDDEN_LABEL_NAMES",
    "METRIC_SPECS",
    "SPAN_AGENT_INVOKE",
    "SPAN_ATTRIBUTE_NAMES",
    "Metrics",
    "ObservationContext",
    "configure_observability",
    "configure_tracing",
    "current_context",
    "current_request_id",
    "get_metrics",
    "metrics_enabled",
    "request_context",
    "reset_metrics",
    "reset_tracing",
    "span_attributes",
    "tracing_enabled",
]


def configure_observability(settings: object) -> None:
    """Apply the observability configuration to the process.

    Called once from the application entry point, next to the logging
    configuration and for the same reason: a library configures nothing, an
    application configures once.

    Args:
        settings: A :class:`~app.config.Settings` instance. Typed loosely to keep
            this module from importing the settings class, which would make the
            observability package depend on the configuration layer.

    Tracing is configured here and nowhere else. Metrics need no configuration
    step: the collectors are created lazily on first use, and whether they are
    incremented is decided per event from ``METRICS_ENABLED``.
    """
    configure_tracing(
        enabled=bool(getattr(settings, "otel_enabled", False)),
        service_name=str(getattr(settings, "otel_service_name", "industrial-maintenance-agent")),
        otlp_endpoint=str(getattr(settings, "otel_exporter_otlp_endpoint", "")),
    )

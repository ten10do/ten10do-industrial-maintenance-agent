"""Prometheus metrics for the agent pipeline.

Design decisions worth stating, because each one has an alternative:

**A private registry.** The metrics are registered on a
:class:`~prometheus_client.CollectorRegistry` owned by this module rather than
the library's process-global default. A process that imports the application
twice, or a test that imports it after another test, would otherwise collect
duplicate-registration errors or see another suite's samples. Owning the registry
also makes :func:`reset_metrics` possible, which is what lets a test assert a
delta without restarting the interpreter.

**Labels are drawn from frozen vocabularies.**
:mod:`app.observability.labels` allow-lists every label value and maps anything
else onto ``other``. No label is ever taken verbatim from a request.

**One table declares every metric.** :data:`METRIC_SPECS` is the single source of
truth for names, types, labels and histogram buckets. The collector objects are
built from it and the cardinality test reads it, so a metric cannot acquire a
label that the test does not inspect.

**Latency is recorded in seconds.** Prometheus convention, and the metric name
carries the unit. The application reports milliseconds elsewhere; conversion
happens at exactly one place, :func:`_seconds`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from app.observability.labels import (
    INTENT_LABELS,
    PLANNER_LABELS,
    PROVIDER_LABELS,
    REASON_LABELS,
    STATUS_LABELS,
    TOKEN_TYPE_LABELS,
    TOOL_LABELS,
)

#: Content type required by the Prometheus exposition format.
CONTENT_TYPE_LATEST: Final = "text/plain; version=0.0.4; charset=utf-8"

MetricKind = Literal["counter", "histogram"]

#: Request latency spans a health probe up to a cold in-process RAG retrieval,
#: which re-reads its index on first use and is measured in seconds.
_REQUEST_BUCKETS: Final = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)
#: Planning is sub-millisecond for the rule planner and seconds for an LLM call,
#: so the low buckets carry real information here.
_PLANNER_BUCKETS: Final = (
    0.0001,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    2.5,
    5.0,
    30.0,
)
#: Tools are in-process lookups except the manual search, which reaches RAG.
_TOOL_BUCKETS: Final = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_RAG_BUCKETS: Final = (0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_LLM_BUCKETS: Final = (0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


@dataclass(frozen=True)
class MetricSpec:
    """Declaration of one metric.

    ``labelnames`` is asserted against the matching vocabulary by the test suite,
    so a label added here without a vocabulary, or a vocabulary that drifts from
    its producer, fails a build rather than a dashboard.
    """

    name: str
    kind: MetricKind
    documentation: str
    labelnames: tuple[str, ...] = ()
    buckets: tuple[float, ...] = ()
    vocabulary: tuple[frozenset[str], ...] = field(default=())


#: The single source of truth for the metric surface. Every entry is built into a
#: collector below and inspected by ``tests/test_observability.py``.
METRIC_SPECS: Final[tuple[MetricSpec, ...]] = (
    MetricSpec(
        name="industrial_agent_requests_total",
        kind="counter",
        documentation="Agent invocations by planner, intent and outcome.",
        labelnames=("planner", "intent", "status"),
        vocabulary=(PLANNER_LABELS, INTENT_LABELS, STATUS_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_request_duration_seconds",
        kind="histogram",
        documentation="Wall-clock time of one agent invocation, HTTP entry to response.",
        labelnames=("planner", "status"),
        buckets=_REQUEST_BUCKETS,
        vocabulary=(PLANNER_LABELS, STATUS_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_planner_duration_seconds",
        kind="histogram",
        documentation="Wall-clock time of the planning step alone.",
        labelnames=("planner", "status"),
        buckets=_PLANNER_BUCKETS,
        vocabulary=(PLANNER_LABELS, STATUS_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_planner_failures_total",
        kind="counter",
        documentation="Planning failures by planner and closed reason enumeration.",
        labelnames=("planner", "reason"),
        vocabulary=(PLANNER_LABELS, REASON_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_tool_calls_total",
        kind="counter",
        documentation="Executed tool calls by registry tool name and outcome.",
        labelnames=("tool", "status"),
        vocabulary=(TOOL_LABELS, STATUS_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_tool_duration_seconds",
        kind="histogram",
        documentation="Wall-clock time of one tool execution.",
        labelnames=("tool", "status"),
        buckets=_TOOL_BUCKETS,
        vocabulary=(TOOL_LABELS, STATUS_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_rag_requests_total",
        kind="counter",
        documentation="Retrieval calls by provider and outcome.",
        labelnames=("provider", "status"),
        vocabulary=(PROVIDER_LABELS, STATUS_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_rag_duration_seconds",
        kind="histogram",
        documentation="Wall-clock time of one retrieval call.",
        labelnames=("provider", "status"),
        buckets=_RAG_BUCKETS,
        vocabulary=(PROVIDER_LABELS, STATUS_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_llm_requests_total",
        kind="counter",
        documentation="LLM provider calls by provider and outcome.",
        labelnames=("provider", "status"),
        vocabulary=(PROVIDER_LABELS, STATUS_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_llm_duration_seconds",
        kind="histogram",
        documentation="Wall-clock time of one LLM provider call.",
        labelnames=("provider", "status"),
        buckets=_LLM_BUCKETS,
        vocabulary=(PROVIDER_LABELS, STATUS_LABELS),
    ),
    MetricSpec(
        name="industrial_agent_llm_tokens_total",
        kind="counter",
        documentation=(
            "Tokens reported by a provider, by direction. Only incremented when the "
            "provider actually reports usage; never estimated."
        ),
        labelnames=("provider", "type"),
        vocabulary=(PROVIDER_LABELS, TOKEN_TYPE_LABELS),
    ),
)

#: Label names that must never appear on a metric. Kept here, next to the specs,
#: so the prohibition and the declaration are read together.
FORBIDDEN_LABEL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "request_id",
        "device_id",
        "equipment_id",
        "query",
        "prompt",
        "answer",
        "document",
        "document_path",
        "document_name",
        "page",
        "chunk_id",
        "exception",
        "exception_message",
        "error_message",
        "url",
        "session_id",
        "model",
    }
)


class Metrics:
    """The metric collectors, bound to one registry.

    Exposed as an object rather than as module-level collectors so a test can
    build a private instance, exercise the pipeline and read the result without
    touching, or being touched by, another test.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self.counters: dict[str, Counter] = {}
        self.histograms: dict[str, Histogram] = {}

        for spec in METRIC_SPECS:
            if spec.kind == "counter":
                self.counters[spec.name] = Counter(
                    spec.name,
                    spec.documentation,
                    labelnames=list(spec.labelnames),
                    registry=self.registry,
                )
            else:
                self.histograms[spec.name] = Histogram(
                    spec.name,
                    spec.documentation,
                    labelnames=list(spec.labelnames),
                    buckets=list(spec.buckets),
                    registry=self.registry,
                )

    def counter(self, name: str) -> Counter:
        """Return a counter by name. Raises ``KeyError`` for an unknown name."""
        return self.counters[name]

    def histogram(self, name: str) -> Histogram:
        """Return a histogram by name. Raises ``KeyError`` for an unknown name."""
        return self.histograms[name]

    def observe(
        self,
        name: str,
        seconds: float,
        labels: dict[str, str],
    ) -> None:
        """Record one observation on a histogram."""
        self.histograms[name].labels(**labels).observe(seconds)

    def count(
        self,
        name: str,
        labels: dict[str, str],
        amount: float = 1.0,
    ) -> None:
        """Increment a counter, or add ``amount`` to it."""
        self.counters[name].labels(**labels).inc(amount)

    def render(self) -> bytes:
        """Render the registry in the Prometheus exposition format."""
        return generate_latest(self.registry)


_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    """Return the process-wide metrics, creating them on first use."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


def reset_metrics() -> Metrics:
    """Replace the process-wide metrics with empty ones.

    For tests and for a process that must drop its counters. Production code has
    no reason to call it: a counter reset is indistinguishable from a counter
    restart to the scraper, and a Prometheus counter is meant to be monotonic for
    the life of the process.
    """
    global _metrics
    _metrics = Metrics()
    return _metrics

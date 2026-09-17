"""Instrumentation helpers: the only observability surface business code touches.

The planner, the executor, the tools and the integrations do not import
``prometheus_client`` or ``opentelemetry``. They call one function or open one
context manager from here, and this module owns the mapping from an application
event onto a metric, a span and a log line. The benefits are concrete:

* a metric name or label changes in one file;
* a call site cannot invent a label, because label values pass through
  :func:`app.observability.labels.bounded`;
* when metrics are disabled the whole recording step is skipped, and the cost of
  observability drops to the log calls the service already made.

Status vocabulary, applied identically to every metric, so one rule covers every
dashboard. The four values are frozen in
:data:`app.observability.labels.STATUS_LABELS`:

``success``
    The operation completed.
``not_found``
    The operation completed and legitimately found nothing. A missing device, an
    absent alarm code and an empty retrieval are normal answers, and reporting
    them as failures would make a healthy service look broken.
``unavailable``
    A backing source could not be read or reached: a RAG provider error, an
    external device source that raised, an LLM endpoint that failed or timed out.
``error``
    The operation raised, or produced a result that could not be used.

Timing: every duration is measured around the operation itself, at the boundary
the application already uses. Nothing is estimated, and no duration is reused
across two operations.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Final

from app.config import get_settings
from app.observability import logging as event_logging
from app.observability import tracing
from app.observability.context import current_context
from app.observability.labels import (
    INTENT_LABELS,
    PLANNER_LABELS,
    PROVIDER_LABELS,
    REASON_LABELS,
    STATUS_LABELS,
    TOKEN_TYPE_LABELS,
    TOOL_LABELS,
    bounded,
)
from app.observability.metrics import get_metrics

logger = logging.getLogger("app.observability")

STATUS_SUCCESS: Final = "success"
STATUS_NOT_FOUND: Final = "not_found"
STATUS_UNAVAILABLE: Final = "unavailable"
STATUS_ERROR: Final = "error"

#: Metric names, referenced by name so a rename is a one-line change.
M_REQUESTS: Final = "industrial_agent_requests_total"
M_REQUEST_DURATION: Final = "industrial_agent_request_duration_seconds"
M_PLANNER_DURATION: Final = "industrial_agent_planner_duration_seconds"
M_PLANNER_FAILURES: Final = "industrial_agent_planner_failures_total"
M_TOOL_CALLS: Final = "industrial_agent_tool_calls_total"
M_TOOL_DURATION: Final = "industrial_agent_tool_duration_seconds"
M_RAG_REQUESTS: Final = "industrial_agent_rag_requests_total"
M_RAG_DURATION: Final = "industrial_agent_rag_duration_seconds"
M_LLM_REQUESTS: Final = "industrial_agent_llm_requests_total"
M_LLM_DURATION: Final = "industrial_agent_llm_duration_seconds"
M_LLM_TOKENS: Final = "industrial_agent_llm_tokens_total"


def metrics_enabled() -> bool:
    """Return whether metric recording is switched on."""
    return bool(get_settings().metrics_enabled)


def elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since ``started``, rounded to microseconds.

    Shared by every instrumented boundary so durations are measured and rounded
    identically. Each boundary passes its own start value, which is what keeps two
    operations from reporting the same span of wall-clock time.
    """
    return round((perf_counter() - started) * 1000, 3)


def _correlation() -> dict[str, Any]:
    """Return the correlation fields for the active request, if any."""
    context = current_context()
    return context.fields() if context is not None else {}


def _seconds(duration_ms: float) -> float:
    """Convert milliseconds to seconds for a Prometheus histogram.

    Prometheus convention is seconds, and the application measures milliseconds.
    The conversion happens here and nowhere else, so a duration can never be
    recorded twice in two units.
    """
    return duration_ms / 1000.0


def _record(
    *,
    level: int,
    event: str,
    message: str,
    counter: tuple[str, dict[str, str]] | None = None,
    histogram: tuple[str, dict[str, str], float] | None = None,
    **fields: Any,
) -> None:
    """Emit one event: log always, metrics only when enabled.

    The log call is unconditional because the log contract predates this module
    and callers, tests and operators depend on it. Metric recording is gated so
    that disabling metrics removes the series work without removing the audit
    trail.
    """
    event_logging.log_event(
        logger,
        level,
        event,
        message,
        duration_ms=round(fields.pop("duration_ms", 0.0), 3),
        **fields,
    )

    if not metrics_enabled():
        return

    metrics = get_metrics()
    if counter is not None:
        metrics.count(counter[0], counter[1])
    if histogram is not None:
        metrics.observe(histogram[0], _seconds(histogram[2]), histogram[1])


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


def request_started(*, planner_mode: str, query_chars: int) -> None:
    """Record the start of an invocation.

    ``query_chars`` is a length, not the query. It answers "was this a one-word
    lookup or a paragraph" without putting user text in a log or a metric.
    """
    event_logging.log_event(
        logger,
        logging.INFO,
        event_logging.EVENT_REQUEST_STARTED,
        f"agent_request_started planner_mode={planner_mode} query_chars={query_chars}",
        planner=bounded(planner_mode, PLANNER_LABELS),
        count=query_chars,
        **_correlation(),
    )


def request_completed(
    *,
    planner: str | None,
    intent: str | None,
    status: str,
    duration_ms: float,
) -> None:
    """Record a finished invocation, successful or not."""
    planner_label = bounded(planner, PLANNER_LABELS)
    intent_label = bounded(intent, INTENT_LABELS)
    status_label = bounded(status, STATUS_LABELS)

    _record(
        level=logging.INFO if status_label == STATUS_SUCCESS else logging.ERROR,
        event=event_logging.EVENT_REQUEST_COMPLETED,
        message=(
            f"agent_request_completed status={status_label} planner={planner_label} "
            f"intent={intent_label} duration_ms={duration_ms}"
        ),
        counter=(
            M_REQUESTS,
            {"planner": planner_label, "intent": intent_label, "status": status_label},
        ),
        histogram=(
            M_REQUEST_DURATION,
            {"planner": planner_label, "status": status_label},
            duration_ms,
        ),
        duration_ms=duration_ms,
        planner=planner_label,
        intent=intent_label,
        status=status_label,
        error_type=None if status_label == STATUS_SUCCESS else status_label,
        **_correlation(),
    )


@contextmanager
def request_span(*, planner_mode: str) -> Iterator[Any]:
    """Open the root span for an invocation."""
    with tracing.span(tracing.SPAN_AGENT_INVOKE, **{"planner.type": planner_mode}) as active:
        yield active


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


def planner_started(*, planner: str) -> None:
    """Record that planning began."""
    event_logging.log_event(
        logger,
        logging.INFO,
        event_logging.EVENT_PLANNER_STARTED,
        f"planner_started planner={planner}",
        planner=bounded(planner, PLANNER_LABELS),
        **_correlation(),
    )


def planner_completed(*, planner: str, duration_ms: float, tools: int) -> None:
    """Record a plan that was produced."""
    planner_label = bounded(planner, PLANNER_LABELS)
    _record(
        level=logging.INFO,
        event=event_logging.EVENT_PLANNER_COMPLETED,
        message=(
            f"planner_completed planner={planner_label} tools={tools} duration_ms={duration_ms}"
        ),
        histogram=(
            M_PLANNER_DURATION,
            {"planner": planner_label, "status": STATUS_SUCCESS},
            duration_ms,
        ),
        duration_ms=duration_ms,
        planner=planner_label,
        status=STATUS_SUCCESS,
        count=tools,
        **_correlation(),
    )


def planner_failed(*, planner: str, reason: str, duration_ms: float) -> None:
    """Record a planning failure.

    ``reason`` must be a member of :data:`~app.observability.labels.REASON_LABELS`.
    An exception message is never a reason: it can carry a URL, a header or a key
    fragment, and a metric label is published to every scraper.
    """
    planner_label = bounded(planner, PLANNER_LABELS)
    reason_label = bounded(reason, REASON_LABELS)

    _record(
        level=logging.ERROR,
        event=event_logging.EVENT_PLANNER_FAILED,
        message=(
            f"planner_failed planner={planner_label} reason={reason_label} "
            f"duration_ms={duration_ms}"
        ),
        counter=(M_PLANNER_FAILURES, {"planner": planner_label, "reason": reason_label}),
        histogram=(
            M_PLANNER_DURATION,
            {"planner": planner_label, "status": STATUS_ERROR},
            duration_ms,
        ),
        duration_ms=duration_ms,
        planner=planner_label,
        status=STATUS_ERROR,
        reason=reason_label,
        error_type=reason_label,
        **_correlation(),
    )


@contextmanager
def planner_span(*, planner: str) -> Iterator[Any]:
    """Open the planning span."""
    with tracing.span(tracing.SPAN_PLANNER_PLAN, **{"planner.type": planner}) as active:
        yield active


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def tool_status_from_payload(payload: Any) -> str:
    """Derive a tool's outcome from the payload the tool itself returned.

    This reads the tool's own contract rather than guessing. A tool that reports a
    non-empty ``error`` could not reach its backing source, which is
    ``unavailable``; ``found`` being ``False`` is the normal "not there" answer,
    which is ``not_found``. Neither is an error, and neither raises.
    """
    if not isinstance(payload, dict):
        return STATUS_SUCCESS
    if payload.get("error"):
        return STATUS_UNAVAILABLE
    if payload.get("found") is False:
        return STATUS_NOT_FOUND
    return STATUS_SUCCESS


def tool_started(*, tool: str) -> None:
    """Record that a tool call began."""
    event_logging.log_event(
        logger,
        logging.INFO,
        event_logging.EVENT_TOOL_STARTED,
        f"tool_started tool={tool}",
        tool=bounded(tool, TOOL_LABELS),
        **_correlation(),
    )


def tool_completed(*, tool: str, status: str, duration_ms: float) -> None:
    """Record a finished tool call."""
    tool_label = bounded(tool, TOOL_LABELS)
    status_label = bounded(status, STATUS_LABELS)
    _record(
        level=logging.INFO,
        event=event_logging.EVENT_TOOL_COMPLETED,
        message=f"tool_completed tool={tool_label} status={status_label} duration_ms={duration_ms}",
        counter=(M_TOOL_CALLS, {"tool": tool_label, "status": status_label}),
        histogram=(M_TOOL_DURATION, {"tool": tool_label, "status": status_label}, duration_ms),
        duration_ms=duration_ms,
        tool=tool_label,
        status=status_label,
        **_correlation(),
    )


def tool_failed(*, tool: str, error_type: str, duration_ms: float) -> None:
    """Record a tool that raised.

    Only the exception's class name travels. A tool exception can embed a
    filesystem path or a source diagnostic.
    """
    tool_label = bounded(tool, TOOL_LABELS)
    _record(
        level=logging.ERROR,
        event=event_logging.EVENT_TOOL_FAILED,
        message=(
            f"tool_failed tool={tool_label} error_type={error_type} duration_ms={duration_ms}"
        ),
        counter=(M_TOOL_CALLS, {"tool": tool_label, "status": STATUS_ERROR}),
        histogram=(M_TOOL_DURATION, {"tool": tool_label, "status": STATUS_ERROR}, duration_ms),
        duration_ms=duration_ms,
        tool=tool_label,
        status=STATUS_ERROR,
        error_type=error_type,
        **_correlation(),
    )


@contextmanager
def tool_span(*, tool: str) -> Iterator[Any]:
    """Open the execution span for one tool."""
    with tracing.span(tracing.SPAN_TOOL_EXECUTE, **{"tool.name": tool}) as active:
        yield active


# --------------------------------------------------------------------------- #
# RAG
# --------------------------------------------------------------------------- #


def rag_started(*, provider: str, top_k: int) -> None:
    """Record that a retrieval began. No query text is passed."""
    event_logging.log_event(
        logger,
        logging.INFO,
        event_logging.EVENT_RAG_STARTED,
        f"rag_started provider={provider} top_k={top_k}",
        provider=bounded(provider, PROVIDER_LABELS),
        count=top_k,
        **_correlation(),
    )


def rag_completed(*, provider: str, status: str, hits: int, duration_ms: float) -> None:
    """Record a retrieval that returned, with or without hits."""
    provider_label = bounded(provider, PROVIDER_LABELS)
    status_label = bounded(status, STATUS_LABELS)
    _record(
        level=logging.INFO,
        event=event_logging.EVENT_RAG_COMPLETED,
        message=(
            f"rag_completed provider={provider_label} status={status_label} "
            f"hits={hits} duration_ms={duration_ms}"
        ),
        counter=(M_RAG_REQUESTS, {"provider": provider_label, "status": status_label}),
        histogram=(
            M_RAG_DURATION,
            {"provider": provider_label, "status": status_label},
            duration_ms,
        ),
        duration_ms=duration_ms,
        provider=provider_label,
        status=status_label,
        count=hits,
        **_correlation(),
    )


def rag_unavailable(*, provider: str, duration_ms: float) -> None:
    """Record a retrieval that could not run.

    Deliberately separate from an empty result: a broken deployment and a manual
    with nothing on the subject are different situations and must stay
    distinguishable in both the log and the metric.
    """
    provider_label = bounded(provider, PROVIDER_LABELS)
    _record(
        level=logging.WARNING,
        event=event_logging.EVENT_RAG_UNAVAILABLE,
        message=f"rag_unavailable provider={provider_label} duration_ms={duration_ms}",
        counter=(M_RAG_REQUESTS, {"provider": provider_label, "status": STATUS_UNAVAILABLE}),
        histogram=(
            M_RAG_DURATION,
            {"provider": provider_label, "status": STATUS_UNAVAILABLE},
            duration_ms,
        ),
        duration_ms=duration_ms,
        provider=provider_label,
        status=STATUS_UNAVAILABLE,
        error_type=STATUS_UNAVAILABLE,
        **_correlation(),
    )


@contextmanager
def rag_span(*, provider: str) -> Iterator[Any]:
    """Open the retrieval span."""
    with tracing.span(tracing.SPAN_RAG_RETRIEVE, **{"provider": provider}) as active:
        yield active


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #


def llm_started(*, provider: str) -> None:
    """Record that a provider call began."""
    event_logging.log_event(
        logger,
        logging.INFO,
        event_logging.EVENT_LLM_STARTED,
        f"llm_started provider={provider}",
        provider=bounded(provider, PROVIDER_LABELS),
        **_correlation(),
    )


def llm_completed(
    *,
    provider: str,
    duration_ms: float,
    finish_reason: str | None = None,
) -> None:
    """Record a completed provider call."""
    provider_label = bounded(provider, PROVIDER_LABELS)
    _record(
        level=logging.INFO,
        event=event_logging.EVENT_LLM_COMPLETED,
        message=(
            f"llm_completed provider={provider_label} duration_ms={duration_ms} "
            f"finish_reason={finish_reason or '-'}"
        ),
        counter=(M_LLM_REQUESTS, {"provider": provider_label, "status": STATUS_SUCCESS}),
        histogram=(
            M_LLM_DURATION,
            {"provider": provider_label, "status": STATUS_SUCCESS},
            duration_ms,
        ),
        duration_ms=duration_ms,
        provider=provider_label,
        status=STATUS_SUCCESS,
        **_correlation(),
    )


def llm_failed(*, provider: str, duration_ms: float, status: str = STATUS_UNAVAILABLE) -> None:
    """Record a provider call that did not produce a completion.

    ``status`` defaults to ``unavailable``, which is what an unreachable endpoint,
    a non-2xx response or a deadline breach means. The prompt and the response
    body are never recorded, and the error text is not accepted as an argument at
    all, so it cannot leak by accident.
    """
    provider_label = bounded(provider, PROVIDER_LABELS)
    status_label = bounded(status, STATUS_LABELS)
    _record(
        level=logging.WARNING,
        event=event_logging.EVENT_LLM_FAILED,
        message=(
            f"llm_failed provider={provider_label} status={status_label} "
            f"duration_ms={duration_ms}"
        ),
        counter=(M_LLM_REQUESTS, {"provider": provider_label, "status": status_label}),
        histogram=(
            M_LLM_DURATION,
            {"provider": provider_label, "status": status_label},
            duration_ms,
        ),
        duration_ms=duration_ms,
        provider=provider_label,
        status=status_label,
        error_type=status_label,
        **_correlation(),
    )


def llm_tokens(*, provider: str, prompt_tokens: int | None, completion_tokens: int | None) -> None:
    """Record provider-reported token usage.

    A provider that does not report usage contributes nothing. Estimating tokens
    from the text length would produce a number that looks measured and is not,
    so an absent value is recorded as absent.
    """
    if not metrics_enabled():
        return
    provider_label = bounded(provider, PROVIDER_LABELS)
    metrics = get_metrics()
    for token_type, value in (("prompt", prompt_tokens), ("completion", completion_tokens)):
        if value is None or value < 0:
            continue
        metrics.count(
            M_LLM_TOKENS,
            {"provider": provider_label, "type": bounded(token_type, TOKEN_TYPE_LABELS)},
            float(value),
        )


@contextmanager
def llm_span(*, provider: str) -> Iterator[Any]:
    """Open the provider-call span."""
    with tracing.span(tracing.SPAN_LLM_REQUEST, **{"provider": provider}) as active:
        yield active


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #


@contextmanager
def synthesis_span(*, evidence_count: int) -> Iterator[Any]:
    """Open the synthesis span, carrying only the evidence count."""
    with tracing.span(tracing.SPAN_SYNTHESIZE, **{"evidence.count": evidence_count}) as active:
        yield active


__all__ = [
    "M_LLM_DURATION",
    "M_LLM_REQUESTS",
    "M_LLM_TOKENS",
    "M_PLANNER_DURATION",
    "M_PLANNER_FAILURES",
    "M_RAG_DURATION",
    "M_RAG_REQUESTS",
    "M_REQUEST_DURATION",
    "M_REQUESTS",
    "M_TOOL_CALLS",
    "M_TOOL_DURATION",
    "STATUS_ERROR",
    "STATUS_NOT_FOUND",
    "STATUS_SUCCESS",
    "STATUS_UNAVAILABLE",
    "elapsed_ms",
    "llm_completed",
    "llm_failed",
    "llm_span",
    "llm_started",
    "llm_tokens",
    "metrics_enabled",
    "planner_completed",
    "planner_failed",
    "planner_span",
    "planner_started",
    "rag_completed",
    "rag_span",
    "rag_started",
    "rag_unavailable",
    "request_completed",
    "request_span",
    "request_started",
    "synthesis_span",
    "tool_completed",
    "tool_failed",
    "tool_span",
    "tool_started",
    "tool_status_from_payload",
]

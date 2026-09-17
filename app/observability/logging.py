"""Structured event logging with an explicit redaction boundary.

Two output shapes are supported and both are produced from the same call site:

* ``text`` keeps the V0.4 through V0.8 human-readable single line, so existing
  log-based tooling and the pinned log-contract tests keep working unchanged;
* ``json`` emits one object per line with a fixed field set for a log pipeline.

The design rule is that a field is emitted only when it is a member of
:data:`FIELD_ORDER`, and it is projected through :func:`_sanitize` first. There
is no path that dumps an arbitrary mapping into the log, and ``extra`` keys other
than the one declared name are ignored. That is what stops a future refactor from
accidentally publishing a provider payload.

The list of things that must never be logged is a policy, not a filter, and it is
enforced at the source: no call site in this application passes the query text,
the prompt, the completion, a document body, a header or a credential to an event.
:func:`sanitize` is the second line of defence for a value that arrives from
elsewhere, such as a provider error message.

Error handling deserves one note. ``error_type`` carries a class name or a stable
application code, never ``str(exc)``. A provider exception message can embed the
request URL, and for some transports the request headers, which carry the
credential.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Final

#: Field order for a JSON event. Fixed so two events of the same type serialize
#: identically and a log pipeline can rely on the shape.
FIELD_ORDER: Final[tuple[str, ...]] = (
    "timestamp",
    "level",
    "event",
    "request_id",
    "planner",
    "intent",
    "tool",
    "provider",
    "status",
    "duration_ms",
    "error_type",
    "count",
    "reason",
    "token_type",
)

#: Name of the ``extra`` key carrying structured fields. A single declared name
#: means nothing else in ``record.__dict__`` is ever serialized.
EXTRA_KEY: Final = "event_fields"

#: Event names. Every event this application emits appears here, so the vocabulary
#: is enumerable rather than emergent.
EVENT_REQUEST_STARTED: Final = "agent.request.started"
EVENT_REQUEST_COMPLETED: Final = "agent.request.completed"

EVENT_PLANNER_STARTED: Final = "planner.started"
EVENT_PLANNER_COMPLETED: Final = "planner.completed"
EVENT_PLANNER_FAILED: Final = "planner.failed"

EVENT_TOOL_STARTED: Final = "tool.started"
EVENT_TOOL_COMPLETED: Final = "tool.completed"
EVENT_TOOL_FAILED: Final = "tool.failed"

EVENT_RAG_STARTED: Final = "rag.started"
EVENT_RAG_COMPLETED: Final = "rag.completed"
EVENT_RAG_UNAVAILABLE: Final = "rag.unavailable"

EVENT_LLM_STARTED: Final = "llm.started"
EVENT_LLM_COMPLETED: Final = "llm.completed"
EVENT_LLM_FAILED: Final = "llm.failed"

#: The human-readable line format. It is byte-for-byte the format this service
#: emitted before structured logging existed, so ``LOG_FORMAT=text`` is not a new
#: output shape, it is the old one.
TEXT_LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(name)s %(message)s"

#: A value longer than this is truncated and marked. A log line is not a payload.
MAX_FIELD_CHARS: Final = 200

#: Patterns that must never survive into a log, whatever the call site. Each
#: carries the replacement used, so a redacted value is recognisable as redacted
#: rather than looking like a plausible short string.
_REDACTIONS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{8,}"), r"\1 <redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9]{8,}"), "<redacted-key>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{8,}\b"), "<redacted-token>"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{12,}\b"), "<redacted-key>"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*\S+"), r"\1=<redacted>"),
    (
        re.compile(r"(?i)\b(?:https?|postgres(?:ql)?|mysql|sqlite)://\S*@\S+"),
        "<redacted-url>",
    ),
)

#: User text embedded in a legacy single-line message, scrubbed only when that
#: message is serialized into a structured record.
#:
#: The text format renders the query, and that rendering is the historical
#: operator-facing contract: three tests pin it and it is documented in the
#: README. A JSON pipeline has a different contract, and the requirement there is
#: that no user text is carried. Both hold because the scrub is applied by
#: :class:`JsonFormatter` and nowhere else. The pattern covers a Python repr,
#: which is how the existing call sites render a query, and a bare token, which
#: is how a future one might. A repr can contain spaces and escapes, so ``\S+``
#: alone would truncate the value and leave the remainder of the query visible.
_MESSAGE_REDACTIONS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(r"(?<![\w])(query=)(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)"),
        r"\1<redacted>",
    ),
)


def sanitize(value: Any) -> Any:
    """Return a log-safe rendering of ``value``.

    Non-scalars are reduced to their type name, which keeps a nested payload from
    ever reaching the output. Strings are redacted then truncated.
    """
    if value is None or isinstance(value, bool | int | float):
        return value
    if not isinstance(value, str):
        return type(value).__name__

    text = value
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    if len(text) > MAX_FIELD_CHARS:
        text = text[:MAX_FIELD_CHARS] + "...<truncated>"
    return text


def event_fields(
    event: str,
    **fields: Any,
) -> dict[str, Any]:
    """Build the structured payload for one event.

    Unknown field names are dropped rather than carried. A typo therefore loses a
    field instead of adding an undocumented one, and the JSON shape stays fixed.
    """
    payload: dict[str, Any] = {"event": event}
    for name in FIELD_ORDER:
        if name in ("timestamp", "level", "event"):
            continue
        if name in fields and fields[name] is not None:
            payload[name] = sanitize(fields[name])
    return payload


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    """Emit one structured event.

    ``message`` is the human-readable text format rendering. In JSON output the
    formatter serializes ``event`` and the declared fields and does not use the
    message, so the two formats share one call site without one degrading the
    other.
    """
    logger.log(
        level,
        message,
        extra={EXTRA_KEY: event_fields(event, **fields), "event_name": event},
    )


class JsonFormatter(logging.Formatter):
    """Serialize a record as a single JSON object.

    A record without structured fields still serializes, with its message kept as
    ``message``. That keeps a third-party library's record, or one emitted before
    configuration, from producing a malformed line.

    The message is scrubbed of embedded user text before serialization, because a
    message written for the text format can carry a query while a structured
    record must not.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as one JSON line."""
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": getattr(record, "event_name", "log"),
            "logger": record.name,
            "message": redact_message(record.getMessage()),
        }

        fields = getattr(record, EXTRA_KEY, None)
        if isinstance(fields, dict):
            for name in FIELD_ORDER:
                if name in fields and fields[name] is not None:
                    payload[name] = fields[name]

        if record.exc_info:
            # The traceback is not serialized. A provider traceback can carry the
            # request headers, and this project's error contract already forbids
            # letting either reach a log.
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "error"
        return json.dumps(payload, ensure_ascii=False)


def redact_message(message: str) -> str:
    """Return ``message`` with embedded user text removed.

    Applied by :class:`JsonFormatter` to every record it serializes. The text
    format does not call it, so the historical single-line output is unchanged.
    """
    text = message
    for pattern, replacement in _MESSAGE_REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def build_formatter(log_format: str) -> logging.Formatter:
    """Return the formatter for a configured output shape.

    ``json`` selects :class:`JsonFormatter`; anything else, including the default
    ``text``, keeps the human-readable line this service has always emitted. The
    fallback is deliberately permissive: an unrecognised value must not take the
    service down, and text is the format whose absence is immediately visible.
    """
    if log_format.strip().lower() == "json":
        return JsonFormatter()
    return logging.Formatter(TEXT_LOG_FORMAT)

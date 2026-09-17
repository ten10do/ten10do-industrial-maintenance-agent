"""Bounded label vocabularies for every metric and span attribute.

A Prometheus label is a dimension of the time series. Any value that can grow
without bound multiplies the series count, and a value taken from user input lets
a caller choose how much memory the process spends. ``request_id``,
``equipment_id``, the raw query, a document path and an exception message all
have that shape, so none of them may become a label.

The design used here is allow-listing rather than escaping.

* Each label has one frozen set of legal values, declared in this module.
* :func:`bounded` maps anything outside that set onto the ``other`` sentinel.
* The sentinel is part of every vocabulary, so the fallback is always a legal
  value and never a surprise series.

The consequence worth stating plainly is that a cardinality mistake becomes a
visible one. A new intent, tool or provider shows up as ``other`` in the metric
instead of silently creating series, and ``tests/test_observability.py`` fails
when a vocabulary and its producer drift apart.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

#: Value used for anything outside a vocabulary. Every vocabulary contains it, so
#: the number of series per label is fixed by the size of the vocabulary alone.
OTHER: Final = "other"

#: Which planner produced a plan. ``none`` means no planner ran at all.
PLANNER_LABELS: Final[frozenset[str]] = frozenset({"rule", "llm", "none", OTHER})

#: Coarse request classification, mirroring :class:`app.agent.parser.Intent`.
INTENT_LABELS: Final[frozenset[str]] = frozenset(
    {"alarm_diagnosis", "maintenance_advice", "device_status", "unknown", "none", OTHER}
)

#: Outcome of an operation.
#:
#: The four business outcomes are deliberately distinct. ``not_found`` is a
#: normal answer (the device or alarm code is simply not there) and ``unavailable``
#: means the source exists but could not be read. Reporting either of them as
#: ``error`` would turn a working service into an alarming dashboard.
STATUS_LABELS: Final[frozenset[str]] = frozenset(
    {"success", "not_found", "unavailable", "error", OTHER}
)

#: Registry tool names. Kept as a literal set so this module does not import the
#: tool package, which would create an import cycle through the registry.
TOOL_LABELS: Final[frozenset[str]] = frozenset(
    {
        "get_device_status",
        "query_alarm_code",
        "search_maintenance_manual",
        "none",
        OTHER,
    }
)

#: Integration providers. ``local`` and ``http`` are the RAG providers,
#: ``openai_compatible`` is the only shipped LLM provider.
PROVIDER_LABELS: Final[frozenset[str]] = frozenset(
    {"local", "http", "openai_compatible", "none", "unknown", OTHER}
)

#: Why a planner failed. This is a closed enumeration on purpose: an exception
#: message can embed a URL, a header or a credential, and a metric label is
#: published to every scraper.
REASON_LABELS: Final[frozenset[str]] = frozenset(
    {"invalid_output", "provider_error", "timeout", "configuration", "unknown", OTHER}
)

#: Token accounting direction. Only these two exist, so the label cannot grow.
TOKEN_TYPE_LABELS: Final[frozenset[str]] = frozenset({"prompt", "completion"})


def bounded(value: object, vocabulary: frozenset[str]) -> str:
    """Return ``value`` when it belongs to ``vocabulary``, otherwise ``other``.

    The comparison is done on the string form, so an :class:`enum.Enum` member
    and its value both map to the same label. ``None`` and the empty string map
    to ``other`` because they carry no information and would otherwise create a
    third, meaningless series.

    Args:
        value: Raw label candidate, typically taken from application state.
        vocabulary: The frozen set of legal values, from this module.

    Returns:
        A value guaranteed to be a member of ``vocabulary``.
    """
    if value is None:
        return OTHER
    # ``str``-mixin enum members and plain strings must both land on the same
    # label, so the plain value is unwrapped before comparison.
    text = value.value if isinstance(value, Enum) else str(value)
    if not text:
        return OTHER
    return text if text in vocabulary else OTHER

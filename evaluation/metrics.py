"""Metric primitives for the agent evaluation.

Every ratio goes through :func:`ratio`, which returns ``None`` when the
denominator is zero. That is the whole point of this module: a metric with no
observations must read as "not measured", never as 100 percent. A planner that
never calls a tool has no precision, and reporting ``1.0`` for it would be a
fabricated success.

The functions here are pure and take plain values, so the metric definitions can
be tested directly instead of through a pipeline run.
"""

from __future__ import annotations

from typing import Any

#: A metric value. ``None`` means the denominator was zero.
Metric = float | None


def ratio(numerator: int, denominator: int) -> Metric:
    """Return ``numerator / denominator``, or ``None`` when ``denominator`` is 0."""
    if denominator == 0:
        return None
    return numerator / denominator


def mean(values: list[float]) -> Metric:
    """Return the arithmetic mean, or ``None`` for an empty sample."""
    if not values:
        return None
    return sum(values) / len(values)


def median(values: list[float]) -> Metric:
    """Return the median, or ``None`` for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def tool_counts(expected: list[str], predicted: list[str]) -> tuple[int, int, int]:
    """Return ``(true_positive, false_positive, false_negative)`` tool counts.

    Tool selection is treated as a set problem: a tool is either planned or not,
    and a duplicate mention of the same tool neither helps nor hurts. Order is not
    scored here because the executor derives arguments per tool and the two
    depends-on relationships are captured by the argument checks instead.
    """
    expected_set = set(expected)
    predicted_set = set(predicted)
    return (
        len(expected_set & predicted_set),
        len(predicted_set - expected_set),
        len(expected_set - predicted_set),
    )


def missing_tools(expected: list[str], predicted: list[str]) -> list[str]:
    """Return expected tools that were not planned, sorted."""
    return sorted(set(expected) - set(predicted))


def unnecessary_tools(expected: list[str], predicted: list[str]) -> list[str]:
    """Return planned tools that the case did not expect, sorted.

    This is the false-positive set. For an out-of-domain case it is the complete
    list of harmful calls, because nothing was expected.
    """
    return sorted(set(predicted) - set(expected))


def invalid_tools(predicted: list[str], known: set[str]) -> list[str]:
    """Return planned names that do not exist in the registry, sorted.

    A planner that invents a tool name has produced a plan the executor cannot
    run. That is a different failure from choosing a real but unnecessary tool,
    so the two are counted separately.
    """
    return sorted(set(predicted) - known)


def argument_values_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Return ``True`` when every declared expected argument matches.

    Only the keys the case declares are compared. Optional arguments a planner
    legitimately adds, such as ``top_k`` on the manual search, are not treated as
    errors, and an argument the case does not constrain is not scored.
    """
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


__all__ = [
    "Metric",
    "argument_values_match",
    "invalid_tools",
    "mean",
    "median",
    "missing_tools",
    "ratio",
    "tool_counts",
    "unnecessary_tools",
]

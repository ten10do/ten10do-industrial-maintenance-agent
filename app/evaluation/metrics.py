"""Placeholder evaluation utilities.

Scoring is deliberately simple at this stage and will be replaced by
domain-specific metrics once the workflow is functional.
"""

from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """A single query / reference answer pair."""

    query: str
    expected: str
    predicted: str = ""


@dataclass
class EvalSummary:
    """Aggregated evaluation result."""

    total: int = 0
    exact_match: float = 0.0
    contains_match: float = 0.0
    details: list[dict] = field(default_factory=list)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def evaluate(samples: list[EvalSample]) -> EvalSummary:
    """Compute simple overlap metrics over a list of samples."""
    if not samples:
        return EvalSummary()

    exact_hits = 0
    contains_hits = 0
    details: list[dict] = []

    for sample in samples:
        expected = _normalize(sample.expected)
        predicted = _normalize(sample.predicted)
        is_exact = predicted == expected and expected != ""
        is_contains = expected != "" and expected in predicted

        exact_hits += int(is_exact)
        contains_hits += int(is_contains or is_exact)
        details.append(
            {
                "query": sample.query,
                "exact_match": is_exact,
                "contains_match": is_contains or is_exact,
            }
        )

    total = len(samples)
    return EvalSummary(
        total=total,
        exact_match=exact_hits / total,
        contains_match=contains_hits / total,
        details=details,
    )

"""Evaluation dataset loading and answer-key validation.

The dataset is hand-authored JSON. This module turns it into validated models and
refuses a malformed answer key, so a typo cannot silently deflate a metric.

The integrity rules enforced here are authoring invariants, not scoring choices:

* case ids are unique, because every report is keyed by them;
* every case carries a category that the dataset's own ``categories`` map
  documents, so a breakdown can never reference an undocumented bucket;
* every tool named in ``expected_arguments`` is also named in
  ``expected_tools``, so the argument check cannot require a tool the case does
  not expect;
* ``expected_intent`` uses the parser's vocabulary, keeping one source of truth;
* an out-of-domain case expects no tool and no intent, and a case that expects no
  intent is out of domain. The two must agree, otherwise the OOD metrics would be
  computed over a set that is not actually out of domain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.parser import Intent

#: Category whose cases must not trigger any tool call.
OOD_CATEGORY = "ood"

#: The intent vocabulary, taken from the parser so it cannot drift.
INTENT_VOCABULARY: frozenset[str] = frozenset(member.value for member in Intent)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


class EvalCase(BaseModel):
    """One hand-authored evaluation case."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    intent: str | None = Field(
        default=None,
        alias="expected_intent",
        description="Ground-truth intent, or None for an out-of-domain case.",
    )
    tools: list[str] = Field(
        default_factory=list,
        alias="expected_tools",
        description="Ground-truth tool set. Order is not significant.",
    )
    arguments: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        alias="expected_arguments",
        description="Ground-truth arguments, keyed by tool name.",
    )
    notes: str | None = None

    @property
    def is_ood(self) -> bool:
        """Return ``True`` when the case is out of domain."""
        return self.category == OOD_CATEGORY

    @model_validator(mode="after")
    def _check_cross_fields(self) -> EvalCase:
        if self.intent is not None and self.intent not in INTENT_VOCABULARY:
            raise ValueError(f"{self.id}: unknown expected_intent {self.intent!r}")

        undeclared = sorted(set(self.arguments) - set(self.tools))
        if undeclared:
            raise ValueError(
                f"{self.id}: expected_arguments names tools absent from "
                f"expected_tools: {', '.join(undeclared)}"
            )

        if self.is_ood:
            if self.tools:
                raise ValueError(f"{self.id}: an out-of-domain case must expect no tool")
            if self.intent is not None:
                raise ValueError(f"{self.id}: an out-of-domain case must not expect an intent")
        elif self.intent is None:
            raise ValueError(f"{self.id}: only an out-of-domain case may omit expected_intent")

        return self


class EvalDataset(BaseModel):
    """A validated evaluation dataset."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str
    name: str
    description: str
    policy: dict[str, str] = Field(default_factory=dict, alias="ground_truth_policy")
    categories: dict[str, str] = Field(default_factory=dict)
    cases: list[EvalCase] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_cases(self) -> EvalDataset:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id}")
            seen.add(case.id)
            if self.categories and case.category not in self.categories:
                raise ValueError(f"{case.id}: undocumented category {case.category!r}")
        return self

    @property
    def category_counts(self) -> dict[str, int]:
        """Return the number of cases per category, in dataset order."""
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.category] = counts.get(case.category, 0) + 1
        return counts

    @property
    def ood_cases(self) -> list[EvalCase]:
        """Return the out-of-domain cases."""
        return [case for case in self.cases if case.is_ood]

    def case_by_id(self, case_id: str) -> EvalCase:
        """Return the case with ``case_id``.

        Raises:
            KeyError: No such case.
        """
        for case in self.cases:
            if case.id == case_id:
                return case
        raise KeyError(case_id)


def dataset_sha256(path: Path | str) -> str:
    """Return the SHA-256 of the dataset file, so a report names its exact input."""
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def parse_dataset(payload: dict[str, Any]) -> EvalDataset:
    """Validate a decoded dataset payload.

    Raises:
        pydantic.ValidationError: The answer key violates an authoring invariant.
    """
    return EvalDataset.model_validate(payload)


def load_dataset(path: Path | str = DEFAULT_DATASET_PATH) -> EvalDataset:
    """Load and validate the dataset at ``path``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_dataset(payload)


__all__ = [
    "DEFAULT_DATASET_PATH",
    "INTENT_VOCABULARY",
    "OOD_CATEGORY",
    "PROJECT_ROOT",
    "EvalCase",
    "EvalDataset",
    "dataset_sha256",
    "load_dataset",
    "parse_dataset",
]

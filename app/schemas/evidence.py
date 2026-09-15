"""Evidence model produced by the synthesis step.

Each record captures one fact the agent relied on together with where it came
from, so a maintenance answer can be traced back to its source.

Records with ``source_type=tool`` describe local lookups. Records with
``source_type=document`` come from RAG retrieval and carry the manual citation
fields ``document`` / ``page`` / ``score``, plus ``score_semantics`` and
``higher_is_better`` so the score's meaning travels with it.
"""

from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Where an evidence record came from."""

    TOOL = "tool"
    DOCUMENT = "document"


class Evidence(BaseModel):
    """A single citable fact."""

    source_type: SourceType = Field(
        default=SourceType.TOOL,
        description="Origin category of the fact.",
    )
    source: str = Field(
        ...,
        description="Originating resource, for example data/alarms.json or sqlite:devices.",
    )
    tool_name: str = Field(
        ...,
        description="Registry name of the tool that produced the fact.",
    )
    content: str = Field(
        ...,
        description="Fact text, usable directly in the final answer.",
    )

    # Populated only for ``source_type=document`` evidence produced by the RAG
    # retrieval tool.
    document: str | None = Field(default=None, description="Source document name.")
    page: int | None = Field(default=None, description="Page number in the document.")
    score: float | None = Field(default=None, description="Retrieval relevance score.")
    score_semantics: str | None = Field(
        default=None,
        description=(
            "What the score measures, for example vector_cosine_distance or "
            "lexical_relevance_flag. The value is never converted into another "
            "scale."
        ),
    )
    higher_is_better: bool | None = Field(
        default=None,
        description=(
            "Direction of the score. The RAG engine exposes distance-like and "
            "flag-like quantities, so this is False whenever a score is labelled."
        ),
    )

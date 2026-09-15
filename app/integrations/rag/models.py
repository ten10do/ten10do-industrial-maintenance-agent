"""Transport-neutral models for manual retrieval.

These models are the contract every RAG provider must satisfy. They mirror the
fields the Industrial Knowledge RAG service actually returns (``content``,
``source``/``document``, ``page``, ``section``, ``chunk_id``, ``score``) and
nothing more. A field that the upstream system did not supply stays ``None``;
it is never back-filled with a guess.
"""

from pydantic import BaseModel, Field


class RAGSearchRequest(BaseModel):
    """A retrieval request for the maintenance manual knowledge base."""

    query: str = Field(..., min_length=1, description="Natural-language query text.")
    top_k: int = Field(default=4, ge=1, le=8, description="Maximum hits requested.")


class RAGSearchHit(BaseModel):
    """One retrieved manual fragment.

    ``document``, ``page``, ``section``, ``chunk_id`` and ``score`` are optional
    because the upstream retrieval result may omit them for a given corpus
    configuration. They stay ``None`` when absent.

    ``score_semantics`` and ``higher_is_better`` label what ``score`` actually
    means and which direction is better. The value is never reinterpreted, and
    the RAG engine's scores are distance-like or flag-like, so
    ``higher_is_better`` is ``False`` whenever a score is annotated. See
    :mod:`app.integrations.rag.scores`.
    """

    content: str = Field(..., description="Retrieved text fragment.")
    document: str | None = Field(default=None, description="Source document name.")
    page: int | None = Field(default=None, description="One-based page number.")
    section: str | None = Field(default=None, description="Section heading, when known.")
    chunk_id: str | None = Field(default=None, description="Upstream chunk identifier.")
    score: float | None = Field(default=None, description="Upstream relevance score.")
    score_semantics: str | None = Field(
        default=None,
        description="What the score measures, for example vector_cosine_distance.",
    )
    higher_is_better: bool | None = Field(
        default=None,
        description="Direction of the score. False means smaller is more relevant.",
    )


class RAGSearchResponse(BaseModel):
    """Provider-agnostic retrieval response."""

    query: str = Field(..., description="Echo of the executed query.")
    hits: list[RAGSearchHit] = Field(default_factory=list)
    retrieval_mode: str | None = Field(
        default=None,
        description="Retrieval mode reported by the provider, for example hybrid.",
    )

    @property
    def found(self) -> bool:
        """Return ``True`` when at least one fragment was retrieved."""
        return bool(self.hits)

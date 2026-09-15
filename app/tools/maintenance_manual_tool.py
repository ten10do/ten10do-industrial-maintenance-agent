"""Maintenance manual retrieval tool.

Wraps the Industrial Knowledge RAG retrieval behind a tool with explicit
Pydantic input and output models. The tool itself contains no retrieval logic:
it delegates to a :class:`~app.integrations.rag.base.RAGProvider` and only maps
the result onto the tool contract.

Two failure modes are kept distinct on purpose, because collapsing them would
let a broken deployment masquerade as "the manual has nothing on this":

* retrieval succeeded and returned nothing -> ``found`` is ``False`` and
  ``error`` is ``None``;
* retrieval could not run at all -> ``found`` is ``False`` and ``error`` carries
  the provider diagnostic.

No field is ever invented. When the upstream result omits a page, section or
score, the value stays ``None``.

This tool performs no LLM call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import get_settings
from app.integrations.rag import factory
from app.integrations.rag.base import RAGProvider, RAGProviderError
from app.integrations.rag.models import RAGSearchHit

MANUAL_TOOL_SOURCE = "rag:maintenance_manual"


class ManualSearchInput(BaseModel):
    """Input model for :func:`search_maintenance_manual`."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language query, for example an alarm description.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description="Maximum number of fragments. Falls back to configuration.",
    )


class ManualSearchResult(RAGSearchHit):
    """One retrieved maintenance manual fragment."""


class ManualSearchOutput(BaseModel):
    """Structured manual retrieval result.

    ``found`` is ``True`` only when at least one fragment was retrieved.
    """

    query: str
    found: bool = False
    results: list[ManualSearchResult] = Field(default_factory=list)
    provider: str = ""
    retrieval_mode: str | None = None
    error: str | None = None


def _resolve_provider(provider: RAGProvider | None) -> RAGProvider:
    """Return the injected provider or the configured default."""
    if provider is not None:
        return provider
    return factory.get_rag_provider()


def search_maintenance_manual(
    query: str,
    top_k: int | None = None,
    provider: RAGProvider | None = None,
) -> ManualSearchOutput:
    """Retrieve maintenance manual fragments relevant to ``query``.

    Args:
        query: Query text.
        top_k: Optional hit limit. When omitted the configured default is used.
        provider: Optional provider override, mainly for tests and for callers
            that want to pin a specific knowledge base.
    """
    payload = ManualSearchInput(query=query, top_k=top_k)
    limit = payload.top_k if payload.top_k is not None else get_settings().rag_top_k

    try:
        active = _resolve_provider(provider)
    except RAGProviderError as exc:
        return ManualSearchOutput(query=payload.query, error=str(exc))

    provider_id = active.provider_id

    try:
        response = active.search(payload.query, top_k=limit)
    except RAGProviderError as exc:
        return ManualSearchOutput(
            query=payload.query,
            provider=provider_id,
            error=str(exc),
        )

    results = [
        ManualSearchResult(
            content=hit.content,
            document=hit.document,
            page=hit.page,
            section=hit.section,
            chunk_id=hit.chunk_id,
            score=hit.score,
            score_semantics=hit.score_semantics,
            higher_is_better=hit.higher_is_better,
        )
        for hit in response.hits
    ]

    return ManualSearchOutput(
        query=payload.query,
        found=bool(results),
        results=results,
        provider=provider_id,
        retrieval_mode=response.retrieval_mode,
    )

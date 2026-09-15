"""HTTP provider for the Industrial Knowledge RAG service.

The RAG service exposes exactly one retrieval-capable route: ``POST /ask``.
There is no retrieval-only endpoint. ``/ask`` therefore runs retrieval *and*
the service's own answer generator, and it returns both surfaces:

* ``sources``: the retrieval evidence, with ``content``, ``source``, ``page``,
  ``section``, ``chunk_id`` and ``score`` per fragment;
* ``answer``: an LLM-generated summary.

This provider consumes ``sources`` only and discards ``answer``, so the agent
workflow itself stays deterministic and never reads LLM output. The upstream
generation cost and latency are the reason the ``local`` provider is the
default; this provider exists for the case where the RAG service is deployed
remotely and the agent must not hold a checkout of it.

Every URL, port, timeout and knowledge base identifier is supplied by
configuration. Nothing is hardcoded here.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import httpx

from app.integrations.rag.base import RAGProvider, RAGProviderError
from app.integrations.rag.models import RAGSearchHit, RAGSearchResponse
from app.integrations.rag.scores import describe_score, has_vector_component

KNOWLEDGE_BASE_HEADER = "X-Knowledge-Base-ID"
MIN_KNOWLEDGE_BASE_ID_LENGTH = 19
MAX_KNOWLEDGE_BASE_ID_LENGTH = 67


def _mode_from_retrieval_source(retrieval_source: str) -> str | None:
    """Infer the retrieval side from a ``SourceItem.retrieval_source`` value.

    The RAG service reports ``lexical``, ``vector`` or ``hybrid`` per source, so
    the score's meaning can be established from the payload itself rather than
    from an assumption about the deployment.
    """
    value = retrieval_source.lower()
    for mode in ("hybrid", "vector", "lexical"):
        if mode in value:
            return mode
    return None


class HttpRAGProvider(RAGProvider):
    """Retrieval provider that talks to the RAG FastAPI service."""

    provider_id = "http"

    def __init__(
        self,
        base_url: str,
        knowledge_base_id: str,
        *,
        timeout_seconds: float = 10.0,
        model_provider: str = "DeepSeek",
        backend: str = "light",
        ask_path: str = "/ask",
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("RAG base URL is required for the HTTP provider.")
        if not knowledge_base_id.strip():
            raise ValueError("Knowledge base id is required for the HTTP provider.")
        if not (
            MIN_KNOWLEDGE_BASE_ID_LENGTH <= len(knowledge_base_id) <= MAX_KNOWLEDGE_BASE_ID_LENGTH
        ):
            raise ValueError(
                "Knowledge base id must be "
                f"{MIN_KNOWLEDGE_BASE_ID_LENGTH}-{MAX_KNOWLEDGE_BASE_ID_LENGTH} "
                "characters, as required by the RAG service."
            )

        self.base_url = base_url.rstrip("/") + "/"
        self.knowledge_base_id = knowledge_base_id
        self.timeout_seconds = timeout_seconds
        self.model_provider = model_provider
        self.backend = backend
        self.ask_url = urljoin(self.base_url, ask_path.lstrip("/"))
        self._client = client

    def describe(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "base_url": self.base_url,
            "knowledge_base_id": self.knowledge_base_id,
            "timeout_seconds": self.timeout_seconds,
            "backend": self.backend,
        }

    def _request(self, query: str, top_k: int) -> httpx.Response:
        """POST the retrieval request to the RAG service."""
        payload = {
            "question": query,
            "top_k": top_k,
            "model_provider": self.model_provider,
        }
        headers = {
            KNOWLEDGE_BASE_HEADER: self.knowledge_base_id,
            "Content-Type": "application/json",
        }
        if self._client is not None:
            return self._client.post(self.ask_url, json=payload, headers=headers)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(self.ask_url, json=payload, headers=headers)

    def _to_hit(self, source: dict[str, Any]) -> RAGSearchHit:
        """Map one ``SourceItem`` payload onto a transport model."""
        raw_page = source.get("page")
        page = raw_page if isinstance(raw_page, int) and not isinstance(raw_page, bool) else None

        raw_score = source.get("score")
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score = None

        section = source.get("section") or None
        chunk_id = source.get("chunk_id") or None

        retrieval_source = str(source.get("retrieval_source") or "")
        semantics = describe_score(
            _mode_from_retrieval_source(retrieval_source),
            has_vector_component=has_vector_component(
                retrieval_source,
                source.get("vector_rank") if isinstance(source.get("vector_rank"), int) else None,
            ),
            backend=self.backend,
        )

        return RAGSearchHit(
            content=str(source.get("content", "")),
            document=str(source["source"]) if source.get("source") else None,
            page=page,
            section=str(section) if section else None,
            chunk_id=str(chunk_id) if chunk_id else None,
            score=score,
            score_semantics=semantics.semantics if score is not None else None,
            higher_is_better=semantics.higher_is_better if score is not None else None,
        )

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        """Retrieve manual fragments from the RAG service."""
        try:
            response = self._request(query, top_k)
        except httpx.HTTPError as exc:
            raise RAGProviderError(f"RAG service unreachable at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise RAGProviderError(
                f"RAG service returned HTTP {response.status_code} for {self.ask_url}."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RAGProviderError("RAG service returned a non-JSON response.") from exc

        if not isinstance(payload, dict):
            raise RAGProviderError("RAG service returned an unexpected payload shape.")

        # The service refuses when its own evidence gate is not satisfied.
        # That is an upstream decision and is honoured rather than overridden.
        if payload.get("is_refused"):
            return RAGSearchResponse(query=query, hits=[], retrieval_mode=None)

        sources = payload.get("sources") or []
        hits = [self._to_hit(item) for item in sources if isinstance(item, dict)]
        return RAGSearchResponse(query=query, hits=hits, retrieval_mode=None)

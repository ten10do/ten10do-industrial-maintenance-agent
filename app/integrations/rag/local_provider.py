"""In-process RAG provider.

This provider calls the Industrial Knowledge RAG retrieval engine directly, in
the current process, through its public Python function
``light_rag_core.retrieve_docs`` (or ``rag_core.retrieve_docs`` in ``full``
mode). It is the only integration path that is free of any LLM call: the HTTP
surface of the RAG service exposes retrieval exclusively through ``POST /ask``,
which always runs the service's answer generator, whereas ``retrieve_docs`` is
the pure retrieval core that ``/ask`` itself consumes.

The external repository is treated as read-only. This module imports it, never
writes to it, and never copies its source.

Two side effects of importing the RAG package are worth stating explicitly:

* the RAG package root is prepended to ``sys.path`` so ``backend`` resolves as a
  package (the repository is not installed as a distribution);
* ``backend.llm_client`` loads the RAG repository ``.env`` on import, so the RAG
  process settings become visible in this process environment.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from app.integrations.rag.base import RAGProvider, RAGProviderError
from app.integrations.rag.models import RAGSearchHit, RAGSearchResponse
from app.integrations.rag.scores import ScoreSemantics, describe_score, has_vector_component

BACKEND_MODULES: dict[str, str] = {
    "light": "light_rag_core",
    "full": "rag_core",
}


def _as_page(value: Any) -> int | None:
    """Convert a zero-based upstream page index into a one-based page number."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value + 1


class LocalRAGProvider(RAGProvider):
    """Retrieval provider backed by the RAG repository source checkout."""

    provider_id = "local"

    def __init__(
        self,
        repo_root: str | Path,
        *,
        knowledge_base_id: str = "default",
        rag_backend: str = "light",
        retrieval_mode: str | None = None,
    ) -> None:
        if not str(repo_root).strip():
            raise ValueError("RAG repo root is required for the local provider.")
        if rag_backend not in BACKEND_MODULES:
            supported = ", ".join(sorted(BACKEND_MODULES))
            raise ValueError(f"Unsupported RAG backend: {rag_backend} (expected {supported}).")

        self.repo_root = Path(repo_root).expanduser()
        self.knowledge_base_id = knowledge_base_id or "default"
        self.rag_backend = rag_backend
        self.retrieval_mode = retrieval_mode
        self._module: Any | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "repo_root": str(self.repo_root),
            "rag_backend": self.rag_backend,
            "knowledge_base_id": self.knowledge_base_id,
        }

    def _load_module(self) -> Any:
        """Import and cache the RAG retrieval module.

        The import is deferred so that merely constructing an agent, or running
        tests that never touch the RAG tool, does not require the sibling
        repository or its dependencies to be present.
        """
        if self._module is not None:
            return self._module

        if not self.repo_root.is_dir():
            raise RAGProviderError(f"RAG repository not found: {self.repo_root}")

        root = str(self.repo_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        module_name = f"backend.{BACKEND_MODULES[self.rag_backend]}"
        try:
            self._module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RAGProviderError(
                f"Cannot import {module_name} from {self.repo_root}: {exc}"
            ) from exc
        return self._module

    def _to_hit(
        self,
        document: Any,
        score: Any,
        semantics: ScoreSemantics,
    ) -> RAGSearchHit:
        """Map one upstream ``(document, score)`` pair onto a transport model."""
        metadata = getattr(document, "metadata", None) or {}

        document_name = metadata.get("source")
        document_name = Path(str(document_name)).name if document_name else None

        page = _as_page(metadata.get("page"))
        if page is None:
            page = _as_page(metadata.get("page_start"))

        section = metadata.get("section") or None
        chunk_id = metadata.get("chunk_id") or None

        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = None

        return RAGSearchHit(
            content=str(getattr(document, "page_content", "") or ""),
            document=document_name,
            page=page,
            section=str(section) if section else None,
            chunk_id=str(chunk_id) if chunk_id else None,
            score=numeric_score,
            score_semantics=semantics.semantics if numeric_score is not None else None,
            higher_is_better=semantics.higher_is_better if numeric_score is not None else None,
        )

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        """Run ``retrieve_docs`` in-process and normalize the result."""
        module = self._load_module()

        try:
            result = module.retrieve_docs(
                query,
                k=top_k,
                knowledge_base_id=self.knowledge_base_id,
                retrieval_mode=self.retrieval_mode,
            )
        except Exception as exc:
            raise RAGProviderError(str(exc)) from exc

        retrieval_mode = getattr(result, "retrieval_mode", None) or None
        # ``RetrievalResult`` is a list built from ``candidates`` in order, so the
        # index alignment below is the same one the RAG service's own
        # ``serialize_sources`` relies on.
        candidates = list(getattr(result, "candidates", None) or [])

        hits: list[RAGSearchHit] = []
        for index, (document, score) in enumerate(result):
            candidate = candidates[index] if index < len(candidates) else None
            vector_side = has_vector_component(
                getattr(candidate, "retrieval_source", None),
                getattr(candidate, "vector_rank", None),
            )
            semantics = describe_score(
                retrieval_mode,
                has_vector_component=vector_side,
                backend=self.rag_backend,
            )
            hits.append(self._to_hit(document, score, semantics))

        return RAGSearchResponse(query=query, hits=hits, retrieval_mode=retrieval_mode)

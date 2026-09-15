"""Provider selection for manual retrieval.

The provider is chosen by configuration and built lazily. Neither provider
imports the sibling RAG repository or opens a socket at import time, so the
agent remains fully usable when the knowledge base is absent.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.integrations.rag.base import RAGProvider, RAGProviderError
from app.integrations.rag.http_provider import HttpRAGProvider
from app.integrations.rag.local_provider import LocalRAGProvider

SUPPORTED_PROVIDERS: tuple[str, ...] = ("local", "http")


def build_provider(settings: Settings) -> RAGProvider:
    """Construct the provider named by ``settings.rag_provider``.

    Raises:
        RAGProviderError: when the selected provider is unknown or its required
            configuration is missing. The error is raised here, at construction
            time, so a misconfigured deployment fails loudly instead of
            returning an empty result that looks like "no evidence found".
    """
    selected = (settings.rag_provider or "").strip().lower()

    if selected == "local":
        try:
            return LocalRAGProvider(
                repo_root=settings.rag_repo_root,
                knowledge_base_id=settings.rag_knowledge_base_id,
                rag_backend=settings.rag_backend,
                retrieval_mode=settings.rag_retrieval_mode,
            )
        except ValueError as exc:
            raise RAGProviderError(str(exc)) from exc

    if selected == "http":
        try:
            return HttpRAGProvider(
                base_url=settings.rag_base_url,
                knowledge_base_id=settings.rag_knowledge_base_id,
                timeout_seconds=settings.rag_timeout_seconds,
                model_provider=settings.rag_http_model_provider,
                backend=settings.rag_backend,
            )
        except ValueError as exc:
            raise RAGProviderError(str(exc)) from exc

    supported = ", ".join(SUPPORTED_PROVIDERS)
    raise RAGProviderError(f"Unknown RAG provider: {selected or '(empty)'} (expected {supported}).")


@lru_cache(maxsize=1)
def get_rag_provider() -> RAGProvider:
    """Return the configured provider, constructing it once per process."""
    return build_provider(get_settings())


def reset_rag_provider() -> None:
    """Drop the cached provider so the next call rebuilds it."""
    get_rag_provider.cache_clear()

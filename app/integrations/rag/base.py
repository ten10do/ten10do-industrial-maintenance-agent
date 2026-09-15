"""Provider interface for maintenance manual retrieval."""

from abc import ABC, abstractmethod
from typing import Any

from app.integrations.rag.models import RAGSearchResponse


class RAGProviderError(RuntimeError):
    """Raised when a RAG provider cannot produce a retrieval result.

    The error is raised rather than swallowed so the tool layer can surface it
    as an explicit ``error`` field instead of reporting a fabricated empty
    result. An unreachable knowledge base and an empty knowledge base are
    different situations and must stay distinguishable.
    """


class RAGProvider(ABC):
    """Abstract retrieval provider for the Industrial Knowledge RAG system.

    Implementations must be side-effect free with respect to the agent: they
    take a query and return hits, and they never mutate agent state.
    """

    provider_id: str = "rag"

    @abstractmethod
    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        """Retrieve manual fragments relevant to ``query``.

        Raises:
            RAGProviderError: when the knowledge base cannot be queried.
        """

    def describe(self) -> dict[str, Any]:
        """Return non-secret provider metadata, useful for diagnostics."""
        return {"provider_id": self.provider_id}

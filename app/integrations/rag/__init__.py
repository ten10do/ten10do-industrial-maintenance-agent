"""RAG integration package.

Public surface:

* :class:`RAGProvider` - the interface the agent depends on;
* :class:`RAGSearchHit` / :class:`RAGSearchResponse` - the transport-neutral
  retrieval models;
* :func:`get_rag_provider` - config-driven provider selection.
"""

from app.integrations.rag.base import RAGProvider, RAGProviderError
from app.integrations.rag.factory import (
    SUPPORTED_PROVIDERS,
    build_provider,
    get_rag_provider,
    reset_rag_provider,
)
from app.integrations.rag.http_provider import HttpRAGProvider
from app.integrations.rag.local_provider import LocalRAGProvider
from app.integrations.rag.models import RAGSearchHit, RAGSearchRequest, RAGSearchResponse
from app.integrations.rag.scores import (
    LEXICAL_RELEVANCE_FLAG,
    VECTOR_COSINE_DISTANCE,
    VECTOR_STORE_DISTANCE,
    ScoreSemantics,
    comparable,
    describe_score,
)

__all__ = [
    "LEXICAL_RELEVANCE_FLAG",
    "SUPPORTED_PROVIDERS",
    "VECTOR_COSINE_DISTANCE",
    "VECTOR_STORE_DISTANCE",
    "HttpRAGProvider",
    "LocalRAGProvider",
    "RAGProvider",
    "RAGProviderError",
    "RAGSearchHit",
    "RAGSearchRequest",
    "RAGSearchResponse",
    "ScoreSemantics",
    "build_provider",
    "comparable",
    "describe_score",
    "get_rag_provider",
    "reset_rag_provider",
]

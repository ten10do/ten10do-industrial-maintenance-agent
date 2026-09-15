"""Adapters that connect the maintenance agent to external systems.

The RAG and LLM integrations live here. Nothing in this package imports an
external dependency at module import time: every coupling is resolved lazily
inside the provider that needs it, so the agent stays importable and testable
without a RAG checkout, a knowledge base or a configured LLM endpoint.
"""

from app.integrations.llm import (
    LLMCompletionRequest,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
    get_llm_provider,
    is_llm_configured,
    reset_llm_provider,
)
from app.integrations.rag import (
    RAGProvider,
    RAGProviderError,
    RAGSearchHit,
    RAGSearchResponse,
    ScoreSemantics,
    describe_score,
    get_rag_provider,
    reset_rag_provider,
)

__all__ = [
    "LLMCompletionRequest",
    "LLMCompletionResult",
    "LLMMessage",
    "LLMProvider",
    "LLMProviderError",
    "LLMTimeoutError",
    "RAGProvider",
    "RAGProviderError",
    "RAGSearchHit",
    "RAGSearchResponse",
    "ScoreSemantics",
    "describe_score",
    "get_llm_provider",
    "get_rag_provider",
    "is_llm_configured",
    "reset_llm_provider",
    "reset_rag_provider",
]

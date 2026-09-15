"""LLM integration package.

Public surface:

* :class:`LLMProvider` - the interface the planner depends on;
* :class:`LLMCompletionRequest` / :class:`LLMCompletionResult` - the
  vendor-neutral transport models;
* :class:`LLMProviderError` / :class:`LLMTimeoutError` - the provider error
  taxonomy;
* :func:`get_llm_provider` / :func:`build_provider` - config-driven selection;
* :func:`is_llm_configured` - the smoke-test gate.

The provider library is imported lazily by the factory, so importing this
package costs nothing and never requires a configured endpoint.
"""

from app.integrations.llm.base import (
    LLMError,
    LLMErrorCode,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
)
from app.integrations.llm.factory import (
    SUPPORTED_PROVIDERS,
    build_provider,
    get_llm_provider,
    is_llm_configured,
    reset_llm_provider,
)
from app.integrations.llm.models import LLMCompletionRequest, LLMCompletionResult, LLMMessage
from app.integrations.llm.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "SUPPORTED_PROVIDERS",
    "LLMCompletionRequest",
    "LLMCompletionResult",
    "LLMError",
    "LLMErrorCode",
    "LLMMessage",
    "LLMProvider",
    "LLMProviderError",
    "LLMTimeoutError",
    "OpenAICompatibleProvider",
    "build_provider",
    "get_llm_provider",
    "is_llm_configured",
    "reset_llm_provider",
]

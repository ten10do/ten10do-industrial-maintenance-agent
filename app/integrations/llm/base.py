"""LLM provider interface and its error taxonomy.

The provider is deliberately narrow: it sends messages and returns text. It does
no planning, no validation and no tool selection, so the planning contract stays
in the agent layer where it can be tested without a network.

Error codes follow the V0.5 planner taxonomy:

* ``LLM_PROVIDER_ERROR`` - the endpoint is unreachable, misconfigured or answered
  with something unusable (a non-2xx status, a malformed envelope);
* ``LLM_TIMEOUT`` - the endpoint did not answer inside the deadline.

The remaining planner codes, ``INVALID_PLANNER_OUTPUT``, ``UNKNOWN_TOOL`` and
``TOOL_ARGUMENT_VALIDATION_FAILED``, are raised by the planner, not here.

Provider error messages are constructed from status codes and transport class
names only. Response bodies and headers are never included: a body can echo a
request, and headers carry the credential.
"""

from abc import ABC, abstractmethod

from app.integrations.llm.models import LLMCompletionRequest, LLMCompletionResult


class LLMErrorCode:
    """Provider-level error codes."""

    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"


class LLMError(Exception):
    """Base class for LLM provider failures."""

    code = LLMErrorCode.LLM_PROVIDER_ERROR

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class LLMProviderError(LLMError):
    """The provider could not be reached, configured or parsed."""

    code = LLMErrorCode.LLM_PROVIDER_ERROR


class LLMTimeoutError(LLMProviderError):
    """The provider exceeded the configured deadline."""

    code = LLMErrorCode.LLM_TIMEOUT


class LLMProvider(ABC):
    """Abstract completion provider."""

    provider_id: str = "unknown"

    @abstractmethod
    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResult:
        """Run one completion.

        Implementations must raise :class:`LLMProviderError` or
        :class:`LLMTimeoutError` rather than returning a partial result, so a
        caller can never mistake a failed call for an empty plan.
        """

    def describe(self) -> dict[str, str]:
        """Return non-sensitive provider metadata for diagnostics.

        The API key is never part of this mapping.
        """
        return {"provider_id": self.provider_id}

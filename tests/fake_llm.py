"""Test doubles for the optional LLM planner.

Every planner test runs without a network. The provider is a plain object that
returns queued text or raises a queued error, and it records the requests it
received so a test can assert what the planner actually sent.

The double also carries a test-only sentinel constant. Tests assert it never
reaches a response body, the agent state or a log line, which is the property the
real provider must preserve.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import SecretStr

from app.integrations.llm import (
    LLMCompletionRequest,
    LLMCompletionResult,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
)

#: A sentinel that must never reach a log, a state or a response body. The value
#: is deliberately not shaped like any real provider credential, so a leak of it
#: cannot be mistaken for a live key and no secret scanner has to triage it.
FAKE_API_KEY = "TEST_SECRET_DO_NOT_LEAK_LLM_PROVIDER_KEY_7f3a91c42d68"

#: Base URL used by planner tests. The host is not expected to resolve.
FAKE_BASE_URL = "https://llm.invalid/v1"

#: Model identifier used by planner tests.
FAKE_MODEL = "fake-planner-model"


class FakeLLMProvider(LLMProvider):
    """Provider double that returns a fixed completion or raises a fixed error.

    ``complete`` records every request, which lets a test assert the planner
    built the message list it promised (system prompt plus the user's query) and
    never attached the credential to a message body.
    """

    provider_id = "fake"

    def __init__(
        self,
        *,
        text: str = "{}",
        error: BaseException | None = None,
        model: str | None = FAKE_MODEL,
        latency_ms: float = 1.0,
        finish_reason: str | None = "stop",
    ) -> None:
        self._text = text
        self._error = error
        self._model = model
        self._latency_ms = latency_ms
        self._finish_reason = finish_reason
        self.requests: list[LLMCompletionRequest] = []

    @property
    def call_count(self) -> int:
        """Return how many completions were requested."""
        return len(self.requests)

    @property
    def last_request(self) -> LLMCompletionRequest | None:
        """Return the most recent request, or ``None`` when never called."""
        return self.requests[-1] if self.requests else None

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResult:
        """Record the request, then return the queued result or raise."""
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return LLMCompletionResult(
            text=self._text,
            provider=self.provider_id,
            model=self._model,
            latency_ms=self._latency_ms,
            finish_reason=self._finish_reason,
        )


def plan_text(
    tool_calls: list[dict[str, Any]] | None = None,
    *,
    intent: str | None = None,
) -> str:
    """Render a planner response body as the model would return it."""
    return json.dumps({"intent": intent, "tool_calls": tool_calls or []})


def timed_out() -> LLMTimeoutError:
    """Return the error a provider raises when it exceeds its deadline."""
    return LLMTimeoutError("LLM request exceeded 30.0s (TimeoutException)")


def provider_failed(detail: str = "HTTP 502") -> LLMProviderError:
    """Return the error a provider raises when the endpoint misbehaves."""
    return LLMProviderError(f"LLM endpoint returned {detail}")


def fake_secret() -> SecretStr:
    """Return the test-only sentinel wrapped for the settings model."""
    return SecretStr(FAKE_API_KEY)


__all__ = [
    "FAKE_API_KEY",
    "FAKE_BASE_URL",
    "FAKE_MODEL",
    "FakeLLMProvider",
    "fake_secret",
    "plan_text",
    "provider_failed",
    "timed_out",
]

"""Vendor-neutral models for LLM completion requests and results.

The agent layer never sees a vendor SDK object. A provider receives a request in
these terms and returns a result in these terms, so swapping the endpoint behind
an OpenAI-compatible API is a configuration change rather than a code change.

The result carries only the completion text and metadata. Nothing here holds an
API key, and no model carries the request headers.
"""

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """One chat message."""

    role: str = Field(..., description="One of system, user or assistant.")
    content: str = Field(..., description="Message body.")


class LLMCompletionRequest(BaseModel):
    """A completion request expressed without any vendor-specific type."""

    messages: list[LLMMessage] = Field(..., min_length=1)
    model: str = Field(..., min_length=1, description="Model identifier.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    json_object: bool = Field(
        default=True,
        description=(
            "Ask the endpoint for a JSON object response. Structured outputs are "
            "not uniformly supported across OpenAI-compatible servers, so the "
            "schema is enforced by validation after the call instead."
        ),
    )


class LLMCompletionResult(BaseModel):
    """The completion text plus the metadata a caller may report.

    ``text`` is the raw model output. It is deliberately not stored in the agent
    state: a model can echo its prompt, and the prompt must never reach a
    response or a log.
    """

    text: str
    provider: str = Field(..., description="Provider identifier, for example openai_compatible.")
    model: str | None = Field(default=None, description="Model reported by the endpoint.")
    latency_ms: float = Field(default=0.0, ge=0.0)
    finish_reason: str | None = Field(default=None, description="Endpoint stop reason, when given.")

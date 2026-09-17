"""OpenAI-compatible chat completions provider.

The agent talks to the endpoint over plain HTTP instead of importing a vendor
SDK. Any server that implements ``POST {base_url}/chat/completions`` with the
OpenAI request and response envelope works, which keeps the graph free of
provider coupling.

Three properties are load-bearing:

* the API key is read from a ``SecretStr`` at the moment the header is built and
  is never stored on the instance in clear text;
* error messages are built from the status code and the transport exception's
  class name only. A response body can echo the request and a header carries the
  credential, so neither is ever included;
* a bad status, an unparsable envelope or a missing completion text raise,
  rather than returning an empty result that a caller could mistake for "the
  model planned nothing".
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import httpx
from pydantic import SecretStr

from app.integrations.llm.base import LLMProvider, LLMProviderError, LLMTimeoutError
from app.integrations.llm.models import LLMCompletionRequest, LLMCompletionResult

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/chat/completions"

#: Bound on the provider error text, so a pathological class name cannot turn a
#: log line into a payload.
_MAX_REASON_CHARS = 120


def _reason(exc: BaseException) -> str:
    """Return a short, non-sensitive description of a transport failure.

    Only the exception's class name is used. ``str(exc)`` can embed the request
    URL and, for some transports, the request headers.
    """
    return type(exc).__name__[:_MAX_REASON_CHARS]


def _token_count(value: Any) -> int | None:
    """Return a token count the endpoint reported, or ``None``.

    Only a non-negative integer counts. A missing ``usage`` block, a null, a
    string and a negative number all map to ``None``, because reporting a
    plausible-looking number the endpoint never sent would make an unmeasured
    quantity look measured. ``bool`` is excluded explicitly: ``True`` is an
    ``int`` in Python and would otherwise be counted as one token.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


class OpenAICompatibleProvider(LLMProvider):
    """Completion provider for any OpenAI-compatible chat completions endpoint."""

    provider_id = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | str,
        model: str,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        normalized_base = (base_url or "").strip()
        normalized_model = (model or "").strip()
        secret = api_key if isinstance(api_key, SecretStr) else SecretStr(str(api_key))

        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", normalized_base),
                ("LLM_API_KEY", secret.get_secret_value()),
                ("LLM_MODEL", normalized_model),
            )
            if not value
        ]
        if missing:
            raise LLMProviderError("LLM provider is not configured: missing " + ", ".join(missing))

        self.base_url = normalized_base.rstrip("/")
        self.model = normalized_model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

        self._api_key = secret
        self._client = client
        self._owns_client = client is None

    def describe(self) -> dict[str, str]:
        """Return provider metadata. The key and the headers are absent."""
        return {
            "provider_id": self.provider_id,
            "base_url": self.base_url,
            "model": self.model,
        }

    def close(self) -> None:
        """Close the underlying client when this provider created it."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _http_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def _headers(self) -> dict[str, str]:
        """Build the request headers, reading the secret at the last moment."""
        return {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _payload(self, request: LLMCompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
        }
        if request.json_object:
            # A widely supported hint. It does not enforce the schema; the
            # planner validates the returned object against its Pydantic model.
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResult:
        """Run one chat completion against the configured endpoint."""
        url = f"{self.base_url}{CHAT_COMPLETIONS_PATH}"
        started = perf_counter()

        try:
            response = self._http_client().post(
                url,
                json=self._payload(request),
                headers=self._headers(),
                timeout=request.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM request exceeded {request.timeout_seconds}s ({_reason(exc)})"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"LLM transport failure ({_reason(exc)})") from exc

        latency_ms = round((perf_counter() - started) * 1000, 3)

        if response.status_code >= httpx.codes.BAD_REQUEST:
            # The status is reported; the body is not, because it can echo the
            # request and, on some gateways, the credential.
            raise LLMProviderError(f"LLM endpoint returned HTTP {response.status_code}")

        envelope = self._envelope(response)
        prompt_tokens, completion_tokens = self._extract_usage(envelope)
        return LLMCompletionResult(
            text=self._extract_text(envelope),
            provider=self.provider_id,
            model=self.model,
            latency_ms=latency_ms,
            finish_reason=self._extract_finish_reason(envelope),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _envelope(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMProviderError("LLM endpoint returned a non-JSON envelope") from exc
        if not isinstance(payload, dict):
            raise LLMProviderError("LLM endpoint returned an unexpected envelope shape")
        return payload

    @staticmethod
    def _first_choice(envelope: dict[str, Any]) -> dict[str, Any]:
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMProviderError("LLM endpoint returned no choices")
        return choices[0]

    def _extract_text(self, envelope: dict[str, Any]) -> str:
        message = self._first_choice(envelope).get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM endpoint returned no completion text")
        return content

    def _extract_finish_reason(self, envelope: dict[str, Any]) -> str | None:
        reason = self._first_choice(envelope).get("finish_reason")
        return str(reason) if reason else None

    @staticmethod
    def _extract_usage(envelope: dict[str, Any]) -> tuple[int | None, int | None]:
        """Return the ``(prompt, completion)`` token counts the endpoint reported.

        The ``usage`` block is optional in the OpenAI-compatible envelope and some
        servers omit it. Absent usage yields ``(None, None)`` rather than zero, so
        a downstream metric can distinguish "not reported" from "reported zero".
        """
        usage = envelope.get("usage")
        if not isinstance(usage, dict):
            return None, None
        return (
            _token_count(usage.get("prompt_tokens")),
            _token_count(usage.get("completion_tokens")),
        )

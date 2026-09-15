"""Tests for the LLM provider layer.

The provider is the one place the service talks to the network, so these tests
pin its boundary:

1. a well-formed OpenAI-compatible envelope becomes a completion result;
2. every malformed answer raises rather than returning an empty completion that a
   planner could mistake for "the model planned nothing";
3. error text is built from the status code and the transport class name only, so
   a response body that echoes the credential cannot leak into a log;
4. the API key never appears in the provider's own metadata or repr;
5. the factory refuses to invent an endpoint when the configuration is empty.

No test opens a socket: the HTTP client is replaced with an ``httpx.MockTransport``.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.integrations.llm import (
    LLMCompletionRequest,
    LLMMessage,
    LLMProviderError,
    LLMTimeoutError,
    OpenAICompatibleProvider,
    build_provider,
    is_llm_configured,
)
from tests.fake_llm import FAKE_API_KEY, FAKE_BASE_URL, FAKE_MODEL, fake_secret

PLANNER_BODY = json.dumps(
    {
        "intent": "alarm_diagnosis",
        "tool_calls": [
            {"tool_name": "query_alarm_code", "arguments": {"alarm_code": "F0045"}, "reason": None}
        ],
    }
)


def _settings(**overrides: object) -> Settings:
    """Build settings for a configured provider, with per-test overrides."""
    values: dict[str, object] = {
        "llm_base_url": FAKE_BASE_URL,
        "llm_api_key": fake_secret(),
        "llm_model": FAKE_MODEL,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _client(handler: httpx.MockTransport) -> httpx.Client:
    """Wrap a mock handler in a client the provider will not close."""
    return httpx.Client(transport=handler)


def _ok_response(_: httpx.Request) -> httpx.Response:
    """Return a minimal well-formed chat completions envelope."""
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": PLANNER_BODY},
                    "finish_reason": "stop",
                }
            ]
        },
    )


def _provider(client: httpx.Client, **overrides: object) -> OpenAICompatibleProvider:
    """Build a provider wired to ``client``."""
    values: dict[str, object] = {
        "base_url": FAKE_BASE_URL,
        "api_key": fake_secret(),
        "model": FAKE_MODEL,
        "client": client,
    }
    values.update(overrides)
    return OpenAICompatibleProvider(**values)  # type: ignore[arg-type]


def _request() -> LLMCompletionRequest:
    """Build a one-message completion request."""
    return LLMCompletionRequest(
        messages=[LLMMessage(role="user", content="包装线PLC报警F0045怎么办")],
        model=FAKE_MODEL,
        timeout_seconds=5.0,
    )


# --------------------------------------------------------------------------- #
# 1. A well-formed exchange
# --------------------------------------------------------------------------- #


def test_completion_returns_text_and_metadata() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok_response(request)

    provider = _provider(_client(httpx.MockTransport(handler)))
    result = provider.complete(_request())

    assert result.text == PLANNER_BODY
    assert result.provider == "openai_compatible"
    assert result.model == FAKE_MODEL
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0

    # The endpoint is called at the documented path and the key is supplied as a
    # bearer credential rather than an ad-hoc header.
    assert len(seen) == 1
    assert str(seen[0].url) == f"{FAKE_BASE_URL}/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {FAKE_API_KEY}"


def test_json_object_hint_is_requested_when_asked() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return _ok_response(request)

    provider = _provider(_client(httpx.MockTransport(handler)))
    provider.complete(_request())

    assert captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == FAKE_MODEL


# --------------------------------------------------------------------------- #
# 2. Malformed answers raise instead of returning an empty completion
# --------------------------------------------------------------------------- #


def test_non_success_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text=f"upstream error token={FAKE_API_KEY}")

    provider = _provider(_client(httpx.MockTransport(handler)))

    with pytest.raises(LLMProviderError) as excinfo:
        provider.complete(_request())

    message = str(excinfo.value)
    assert "502" in message
    # The body can echo the request, so it is never part of the error text.
    assert FAKE_API_KEY not in message


def test_non_json_envelope_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    provider = _provider(_client(httpx.MockTransport(handler)))

    with pytest.raises(LLMProviderError, match="non-JSON"):
        provider.complete(_request())


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "   "}}]},
        {"choices": [{"message": {}}]},
    ],
)
def test_missing_completion_text_raises(envelope: dict[str, object]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=envelope)

    provider = _provider(_client(httpx.MockTransport(handler)))

    with pytest.raises(LLMProviderError):
        provider.complete(_request())


def test_timeout_raises_a_distinct_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    provider = _provider(_client(httpx.MockTransport(handler)))

    with pytest.raises(LLMTimeoutError) as excinfo:
        provider.complete(_request())

    assert "Timeout" in str(excinfo.value)
    # A timeout is classified apart from a generic provider failure so the caller
    # can tell a deadline from a misconfiguration.
    assert type(excinfo.value) is not LLMProviderError


def test_transport_failure_raises_a_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(_client(httpx.MockTransport(handler)))

    with pytest.raises(LLMProviderError) as excinfo:
        provider.complete(_request())

    assert "ConnectError" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 3. The credential never appears in metadata
# --------------------------------------------------------------------------- #


def test_describe_never_contains_the_key() -> None:
    provider = _provider(_client(httpx.MockTransport(_ok_response)))
    described = provider.describe()

    assert described["provider_id"] == "openai_compatible"
    assert described["base_url"] == FAKE_BASE_URL
    assert FAKE_API_KEY not in str(described)


def test_repr_never_contains_the_key() -> None:
    provider = _provider(_client(httpx.MockTransport(_ok_response)))

    assert FAKE_API_KEY not in repr(provider)
    assert FAKE_API_KEY not in str(vars(provider))


# --------------------------------------------------------------------------- #
# 4. Configuration gate
# --------------------------------------------------------------------------- #


def test_incomplete_configuration_raises_from_the_constructor() -> None:
    with pytest.raises(LLMProviderError, match="LLM_API_KEY"):
        OpenAICompatibleProvider(base_url=FAKE_BASE_URL, api_key="", model=FAKE_MODEL)


def test_factory_refuses_to_invent_an_endpoint() -> None:
    with pytest.raises(LLMProviderError, match="LLM_BASE_URL"):
        build_provider(Settings(llm_base_url="", llm_api_key=fake_secret(), llm_model=FAKE_MODEL))

    with pytest.raises(LLMProviderError, match="LLM_BASE_URL"):
        build_provider(Settings(llm_base_url="", llm_api_key=SecretStr(""), llm_model=""))


def test_factory_builds_a_configured_provider() -> None:
    provider = build_provider(_settings())

    assert provider.provider_id == "openai_compatible"
    assert provider.describe()["model"] == FAKE_MODEL


def test_is_llm_configured_requires_all_three_values() -> None:
    assert is_llm_configured(_settings()) is True
    empty = Settings(llm_base_url="", llm_api_key=SecretStr(""), llm_model="")
    assert is_llm_configured(empty) is False
    assert (
        is_llm_configured(
            Settings(llm_base_url=FAKE_BASE_URL, llm_api_key=fake_secret(), llm_model="")
        )
        is False
    )

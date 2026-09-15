"""Config-driven LLM provider selection.

The factory reads :class:`~app.config.Settings` and returns the provider that the
current configuration supports. With nothing configured it raises instead of
inventing an endpoint, so a missing key surfaces as an explicit provider error
rather than as a request to an unknown host.

The exception messages name the missing environment variables. They never echo a
value.
"""

from functools import lru_cache

from app.config import Settings, get_settings
from app.integrations.llm.base import LLMProvider, LLMProviderError
from app.integrations.llm.openai_compatible import OpenAICompatibleProvider

SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai_compatible",)


def build_provider(settings: Settings | None = None) -> LLMProvider:
    """Build the provider the configuration describes.

    Raises:
        LLMProviderError: The configuration is incomplete. The message names the
            missing variables so a misconfiguration is actionable.
    """
    resolved = settings or get_settings()
    if not resolved.llm_base_url.strip():
        raise LLMProviderError(
            "LLM provider is not configured: missing LLM_BASE_URL, LLM_API_KEY, LLM_MODEL"
        )
    return OpenAICompatibleProvider(
        base_url=resolved.llm_base_url,
        api_key=resolved.llm_api_key,
        model=resolved.llm_model,
        timeout_seconds=resolved.llm_timeout_seconds,
        temperature=resolved.llm_temperature,
    )


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Return a cached provider instance built from the current settings."""
    return build_provider()


def reset_llm_provider() -> None:
    """Drop the cached provider. Used by tests and by configuration reloads."""
    get_llm_provider.cache_clear()


def is_llm_configured(settings: Settings | None = None) -> bool:
    """Return ``True`` when a real provider can be built from the configuration.

    Used by the smoke-test gate: with no configuration the gate reports
    ``REAL_LLM_GATE_NOT_RUN`` instead of attempting a call.
    """
    resolved = settings or get_settings()
    return bool(
        resolved.llm_base_url.strip()
        and resolved.llm_api_key.get_secret_value().strip()
        and resolved.llm_model.strip()
    )

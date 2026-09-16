"""Central application settings loaded from environment variables and .env."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Planner selection. ``rule`` is the frozen deterministic baseline, ``llm``
#: refuses to degrade, ``auto`` prefers the LLM and falls back to the rule
#: planner when the LLM cannot produce a plan.
PlannerMode = Literal["rule", "llm", "auto"]


class Settings(BaseSettings):
    """Runtime configuration for the service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Industrial Maintenance Agent"
    app_version: str = "0.8.1"
    environment: str = "development"
    debug: bool = False
    # Verbosity for the ``app`` logger namespace. A mistyped value fails loudly
    # at startup rather than silently logging at the wrong level.
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Persistence
    database_url: str = "sqlite:///./data/industrial_maintenance.db"
    auto_create_tables: bool = True

    # Planner selection. ``rule`` keeps the V0.4 deterministic planner, which is
    # frozen: the LLM planner is additive and never replaces it.
    planner_mode: PlannerMode = "rule"

    # LLM provider for the optional LLM planner. The agent talks to an
    # OpenAI-compatible chat completions endpoint over HTTP, so no vendor SDK is
    # involved and the graph never imports a provider library.
    # ``llm_base_url`` is the API root, for example https://host/v1. No default
    # host is assumed: with no configuration the LLM planner reports a provider
    # error instead of guessing an endpoint.
    llm_base_url: str = ""
    # SecretStr keeps the value out of logs, tracebacks and repr output. Call
    # ``get_secret_value()`` only at the point the header is built.
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout_seconds: float = 30.0
    llm_temperature: float = 0.0

    # RAG integration (Industrial Knowledge RAG, consumed read-only)
    # ``local`` runs the RAG retrieval engine in-process and is the only
    # LLM-free path. ``http`` calls the RAG FastAPI service instead.
    rag_provider: str = "local"
    # Required when rag_provider == "local". No default path is assumed.
    rag_repo_root: str = ""
    # RAG backend module: "light" (light_rag_core) or "full" (rag_core).
    rag_backend: str = "light"
    rag_knowledge_base_id: str = "default"
    rag_retrieval_mode: str = "hybrid"
    rag_top_k: int = 4
    # Required when rag_provider == "http". No default host or port is assumed.
    rag_base_url: str = ""
    rag_timeout_seconds: float = 10.0
    # POST /ask always runs the RAG service's own answer generator, which needs
    # a provider key on the RAG side. The agent discards the generated answer.
    rag_http_model_provider: str = "DeepSeek"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

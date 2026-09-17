"""Central application settings loaded from environment variables and .env."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Planner selection. ``rule`` is the frozen deterministic baseline, ``llm``
#: refuses to degrade, ``auto`` prefers the LLM and falls back to the rule
#: planner when the LLM cannot produce a plan.
PlannerMode = Literal["rule", "llm", "auto"]

#: Log rendering. ``text`` is the human-readable line this service has always
#: emitted; ``json`` is one object per line for a log pipeline.
LogFormat = Literal["text", "json"]


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
    app_version: str = "0.9.0"
    environment: str = "development"
    debug: bool = False
    # Verbosity for the ``app`` logger namespace. A mistyped value fails loudly
    # at startup rather than silently logging at the wrong level.
    log_level: str = "INFO"
    # Rendering for every record the ``app`` namespace emits. ``text`` is the
    # historical single-line format and stays the default, so enabling structured
    # logging is a deliberate act. See app/observability/logging.py.
    log_format: LogFormat = "text"

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

    # External device data sources (optional, read-only)
    # A local copy of the UCI MetroPT-3 CSV, served under the METRO-APU-001
    # identifier. Empty by default, which leaves the external source absent and
    # the seeded SQLite devices answering exactly as before. The dataset is about
    # 208 MiB, is never bundled with this repository, and is never downloaded by
    # the running service: only the path is read. See docs/data/metropt3.md.
    metropt3_csv_path: str = ""

    # --- Observability ---
    # Prometheus metrics. When false, GET /metrics answers 404 and no series is
    # recorded. Logging is unaffected: the log contract predates this switch.
    metrics_enabled: bool = True

    # OpenTelemetry tracing. Off by default, and off means nothing is imported
    # and no socket is opened. An OTLP endpoint has no default on purpose:
    # pointing at a local collector would turn an optional feature into a
    # runtime dependency on a process the operator may not have started.
    otel_enabled: bool = False
    otel_service_name: str = "industrial-maintenance-agent"
    otel_exporter_otlp_endpoint: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

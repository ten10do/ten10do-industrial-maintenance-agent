"""Measure the latency cost of the observability layer under a deterministic workload.

One process per configuration, because ``METRICS_ENABLED`` and ``OTEL_ENABLED`` are
read from settings and settings are cached: switching them inside one interpreter
would measure a process that is half-reconfigured.

    python -m scripts.observability_overhead --config off
    python -m scripts.observability_overhead --config metrics
    python -m scripts.observability_overhead --config metrics+tracing

The workload is the frozen 49-case dataset, driven through the same call
boundaries ``evaluation.runner`` uses:

* **planning** is ``route_query`` followed by ``dispatch_plan``;
* **total** adds ``execute_tools``, ``retrieve_context`` and ``synthesize``.

Reusing those boundaries is deliberate. A second, slightly different timer would
report a different quantity under the same name, and the two would not be
comparable with the frozen benchmark.

Two things are held constant across configurations so the difference isolates the
observability layer rather than the environment:

* the retrieval provider is an in-process deterministic double, so no
  configuration pays for the real retrieval engine and no disk cache warms up
  during the run;
* logging is configured identically, at the same level and in the same format.
  Log emission is therefore present in every configuration, including ``off``,
  because it is not gated by ``METRICS_ENABLED``. The reported deltas measure
  metric recording and span creation, and the emitted-record count is reported
  alongside so a reader can confirm log volume did not change.

``--log-level`` exists for one further question. This release added per-stage log
events, and the frozen baseline predates them, so an absolute planning figure
measured here is not directly comparable with the published one. Running with
``--log-level WARNING`` silences those events and isolates their cost, which is
the only reason the two figures differ. It is not part of the metric-cost
comparison, which uses the shipped ``INFO`` default throughout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

CONFIGS: dict[str, dict[str, str]] = {
    "off": {"METRICS_ENABLED": "false", "OTEL_ENABLED": "false"},
    "metrics": {"METRICS_ENABLED": "true", "OTEL_ENABLED": "false"},
    "metrics+tracing": {"METRICS_ENABLED": "true", "OTEL_ENABLED": "true"},
}

#: Applied before ``app`` is imported so no ambient configuration can influence a
#: measurement. The rule planner is the only deterministic baseline; the database
#: is a private in-memory one seeded like the test suite's.
BASE_ENV: dict[str, str] = {
    "PLANNER_MODE": "rule",
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "text",
    "AUTO_CREATE_TABLES": "false",
    "APP_VERSION": "0.9.0",
}


class _CountingHandler(logging.Handler):
    """Count records without formatting them, to expose log volume per config."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


class _DeterministicProvider:
    """Retrieval double with a fixed hit list and no I/O."""

    provider_id = "local"

    def search(self, query: str, top_k: int = 4) -> Any:
        from app.integrations.rag.models import RAGSearchHit, RAGSearchResponse
        from app.integrations.rag.scores import VECTOR_COSINE_DISTANCE

        hits = [
            RAGSearchHit(
                content="Deterministic fixture fragment for the overhead measurement.",
                document="overhead_fixture.pdf",
                page=1,
                section="Fixture",
                chunk_id=f"chunk-{index}",
                score=0.5 + index * 0.01,
                score_semantics=VECTOR_COSINE_DISTANCE,
                higher_is_better=False,
            )
            for index in range(2)
        ]
        return RAGSearchResponse(query=query, hits=hits, retrieval_mode="hybrid")


def _prepare_environment(config: str, log_level: str) -> None:
    """Apply the configuration to ``os.environ`` before the application loads."""
    for key, value in BASE_ENV.items():
        os.environ[key] = value
    os.environ["LOG_LEVEL"] = log_level
    for key, value in CONFIGS[config].items():
        os.environ[key] = value


def _install_hermetic_database() -> None:
    """Point the session factory at a seeded in-memory database."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import session as db_session
    from app.database.init_db import init_db

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    factory = sessionmaker(bind=engine, future=True)
    init_db(seed=True, reset=True, engine=engine, session_factory=factory)
    db_session.SessionLocal = factory  # type: ignore[assignment]


def _measure(repeats: int) -> dict[str, Any]:
    """Run the workload and return the raw samples for one configuration."""
    from app.agent.graph import (
        execute_tools,
        plan_actions,
        retrieve_context,
        route_query,
        synthesize,
    )
    from app.agent.planners.dispatch import dispatch_plan
    from app.config import get_settings
    from evaluation.dataset import load_dataset

    settings = get_settings()
    dataset = load_dataset()

    planning_samples: list[float] = []
    total_samples: list[float] = []

    def run_case(query: str) -> None:
        state: dict[str, Any] = {"query": query}

        started = perf_counter()
        state.update(route_query(state))  # type: ignore[arg-type]
        state.update(
            dispatch_plan(
                state,  # type: ignore[arg-type]
                rule_planner=plan_actions,
                settings=settings,
            )
        )
        planning_ms = round((perf_counter() - started) * 1000, 3)

        state.update(execute_tools(state))  # type: ignore[arg-type]
        state.update(retrieve_context(state))  # type: ignore[arg-type]
        state.update(synthesize(state))  # type: ignore[arg-type]
        total_ms = round((perf_counter() - started) * 1000, 3)

        planning_samples.append(planning_ms)
        total_samples.append(total_ms)

    queries = [case.query for case in dataset.cases]

    # Warm-up: the first pass pays for lazy imports, the graph singleton and the
    # parser's lazy catalog read. None of that is observability cost.
    for query in queries:
        run_case(query)
    planning_samples.clear()
    total_samples.clear()

    for _ in range(repeats):
        for query in queries:
            run_case(query)

    return {
        "metrics_enabled": settings.metrics_enabled,
        "otel_enabled": settings.otel_enabled,
        "log_format": settings.log_format,
        "log_level": settings.log_level,
        "case_count": len(queries),
        "repeats": repeats,
        "samples": len(total_samples),
        "planning_ms": planning_samples,
        "total_ms": total_samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Observability overhead measurement.")
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help=(
            "Level for the app logger. INFO is the shipped default and the setting "
            "used for the metric-cost comparison. WARNING silences the per-stage "
            "events and is used only to attribute the cost of the logging this "
            "release added, which the frozen baseline predates."
        ),
    )
    args = parser.parse_args(argv)

    _prepare_environment(args.config, args.log_level)

    # The package is imported only now, so settings see the environment applied
    # above and never the developer's ``.env``.
    from app.config import Settings
    from app.logging_config import configure_logging
    from app.observability import configure_observability

    Settings.model_config["env_file"] = None

    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    _install_hermetic_database()

    # Route the tool layer to the deterministic provider.
    import app.integrations.rag.factory as rag_factory

    rag_factory.get_rag_provider = lambda: _DeterministicProvider()  # type: ignore[assignment]

    handler = _CountingHandler()
    logging.getLogger("app").addHandler(handler)

    configure_logging(settings.log_level, settings.log_format)
    configure_observability(settings)

    from app.observability import tracing

    result = _measure(args.repeats)
    result["config"] = args.config
    result["tracing_active"] = tracing.is_enabled()
    result["log_records"] = handler.count
    # Divided by every request that ran, warm-up included, so the figure is a
    # property of the pipeline rather than of the repeat count.
    total_requests = result["case_count"] * (result["repeats"] + 1)
    result["log_records_per_request"] = round(handler.count / total_requests, 3)
    result["total_requests"] = total_requests

    def summary(values: list[float]) -> dict[str, float]:
        """Summarize a sample with the evaluation harness's own definitions.

        Reusing ``evaluation.metrics`` rather than ``statistics`` matters for the
        tail: the harness interpolates linearly between order statistics, and a
        nearest-rank p95 would report a different number for the same sample.
        """
        from evaluation.metrics import median, percentile

        computed = {
            "mean": round(statistics.fmean(values), 4),
            "median": median(values),
            "p95": percentile(values, 95),
            "min": min(values),
            "max": max(values),
        }
        return {key: float(value) for key, value in computed.items() if value is not None}

    result["planning_summary"] = summary(result["planning_ms"])
    result["total_summary"] = summary(result["total_ms"])

    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")

    headline = {k: v for k, v in result.items() if k not in ("planning_ms", "total_ms")}
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

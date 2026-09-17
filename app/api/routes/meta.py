"""Meta endpoints: service identity, health probes and the metrics scrape.

These carry no business logic. They exist so an operator or a supervisor can
tell whether the process is up, and so a Prometheus server can read the metric
registry, without invoking the agent.
"""

from fastapi import APIRouter, HTTPException, Response, status

from app.config import get_settings
from app.observability.metrics import CONTENT_TYPE_LATEST, get_metrics

settings = get_settings()

router = APIRouter(tags=["meta"])


@router.get("/", summary="Service identity probe")
async def root() -> dict:
    """Report service identity."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "ok",
    }


@router.get("/health", summary="Liveness / readiness probe")
async def health() -> dict:
    """Report liveness."""
    return {"status": "healthy", "environment": settings.environment}


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=Response,
    responses={
        200: {"description": "The registry in the Prometheus exposition format."},
        404: {"description": "Metrics are disabled by configuration."},
    },
)
async def metrics() -> Response:
    """Render the metric registry.

    Disabled metrics answer 404 rather than an empty body. An empty 200 would be
    indistinguishable from a registry that has recorded nothing yet, and a
    scraper would keep polling a target that is deliberately inactive.

    Reading the setting per request, rather than capturing it at import time,
    keeps the endpoint's behaviour consistent with the recording helpers, which
    also re-read it: a process can never be half-on, exposing an endpoint while
    recording nothing, or the reverse.
    """
    if not get_settings().metrics_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics are disabled. Set METRICS_ENABLED=true to expose them.",
        )

    return Response(content=get_metrics().render(), media_type=CONTENT_TYPE_LATEST)

"""Meta endpoints: service identity and health probes.

These carry no business logic. They exist so an operator or a supervisor can
tell whether the process is up without invoking the agent.
"""

from fastapi import APIRouter

from app.config import get_settings

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

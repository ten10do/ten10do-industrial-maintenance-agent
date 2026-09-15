"""FastAPI service entry point.

Composition only: settings, lifespan, routers and error handlers. The HTTP
surface lives in ``app/api`` and the workflow lives in ``app/agent``; neither is
constructed or reimplemented here.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import agent_router, meta_router
from app.config import get_settings
from app.database.init_db import init_db
from app.logging_config import configure_logging

settings = get_settings()

configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the database schema exists before serving requests."""
    if settings.auto_create_tables:
        init_db(seed=False)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Agent service for industrial equipment maintenance assistance.",
    lifespan=lifespan,
)

app.include_router(meta_router)
app.include_router(agent_router)
register_exception_handlers(app)


def run() -> None:
    """Run the development server via uvicorn."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()

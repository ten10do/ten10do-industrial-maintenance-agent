"""Structured error handling for the HTTP layer.

One rule governs everything here: a response body never carries internal
detail. A caller receives a stable machine-readable ``error`` code, a generic
message, and the ``request_id`` needed to correlate the failure with the server
log. The exception, its message, its stack and the configuration stay inside
the process.

Validation failures are left to FastAPI. Its 422 body already names the
offending field and carries no internal state, so reshaping it would only make
the contract less predictable.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.schemas.agent import AgentErrorResponse
from app.services.agent_service import AGENT_INVOCATION_FAILED, AgentInvocationError

logger = logging.getLogger(__name__)

#: Caller-facing text. It names no exception, no module and no configuration.
AGENT_FAILURE_MESSAGE = "Agent 执行失败，请稍后重试；如问题持续，请提供 request_id 以便排查。"


async def agent_invocation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return the structured failure body for a failed invocation.

    The parameter is typed as ``Exception`` because that is what Starlette's
    handler signature accepts. The handler is registered for
    :class:`AgentInvocationError` only; the ``isinstance`` narrowing keeps the
    signature sound and makes the function total rather than relying on the
    registration to guarantee the type.
    """
    error = exc if isinstance(exc, AgentInvocationError) else None
    body = AgentErrorResponse(
        request_id=error.request_id if error else "",
        error=error.code if error else AGENT_INVOCATION_FAILED,
        message=AGENT_FAILURE_MESSAGE,
        latency_ms=error.latency_ms if error else 0.0,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=body.model_dump(mode="json"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the service's exception handlers to the application."""
    app.add_exception_handler(AgentInvocationError, agent_invocation_error_handler)

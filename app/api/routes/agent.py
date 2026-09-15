"""Agent invocation route.

This module is the only place where HTTP meets the agent, and it stays thin:
validate the body, delegate to :class:`AgentService`, return the response
schema. The compiled graph is never imported here, so the workflow cannot be
reached over HTTP by any path other than the documented one.
"""

from fastapi import APIRouter, Depends, status

from app.api.schemas.agent import AgentErrorResponse, AgentInvokeRequest, AgentInvokeResponse
from app.services import AgentService

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_service() -> AgentService:
    """Return the invocation service.

    Declared as a dependency so a test can substitute the graph callable
    through ``app.dependency_overrides`` instead of patching module internals.
    Constructing the service is cheap: it holds only an optional callable.
    """
    return AgentService()


@router.post(
    "/invoke",
    response_model=AgentInvokeResponse,
    status_code=status.HTTP_200_OK,
    summary="Invoke the maintenance agent",
    response_description="Parsed intent, executed tools, answer, evidence and latency.",
    responses={
        # Spelled numerically on purpose: the symbolic constant is deprecated in
        # recent Starlette releases under a new name, and the status code itself
        # has not changed.
        422: {
            "description": "Request body failed validation, such as a blank or oversized query.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": AgentErrorResponse,
            "description": (
                "The pipeline failed. The body carries a stable error code, a "
                "generic message and the request_id; no traceback is returned."
            ),
        },
    },
)
def invoke_agent(
    payload: AgentInvokeRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentInvokeResponse:
    """Run one agent invocation.

    Declared with ``def`` rather than ``async def`` on purpose. The pipeline is
    synchronous and CPU-bound, so Starlette dispatches it to a worker thread
    instead of blocking the event loop for the duration of the retrieval call.

    A pipeline failure raises :class:`AgentInvocationError`, which the
    application-level handler in :mod:`app.api.errors` converts into the 500
    body documented above.
    """
    return service.invoke(payload.query, debug=payload.debug)

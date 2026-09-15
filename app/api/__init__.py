"""HTTP layer: transport schemas, routers and structured error handling.

This package is deliberately inert on import. It re-exports nothing, so that
``app.api.schemas`` can be imported by the service layer without dragging the
FastAPI routers along. Wiring happens in ``app.main``.
"""

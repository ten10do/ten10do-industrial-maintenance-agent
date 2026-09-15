"""Logging configuration for the application.

A library must not configure logging; an application must. ``app.main`` is the
application entry point, so the configuration is applied there, once.

Two properties are deliberate:

* the handler is attached to the ``app`` package logger rather than the root
  logger. Without it, an unconfigured logger inherits the root level
  ``WARNING``, so this service's ``INFO`` records would never be emitted and a
  successful invocation would leave no trace. Keeping the change inside the
  ``app`` namespace leaves third-party verbosity untouched.
* ``propagate`` stays at its default ``True``. A host process that attaches its
  own root handler, pytest's ``caplog`` for example, still receives these
  records. Setting it to ``False`` would silence the tests that pin the log
  contract.
"""

from __future__ import annotations

import logging

APP_LOGGER_NAME = "app"

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Attach a stream handler to the ``app`` logger namespace.

    Idempotent: a later call updates the level without adding a second handler,
    so importing the application twice does not duplicate output.

    Raises:
        ValueError: The level name is not one Python's logging module knows.
            Failing loudly is preferable to silently logging at the wrong
            verbosity.
    """
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(f"unknown log level: {level}")

    logger = logging.getLogger(APP_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(resolved)

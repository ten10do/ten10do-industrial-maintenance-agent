"""Logging configuration for the application.

A library must not configure logging; an application must. ``app.main`` is the
application entry point, so the configuration is applied there, once.

Three properties are deliberate:

* the handler is attached to the ``app`` package logger rather than the root
  logger. Without it, an unconfigured logger inherits the root level
  ``WARNING``, so this service's ``INFO`` records would never be emitted and a
  successful invocation would leave no trace. Keeping the change inside the
  ``app`` namespace leaves third-party verbosity untouched.
* ``propagate`` stays at its default ``True``. A host process that attaches its
  own root handler, pytest's ``caplog`` for example, still receives these
  records. Setting it to ``False`` would silence the tests that pin the log
  contract.
* the renderer comes from :mod:`app.observability.logging`, where the redaction
  policy lives, so the format and the fields that may appear in it are declared
  in one place. ``LOG_FORMAT=text`` keeps the historical line byte-for-byte;
  ``LOG_FORMAT=json`` emits one object per line.

The structured fields themselves travel on the ``LogRecord``, not through this
handler, so a host that replaces the handler still sees them.
"""

from __future__ import annotations

import logging

from app.observability.logging import TEXT_LOG_FORMAT, build_formatter

APP_LOGGER_NAME = "app"

#: Historical name for the text format. Re-exported so an existing import keeps
#: resolving; the value now lives with the formatter it belongs to.
LOG_FORMAT = TEXT_LOG_FORMAT


def configure_logging(level: str = "INFO", log_format: str = "text") -> None:
    """Attach a stream handler to the ``app`` logger namespace.

    Idempotent: a later call updates the level and the formatter without adding a
    second handler, so importing the application twice does not duplicate output
    and reconfiguring between tests does not accumulate handlers.

    Args:
        level: A level name Python's logging module knows.
        log_format: ``text`` or ``json``. An unrecognised value falls back to
            text, which is the format whose absence is immediately visible.

    Raises:
        ValueError: The level name is unknown. Failing loudly is preferable to
            silently logging at the wrong verbosity.
    """
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(f"unknown log level: {level}")

    logger = logging.getLogger(APP_LOGGER_NAME)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    for handler in logger.handlers:
        handler.setFormatter(build_formatter(log_format))
    logger.setLevel(resolved)

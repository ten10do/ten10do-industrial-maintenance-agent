"""Suite-wide test isolation.

:class:`~app.config.Settings` reads a developer's local ``.env`` file. A local
``.env`` is a personal override, so it must not change what the suite asserts:
the tests pin the code defaults and inject their own environment variables
explicitly. This autouse fixture disables the ``.env`` source for every test,
which makes the suite deterministic whether or not a ``.env`` exists on the
machine.

``os.environ`` is deliberately left untouched, so tests that set variables with
``monkeypatch.setenv`` keep working exactly as before.
"""

from collections.abc import Iterator

import pytest

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _ignore_local_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make ``Settings()`` ignore the ambient ``.env`` for the duration of a test."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

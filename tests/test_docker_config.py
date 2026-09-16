"""Contracts for the two Docker build shapes.

The repository ships two containers out of one Dockerfile: a standalone agent
that must stay free of the optional RAG dependency set, and a RAG-enabled agent
that must install it. A regression here is invisible to every other test in the
suite and only shows up at run time, as a manual query that reports the manual
tool as unavailable while the container itself looks healthy.

``tests/`` asserts code matches its specification, so the specification is
pinned here directly. Docker itself is not invoked: the CI job is hermetic and
has no daemon, so these tests read the build files instead.

The Compose file is parsed as YAML rather than matched as text, so reflowing a
comment or adding a service does not break the tests, and a failure names the
exact field that regressed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
RAG_REQUIREMENTS = PROJECT_ROOT / "requirements-rag-local.txt"

STANDALONE_SERVICE = "agent"
RAG_SERVICE = "agent-rag"
RAG_PROFILE = "rag"

_DRIVE_LETTER_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _dockerfile_instructions() -> list[str]:
    """Return the Dockerfile's instructions, comments dropped and lines folded.

    A single instruction may span several physical lines through a trailing
    backslash. Folding them keeps an assertion about one instruction from
    depending on how it happens to be wrapped.
    """
    instructions: list[str] = []
    pending = ""
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}" if pending else line
        if pending.endswith("\\"):
            pending = pending[:-1]
            continue
        instructions.append(" ".join(pending.split()))
        pending = ""
    if pending:
        instructions.append(" ".join(pending.split()))
    return instructions


def _ignore_patterns() -> list[str]:
    """Return the active patterns in ``.dockerignore``."""
    patterns: list[str] = []
    for raw in DOCKERIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    """The Compose file, parsed once per module."""
    loaded: dict[str, Any] = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def services(compose: dict[str, Any]) -> dict[str, Any]:
    """The service map from the Compose file."""
    parsed: dict[str, Any] = compose["services"]
    return parsed


class TestBuildShapes:
    """The Dockerfile must offer both shapes and default to the small one."""

    def test_optional_rag_requirements_file_exists(self) -> None:
        assert RAG_REQUIREMENTS.is_file(), "the RAG shape installs requirements-rag-local.txt"

    def test_rag_install_is_opt_in(self) -> None:
        """The switch defaults to off, so a plain build stays standalone."""
        declarations = [
            i for i in _dockerfile_instructions() if i.startswith("ARG INSTALL_RAG_DEPS")
        ]
        assert declarations == ["ARG INSTALL_RAG_DEPS=0"], declarations

    def test_both_requirement_files_reach_the_image(self) -> None:
        """The optional file must be copied, or the conditional install cannot read it."""
        copied = [i for i in _dockerfile_instructions() if i.startswith("COPY")]
        assert any("requirements.txt" in i for i in copied), copied
        assert any("requirements-rag-local.txt" in i for i in copied), copied

    def test_core_requirements_are_installed_unconditionally(self) -> None:
        install = [i for i in _dockerfile_instructions() if "pip install" in i]
        assert install, "the Dockerfile installs dependencies"
        assert any("requirements.txt" in i for i in install), install

    def test_rag_requirements_install_is_guarded_by_the_argument(self) -> None:
        """The optional set is installed only in the RAG shape.

        The guard is the point of this file: an unguarded install would push
        numpy and scikit-learn into every image.
        """
        guarded = [
            i
            for i in _dockerfile_instructions()
            if "requirements-rag-local.txt" in i and "install" in i
        ]
        assert guarded, "the RAG dependency set is installed somewhere"
        for instruction in guarded:
            assert "INSTALL_RAG_DEPS" in instruction, instruction

    def test_rag_requirements_stay_in_the_build_context(self) -> None:
        """An ignore rule that drops the file would break the RAG build silently."""
        patterns = _ignore_patterns()
        assert "requirements-rag-local.txt" not in patterns, patterns
        assert "requirements*.txt" not in patterns, patterns

    def test_dockerfile_never_copies_local_configuration(self) -> None:
        """The image must not carry a .env, and must not bake the RAG checkout in."""
        copied = [i for i in _dockerfile_instructions() if i.startswith("COPY")]
        for instruction in copied:
            assert ".env" not in instruction, instruction
        joined = " ".join(copied)
        assert "industrial-knowledge-rag" not in joined, joined


class TestComposeShapes:
    """Compose must expose the RAG shape behind a discoverable profile."""

    def test_default_up_is_standalone_only(self, services: dict[str, Any]) -> None:
        """A plain ``up`` must ignore the RAG service."""
        assert STANDALONE_SERVICE in services, sorted(services)
        assert not services[STANDALONE_SERVICE].get(
            "profiles"
        ), "the default service has no profile"

    def test_rag_service_is_behind_the_rag_profile(self, services: dict[str, Any]) -> None:
        assert RAG_SERVICE in services, sorted(services)
        assert services[RAG_SERVICE].get("profiles") == [RAG_PROFILE]

    def test_the_two_shapes_differ_only_by_the_build_argument(
        self, services: dict[str, Any]
    ) -> None:
        standalone_build = services[STANDALONE_SERVICE]["build"]
        rag_build = services[RAG_SERVICE]["build"]
        assert standalone_build["args"]["INSTALL_RAG_DEPS"] == "0"
        assert rag_build["args"]["INSTALL_RAG_DEPS"] == "1"
        assert standalone_build["context"] == rag_build["context"]
        assert standalone_build["dockerfile"] == rag_build["dockerfile"]

    def test_rag_service_receives_the_full_rag_configuration(
        self, services: dict[str, Any]
    ) -> None:
        """Mounting the checkout is not enough; the provider must also be selected."""
        environment = services[RAG_SERVICE]["environment"]
        assert environment["RAG_PROVIDER"] == "local"
        assert environment["RAG_REPO_ROOT"] == "/rag"
        assert environment["RAG_BACKEND"] == "light"
        assert environment["RAG_RETRIEVAL_MODE"] == "hybrid"

    def test_standalone_service_does_not_enable_rag(self, services: dict[str, Any]) -> None:
        """The default service must not need a RAG checkout to start."""
        environment = services[STANDALONE_SERVICE].get("environment", {})
        assert "RAG_PROVIDER" not in environment, sorted(environment)

    def test_rag_checkout_is_mounted_read_only(self, services: dict[str, Any]) -> None:
        mounts = services[RAG_SERVICE]["volumes"]
        rag_mounts = [m for m in mounts if m.endswith(":/rag:ro")]
        assert len(rag_mounts) == 1, mounts

    def test_rag_mount_source_is_configurable_not_hardcoded(self, services: dict[str, Any]) -> None:
        """The mount source must come from the environment with a placeholder default."""
        mount = next(m for m in services[RAG_SERVICE]["volumes"] if m.endswith(":/rag:ro"))
        source = mount[: -len(":/rag:ro")]
        assert source.startswith("${RAG_HOST_REPO:-"), source
        assert source.endswith("}"), source

    def test_no_service_mounts_an_absolute_host_path(self, services: dict[str, Any]) -> None:
        """A committed absolute path would only work on one machine."""
        for name, service in services.items():
            for mount in service.get("volumes", []):
                assert not _DRIVE_LETTER_PATH.match(mount), (name, mount)
                assert not mount.startswith("/"), (name, mount)

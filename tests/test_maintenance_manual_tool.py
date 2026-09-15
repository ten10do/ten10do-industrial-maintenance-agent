"""Tests for the ``search_maintenance_manual`` tool.

The tool is a thin adapter over a ``RAGProvider``. These tests pin its contract:
what it returns when retrieval succeeds, when it finds nothing, and when it
cannot run at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.integrations.rag import RAGProvider, RAGProviderError, RAGSearchHit, RAGSearchResponse
from app.tools import registry
from app.tools.maintenance_manual_tool import (
    ManualSearchInput,
    ManualSearchOutput,
    search_maintenance_manual,
)
from app.tools.names import ToolName

MANUAL_TOOL = ToolName.SEARCH_MAINTENANCE_MANUAL.value


class _FakeProvider(RAGProvider):
    provider_id = "fake"

    def __init__(
        self,
        hits: list[RAGSearchHit] | None = None,
        *,
        error: Exception | None = None,
        retrieval_mode: str | None = "hybrid",
    ) -> None:
        self._hits = hits or []
        self._error = error
        self._retrieval_mode = retrieval_mode
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        self.calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        return RAGSearchResponse(query=query, hits=self._hits, retrieval_mode=self._retrieval_mode)


def test_tool_is_registered() -> None:
    assert MANUAL_TOOL in registry.names()
    spec = registry.get(MANUAL_TOOL)

    assert "rag" in spec.tags
    assert spec.func is search_maintenance_manual


def test_tool_maps_hits_onto_pydantic_output() -> None:
    provider = _FakeProvider(
        [
            RAGSearchHit(
                content="Drive overload trip is caused by excessive load.",
                document="ABB_ACS580_Firmware_Manual.pdf",
                page=42,
                section="4.3",
                chunk_id="chunk-00042",
                score=0.31,
            )
        ]
    )

    output = search_maintenance_manual("overload trip", provider=provider)

    assert isinstance(output, ManualSearchOutput)
    assert output.found is True
    assert output.provider == "fake"
    assert output.retrieval_mode == "hybrid"
    assert output.error is None
    hit = output.results[0]
    assert hit.document == "ABB_ACS580_Firmware_Manual.pdf"
    assert hit.page == 42
    assert hit.section == "4.3"
    assert hit.chunk_id == "chunk-00042"
    assert hit.score == pytest.approx(0.31)
    assert provider.calls == [("overload trip", 4)]


def test_tool_reports_empty_result_without_fabrication() -> None:
    provider = _FakeProvider([])

    output = search_maintenance_manual("unheard of failure", provider=provider)

    assert output.found is False
    assert output.results == []
    # An empty knowledge base result is not an error.
    assert output.error is None
    assert output.provider == "fake"


def test_tool_surfaces_provider_failure_in_error_field() -> None:
    provider = _FakeProvider(error=RAGProviderError("RAG service unreachable"))

    output = search_maintenance_manual("overload", provider=provider)

    assert output.found is False
    assert output.results == []
    assert output.error == "RAG service unreachable"


def test_tool_reports_unconfigured_provider_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.integrations.rag.factory.get_settings",
        lambda: Settings(rag_provider="local", rag_repo_root=""),
    )
    from app.integrations.rag import reset_rag_provider

    reset_rag_provider()

    output = search_maintenance_manual("overload")

    assert output.found is False
    assert output.error is not None
    assert "repo root is required" in output.error


def test_tool_uses_configured_default_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.tools.maintenance_manual_tool.get_settings",
        lambda: Settings(rag_top_k=2),
    )
    provider = _FakeProvider([])

    search_maintenance_manual("q", provider=provider)

    assert provider.calls == [("q", 2)]


def test_tool_respects_explicit_top_k() -> None:
    provider = _FakeProvider([])

    search_maintenance_manual("q", top_k=8, provider=provider)

    assert provider.calls == [("q", 8)]


def test_tool_input_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        ManualSearchInput(query="")


def test_tool_input_rejects_out_of_range_top_k() -> None:
    with pytest.raises(ValidationError):
        ManualSearchInput(query="q", top_k=9)


def test_tool_never_invents_missing_fields() -> None:
    provider = _FakeProvider([RAGSearchHit(content="fragment only")])

    output = search_maintenance_manual("q", provider=provider)

    hit = output.results[0]
    assert hit.document is None
    assert hit.page is None
    assert hit.section is None
    assert hit.chunk_id is None
    assert hit.score is None


def test_tool_output_is_json_serializable() -> None:
    provider = _FakeProvider([RAGSearchHit(content="fragment", document="a.pdf", page=3)])

    payload: dict[str, Any] = search_maintenance_manual("q", provider=provider).model_dump(
        mode="json"
    )

    assert payload["results"][0]["page"] == 3

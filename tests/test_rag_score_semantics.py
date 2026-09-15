"""Tests for RAG score semantics.

The RAG engine returns different quantities per retrieval mode. These tests pin
down that the label is derived from what the engine actually produced, that a
distance is never rewritten as a similarity, and that mixed semantics inside one
response are detectable rather than silently ranked together.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import httpx
import pytest

from app.agent.graph import retrieve_context
from app.agent.state import MaintenanceState
from app.integrations.rag import (
    HttpRAGProvider,
    LocalRAGProvider,
    RAGProvider,
    RAGSearchHit,
    RAGSearchResponse,
)
from app.integrations.rag.scores import (
    FUSION_SCORE_EXPOSED,
    LEXICAL_RELEVANCE_FLAG,
    VECTOR_COSINE_DISTANCE,
    VECTOR_STORE_DISTANCE,
    comparable,
    describe_score,
    has_vector_component,
)
from app.tools.maintenance_manual_tool import search_maintenance_manual
from app.tools.names import ToolName

VALID_KB_ID = "kb-public-shared-00000001"
RAG_BASE_URL = "http://rag.invalid:8123"
MANUAL_TOOL = ToolName.SEARCH_MAINTENANCE_MANUAL.value


# --------------------------------------------------------------------------- #
# 1. Label derivation
# --------------------------------------------------------------------------- #


def test_lexical_mode_is_labelled_as_a_binary_flag() -> None:
    semantics = describe_score("lexical", has_vector_component=False)

    assert semantics.semantics == LEXICAL_RELEVANCE_FLAG
    assert semantics.higher_is_better is False


def test_vector_mode_uses_the_cosine_distance_label_in_light_mode() -> None:
    semantics = describe_score("vector", has_vector_component=True, backend="light")

    assert semantics.semantics == VECTOR_COSINE_DISTANCE
    assert semantics.higher_is_better is False


def test_vector_mode_uses_the_store_distance_label_in_full_mode() -> None:
    semantics = describe_score("vector", has_vector_component=True, backend="full")

    assert semantics.semantics == VECTOR_STORE_DISTANCE
    assert semantics.higher_is_better is False


def test_hybrid_vector_side_inherits_the_vector_distance() -> None:
    semantics = describe_score("hybrid", has_vector_component=True)

    assert semantics.semantics == VECTOR_COSINE_DISTANCE


def test_hybrid_lexical_only_side_inherits_the_lexical_flag() -> None:
    semantics = describe_score("hybrid", has_vector_component=False)

    assert semantics.semantics == LEXICAL_RELEVANCE_FLAG


@pytest.mark.parametrize("mode", [None, "", "  ", "bm25", "tfidf", "rerank"])
def test_unrecognised_mode_is_left_unannotated(mode: str | None) -> None:
    semantics = describe_score(mode, has_vector_component=True)

    assert semantics.semantics is None
    assert semantics.higher_is_better is None
    assert semantics.annotated is False


def test_unrecognised_backend_is_left_unannotated() -> None:
    semantics = describe_score("vector", has_vector_component=True, backend="chroma")

    assert semantics.annotated is False


def test_mode_matching_is_case_insensitive() -> None:
    assert describe_score("HYBRID", has_vector_component=True).semantics == (VECTOR_COSINE_DISTANCE)


# --------------------------------------------------------------------------- #
# 2. Direction is never inverted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
@pytest.mark.parametrize("backend", ["light", "full"])
@pytest.mark.parametrize("vector_side", [True, False])
def test_no_annotated_score_is_higher_is_better(
    mode: str,
    backend: str,
    vector_side: bool,
) -> None:
    """Every score the RAG engine exposes is distance-like or flag-like.

    A cosine similarity would carry ``higher_is_better=True``. Nothing here may
    produce that, because producing it would mean a distance had been converted
    into a similarity.
    """
    semantics = describe_score(mode, has_vector_component=vector_side, backend=backend)

    assert semantics.annotated is True
    assert semantics.higher_is_better is False


def test_the_fused_score_is_not_exposed_as_the_relevance_score() -> None:
    assert FUSION_SCORE_EXPOSED is False


# --------------------------------------------------------------------------- #
# 3. Mixed semantics inside one response
# --------------------------------------------------------------------------- #


def test_hybrid_response_can_mix_two_score_meanings() -> None:
    from_vector = describe_score("hybrid", has_vector_component=True)
    from_lexical = describe_score("hybrid", has_vector_component=False)

    assert from_vector.semantics != from_lexical.semantics
    assert comparable([from_vector.semantics, from_lexical.semantics]) is False


def test_comparable_accepts_a_single_meaning() -> None:
    assert comparable([VECTOR_COSINE_DISTANCE, VECTOR_COSINE_DISTANCE]) is True
    assert comparable([VECTOR_COSINE_DISTANCE]) is True
    assert comparable([]) is True


def test_comparable_ignores_unlabelled_scores() -> None:
    assert comparable([VECTOR_COSINE_DISTANCE, None]) is True


def test_has_vector_component_reads_rank_or_source() -> None:
    assert has_vector_component(None, 3) is True
    assert has_vector_component("vector", None) is True
    assert has_vector_component("hybrid", None) is True
    assert has_vector_component("lexical", None) is False
    assert has_vector_component(None, None) is False


# --------------------------------------------------------------------------- #
# 4. Local provider annotation
# --------------------------------------------------------------------------- #


class _FakeDocument:
    def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata


class _FakeCandidate:
    def __init__(self, retrieval_source: str, vector_rank: int | None) -> None:
        self.retrieval_source = retrieval_source
        self.vector_rank = vector_rank


class _FakeResult(list):
    def __init__(self, pairs: list[tuple[Any, Any]], *, candidates: list[Any], mode: str) -> None:
        super().__init__(pairs)
        self.candidates = candidates
        self.retrieval_mode = mode


def _install_fake_rag_module(
    monkeypatch: pytest.MonkeyPatch,
    result: Any,
    error: Exception | None = None,
    module_name: str = "backend.light_rag_core",
) -> None:
    module = types.ModuleType(module_name)

    def retrieve_docs(
        question: str,
        k: int = 4,
        knowledge_base_id: str = "default",
        retrieval_mode: str | None = None,
    ) -> Any:
        if error is not None:
            raise error
        return result

    module.retrieve_docs = retrieve_docs  # type: ignore[attr-defined]
    package = types.ModuleType("backend")
    package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "backend", package)
    monkeypatch.setitem(sys.modules, module_name, module)


def test_local_provider_annotates_a_hybrid_vector_hit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    document = _FakeDocument("fragment", {"source": "a.pdf", "page": 4, "chunk_id": "c-1"})
    result = _FakeResult(
        [(document, 0.27)],
        candidates=[_FakeCandidate("hybrid", 1)],
        mode="hybrid",
    )
    _install_fake_rag_module(monkeypatch, result)

    hit = LocalRAGProvider(tmp_path).search("q").hits[0]

    assert hit.score == pytest.approx(0.27)
    assert hit.score_semantics == VECTOR_COSINE_DISTANCE
    assert hit.higher_is_better is False


def test_local_provider_annotates_a_lexical_only_hit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    document = _FakeDocument("fragment", {"source": "a.pdf"})
    result = _FakeResult(
        [(document, 1.0)],
        candidates=[_FakeCandidate("lexical", None)],
        mode="lexical",
    )
    _install_fake_rag_module(monkeypatch, result)

    hit = LocalRAGProvider(tmp_path).search("q").hits[0]

    assert hit.score_semantics == LEXICAL_RELEVANCE_FLAG
    assert hit.higher_is_better is False


def test_local_provider_uses_full_backend_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    document = _FakeDocument("fragment", {"source": "a.pdf"})
    result = _FakeResult(
        [(document, 3.4)],
        candidates=[_FakeCandidate("vector", 1)],
        mode="vector",
    )
    _install_fake_rag_module(monkeypatch, result)
    hit = LocalRAGProvider(tmp_path, rag_backend="light").search("q").hits[0]
    assert hit.score_semantics == VECTOR_COSINE_DISTANCE

    _install_fake_rag_module(monkeypatch, result, module_name="backend.rag_core")
    hit_full = LocalRAGProvider(tmp_path, rag_backend="full").search("q").hits[0]
    assert hit_full.score_semantics == VECTOR_STORE_DISTANCE


def test_local_provider_leaves_semantics_empty_without_a_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    document = _FakeDocument("fragment", {"source": "a.pdf"})
    result = _FakeResult([(document, "not-a-number")], candidates=[], mode="hybrid")
    _install_fake_rag_module(monkeypatch, result)

    hit = LocalRAGProvider(tmp_path).search("q").hits[0]

    assert hit.score is None
    assert hit.score_semantics is None
    assert hit.higher_is_better is None


# --------------------------------------------------------------------------- #
# 5. HTTP provider annotation
# --------------------------------------------------------------------------- #


def _source(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "citation_id": "S1",
        "source": "manual.pdf",
        "page": 12,
        "score": 0.19,
        "content": "text",
        "section": "3.1",
        "chunk_id": "chunk-1",
        "retrieval_source": "hybrid",
        "vector_rank": 1,
    }
    item.update(overrides)
    return item


def _provider(*sources: dict[str, Any], backend: str = "light") -> HttpRAGProvider:
    payload = {"answer": "discarded", "is_refused": False, "sources": list(sources)}
    return HttpRAGProvider(
        RAG_BASE_URL,
        VALID_KB_ID,
        backend=backend,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )


def test_http_provider_annotates_from_retrieval_source() -> None:
    hit = _provider(_source(retrieval_source="vector", vector_rank=2)).search("q").hits[0]

    assert hit.score_semantics == VECTOR_COSINE_DISTANCE
    assert hit.higher_is_better is False


def test_http_provider_annotates_lexical_sources() -> None:
    hit = _provider(_source(retrieval_source="lexical", vector_rank=None)).search("q").hits[0]

    assert hit.score_semantics == LEXICAL_RELEVANCE_FLAG


def test_http_provider_annotates_unlabelled_sources_as_unknown() -> None:
    hit = _provider(_source(retrieval_source="", vector_rank=None)).search("q").hits[0]

    assert hit.score is not None
    assert hit.score_semantics is None
    assert hit.higher_is_better is None


def test_http_provider_honours_the_configured_backend() -> None:
    hit = (
        _provider(_source(retrieval_source="vector", vector_rank=1), backend="full")
        .search("q")
        .hits[0]
    )

    assert hit.score_semantics == VECTOR_STORE_DISTANCE


# --------------------------------------------------------------------------- #
# 6. Tool and Evidence carry the annotation through
# --------------------------------------------------------------------------- #


class _StubProvider(RAGProvider):
    provider_id = "stub"

    def __init__(self, hit: RAGSearchHit) -> None:
        self._hit = hit

    def search(self, query: str, top_k: int = 4) -> RAGSearchResponse:
        return RAGSearchResponse(query=query, hits=[self._hit], retrieval_mode="hybrid")


def test_tool_output_carries_score_semantics() -> None:
    hit = RAGSearchHit(
        content="fragment",
        document="manual.pdf",
        page=7,
        score=0.24,
        score_semantics=VECTOR_COSINE_DISTANCE,
        higher_is_better=False,
    )
    output = search_maintenance_manual("q", provider=_StubProvider(hit))

    result = output.results[0]
    assert result.score_semantics == VECTOR_COSINE_DISTANCE
    assert result.higher_is_better is False


def test_document_evidence_carries_score_semantics() -> None:
    payload = {
        "query": "q",
        "found": True,
        "provider": "stub",
        "retrieval_mode": "hybrid",
        "error": None,
        "results": [
            {
                "content": "fragment",
                "document": "manual.pdf",
                "page": 7,
                "section": "3.1",
                "chunk_id": "chunk-1",
                "score": 0.24,
                "score_semantics": VECTOR_COSINE_DISTANCE,
                "higher_is_better": False,
            }
        ],
    }
    state = MaintenanceState(tool_results=[{"tool": MANUAL_TOOL, "result": payload}])

    evidence = retrieve_context(state)["retrieved_context"][0]

    assert evidence["source_type"] == "document"
    assert evidence["score"] == pytest.approx(0.24)
    assert evidence["score_semantics"] == VECTOR_COSINE_DISTANCE
    assert evidence["higher_is_better"] is False


def test_tool_evidence_has_no_score_semantics() -> None:
    state = MaintenanceState(
        tool_results=[{"tool": ToolName.GET_DEVICE_STATUS.value, "result": {"found": False}}]
    )

    evidence = retrieve_context(state)["retrieved_context"][0]

    assert evidence["score"] is None
    assert evidence["score_semantics"] is None
    assert evidence["higher_is_better"] is None

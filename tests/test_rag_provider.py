"""Tests for the RAG provider abstraction.

Both providers are exercised against recorded, contract-faithful payloads for
the RAG repository's real interfaces: ``backend.light_rag_core.retrieve_docs``
for the in-process provider and ``POST /ask`` for the HTTP provider. No test
contacts a live service or a live knowledge base.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.integrations.rag import (
    HttpRAGProvider,
    LocalRAGProvider,
    RAGProvider,
    RAGProviderError,
    RAGSearchHit,
    RAGSearchResponse,
    build_provider,
    get_rag_provider,
    reset_rag_provider,
)
from app.integrations.rag.scores import VECTOR_COSINE_DISTANCE

VALID_KB_ID = "kb-public-shared-00000001"
RAG_BASE_URL = "http://rag.invalid:8123"


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class _FakeDocument:
    def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata


class _FakeResult(list):
    """Mimics ``backend.retrieval.candidates.RetrievalResult``."""

    def __init__(
        self,
        pairs: list[tuple[Any, Any]],
        retrieval_mode: str = "hybrid",
        candidates: list[Any] | None = None,
    ) -> None:
        super().__init__(pairs)
        self.retrieval_mode = retrieval_mode
        self.candidates = candidates or []


class _FakeCandidate:
    def __init__(self, retrieval_source: str, vector_rank: int | None = None) -> None:
        self.retrieval_source = retrieval_source
        self.vector_rank = vector_rank


def _install_fake_rag_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: Any = None,
    error: Exception | None = None,
) -> list[dict[str, Any]]:
    """Install a stand-in ``backend.light_rag_core`` module and capture calls."""
    calls: list[dict[str, Any]] = []

    module = types.ModuleType("backend.light_rag_core")

    def retrieve_docs(
        question: str,
        k: int = 4,
        knowledge_base_id: str = "default",
        retrieval_mode: str | None = None,
    ) -> Any:
        calls.append(
            {
                "question": question,
                "k": k,
                "knowledge_base_id": knowledge_base_id,
                "retrieval_mode": retrieval_mode,
            }
        )
        if error is not None:
            raise error
        return result

    module.retrieve_docs = retrieve_docs  # type: ignore[attr-defined]

    package = types.ModuleType("backend")
    package.__path__ = []  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "backend", package)
    monkeypatch.setitem(sys.modules, "backend.light_rag_core", module)
    return calls


@pytest.fixture(autouse=True)
def _clear_provider_cache() -> Iterator[None]:
    reset_rag_provider()
    yield
    reset_rag_provider()


# --------------------------------------------------------------------------- #
# 1. Transport models
# --------------------------------------------------------------------------- #


def test_search_response_found_flag_follows_hits() -> None:
    empty = RAGSearchResponse(query="q")
    filled = RAGSearchResponse(query="q", hits=[RAGSearchHit(content="x")])

    assert empty.found is False
    assert empty.hits == []
    assert filled.found is True


def test_search_hit_keeps_missing_fields_as_none() -> None:
    hit = RAGSearchHit(content="only content")

    assert hit.document is None
    assert hit.page is None
    assert hit.section is None
    assert hit.chunk_id is None
    assert hit.score is None


# --------------------------------------------------------------------------- #
# 2. Local provider mapping
# --------------------------------------------------------------------------- #


def test_local_provider_maps_retrieval_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    document = _FakeDocument(
        "热封刀温度上限 80 摄氏度。",
        {
            "source": r"D:\manuals\PLC_User_Manual.pdf",
            "page": 41,
            "section": "4.3",
            "chunk_id": "c-0007",
        },
    )
    calls = _install_fake_rag_module(
        monkeypatch,
        result=_FakeResult(
            [(document, 0.3125)],
            candidates=[_FakeCandidate("hybrid", vector_rank=1)],
        ),
    )
    provider = LocalRAGProvider(tmp_path, knowledge_base_id="kb-1", retrieval_mode="hybrid")

    response = provider.search("热封刀温度上限", top_k=3)

    assert response.found is True
    assert response.retrieval_mode == "hybrid"
    hit = response.hits[0]
    # The upstream page index is zero-based; the tool contract is one-based.
    assert hit.page == 42
    assert hit.document == "PLC_User_Manual.pdf"
    assert hit.section == "4.3"
    assert hit.chunk_id == "c-0007"
    assert hit.score == pytest.approx(0.3125)
    assert hit.score_semantics == VECTOR_COSINE_DISTANCE
    assert hit.higher_is_better is False
    assert calls == [
        {
            "question": "热封刀温度上限",
            "k": 3,
            "knowledge_base_id": "kb-1",
            "retrieval_mode": "hybrid",
        }
    ]


def test_local_provider_falls_back_to_page_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    document = _FakeDocument("fragment", {"source": "a.pdf", "page_start": 9})
    _install_fake_rag_module(monkeypatch, result=_FakeResult([(document, 1.0)]))
    provider = LocalRAGProvider(tmp_path)

    hit = provider.search("q").hits[0]

    assert hit.page == 10


def test_local_provider_does_not_invent_a_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    document = _FakeDocument("fragment", {"source": "a.pdf", "page": "未知页码"})
    _install_fake_rag_module(monkeypatch, result=_FakeResult([(document, 1.0)]))
    provider = LocalRAGProvider(tmp_path)

    hit = provider.search("q").hits[0]

    assert hit.page is None
    assert hit.section is None
    assert hit.chunk_id is None


def test_local_provider_raises_on_empty_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _install_fake_rag_module(monkeypatch, error=ValueError("请先上传 PDF 并构建知识库。"))
    provider = LocalRAGProvider(tmp_path)

    with pytest.raises(RAGProviderError, match="请先上传 PDF"):
        provider.search("q")


def test_local_provider_requires_existing_repo_root(tmp_path: Any) -> None:
    provider = LocalRAGProvider(tmp_path / "missing")

    with pytest.raises(RAGProviderError, match="RAG repository not found"):
        provider.search("q")


def test_local_provider_requires_repo_root() -> None:
    with pytest.raises(ValueError, match="repo root is required"):
        LocalRAGProvider("  ")


def test_local_provider_rejects_unknown_backend(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported RAG backend"):
        LocalRAGProvider(tmp_path, rag_backend="chroma")


# --------------------------------------------------------------------------- #
# 3. HTTP provider mapping and request contract
# --------------------------------------------------------------------------- #


def _ask_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "answer": "LLM generated summary that the agent discards.",
        "is_refused": False,
        "sources": [
            {
                "citation_id": "S1",
                "source": "ABB_ACS580_Firmware_Manual.pdf",
                "page": 42,
                "score": 0.31,
                "content": "Drive overload trip is caused by excessive load.",
                "section": "4.3 Overload protection",
                "chunk_id": "chunk-00042",
                "retrieval_source": "hybrid",
                "vector_rank": 1,
            }
        ],
    }
    payload.update(overrides)
    return payload


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_http_provider_sends_contract_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_ask_payload())

    provider = HttpRAGProvider(
        RAG_BASE_URL,
        VALID_KB_ID,
        timeout_seconds=3.0,
        model_provider="DeepSeek",
        client=_client(handler),
    )

    provider.search("F0045 overload", top_k=4)

    assert captured["url"] == f"{RAG_BASE_URL}/ask"
    assert captured["headers"]["x-knowledge-base-id"] == VALID_KB_ID
    assert captured["body"] == {
        "question": "F0045 overload",
        "top_k": 4,
        "model_provider": "DeepSeek",
    }


def test_http_provider_maps_sources_to_hits() -> None:
    provider = HttpRAGProvider(
        RAG_BASE_URL,
        VALID_KB_ID,
        client=_client(lambda request: httpx.Response(200, json=_ask_payload())),
    )

    response = provider.search("overload", top_k=4)

    assert response.found is True
    hit = response.hits[0]
    assert hit.content == "Drive overload trip is caused by excessive load."
    assert hit.document == "ABB_ACS580_Firmware_Manual.pdf"
    assert hit.page == 42
    assert hit.section == "4.3 Overload protection"
    assert hit.chunk_id == "chunk-00042"
    assert hit.score == pytest.approx(0.31)
    assert hit.score_semantics == VECTOR_COSINE_DISTANCE
    assert hit.higher_is_better is False


def test_http_provider_honours_upstream_refusal() -> None:
    payload = _ask_payload(sources=[], is_refused=True)
    provider = HttpRAGProvider(
        RAG_BASE_URL,
        VALID_KB_ID,
        client=_client(lambda request: httpx.Response(200, json=payload)),
    )

    response = provider.search("unknown topic", top_k=4)

    assert response.found is False
    assert response.hits == []


def test_http_provider_reports_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = HttpRAGProvider(RAG_BASE_URL, VALID_KB_ID, client=_client(handler))

    with pytest.raises(RAGProviderError, match="unreachable"):
        provider.search("q")


def test_http_provider_reports_error_status() -> None:
    provider = HttpRAGProvider(
        RAG_BASE_URL,
        VALID_KB_ID,
        client=_client(lambda request: httpx.Response(500, json={"detail": "boom"})),
    )

    with pytest.raises(RAGProviderError, match="HTTP 500"):
        provider.search("q")


def test_http_provider_requires_base_url_and_valid_kb_id() -> None:
    with pytest.raises(ValueError, match="base URL is required"):
        HttpRAGProvider("", VALID_KB_ID)

    with pytest.raises(ValueError, match="Knowledge base id must be"):
        HttpRAGProvider(RAG_BASE_URL, "short")


# --------------------------------------------------------------------------- #
# 4. Factory selection
# --------------------------------------------------------------------------- #


def test_build_provider_selects_local() -> None:
    settings = Settings(rag_provider="local", rag_repo_root="D:/rag", rag_backend="light")

    provider = build_provider(settings)

    assert isinstance(provider, LocalRAGProvider)
    assert isinstance(provider, RAGProvider)
    assert provider.describe()["rag_backend"] == "light"


def test_build_provider_selects_http() -> None:
    settings = Settings(
        rag_provider="http",
        rag_base_url=RAG_BASE_URL,
        rag_knowledge_base_id=VALID_KB_ID,
    )

    provider = build_provider(settings)

    assert isinstance(provider, HttpRAGProvider)


def test_build_provider_rejects_unknown_provider() -> None:
    with pytest.raises(RAGProviderError, match="Unknown RAG provider"):
        build_provider(Settings(rag_provider="pinecone"))


def test_build_provider_reports_missing_repo_root() -> None:
    with pytest.raises(RAGProviderError, match="repo root is required"):
        build_provider(Settings(rag_provider="local", rag_repo_root=""))


def test_get_rag_provider_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.integrations.rag.factory.get_settings",
        lambda: Settings(rag_provider="local", rag_repo_root="D:/rag"),
    )
    reset_rag_provider()

    assert get_rag_provider() is get_rag_provider()

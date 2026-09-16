"""Build the Industrial Knowledge RAG knowledge base from the frozen real corpus.

This script is an invocation wrapper, not an implementation. It calls the RAG
project's own ingestion and indexing entry point
(``light_rag_core.build_knowledge_base``), so parsing, chunking, embedding and
index persistence all stay owned by the RAG project.

Read-only guarantees with respect to the RAG repository checkout:

* no Python source, requirements, README, ``.env`` or Git metadata is touched;
* the only artifact written is the light index JSON under the RAG repository's
  gitignored ``backend/light_indexes/`` directory, plus an in-memory TF-IDF and
  BM25 index that is never persisted elsewhere.

Usage::

    python scripts/rag_build_knowledge_base.py --rag-repo-root /path/to/industrial-knowledge-rag

Environment variable values are never printed. If importing the RAG package
pulls the RAG ``.env`` into this process (``backend.llm_client`` calls
``load_dotenv`` on import), the script reports only
``rag_env_side_effect_detected`` plus the names of the newly introduced keys.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

#: Conventional location of the corpus inside a RAG checkout. It is a default
#: only: a checkout that keeps its corpus elsewhere must pass ``--corpus-dir``
#: explicitly, and no path outside this repository is assumed to exist.
DEFAULT_CORPUS_SUBPATH = Path("backend/evaluation/benchmark_corpus/documents")
DEFAULT_KB_ID = "default"
DEFAULT_MAX_CHUNKS = 20000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-repo-root", required=True)
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--knowledge-base-id", default=DEFAULT_KB_ID)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(args.rag_repo_root).expanduser().resolve()
    corpus_dir = (
        Path(args.corpus_dir).expanduser().resolve()
        if args.corpus_dir
        else repo_root / DEFAULT_CORPUS_SUBPATH
    )

    report: dict[str, object] = {
        "rag_repo_root": str(repo_root),
        "corpus_dir": str(corpus_dir),
        "knowledge_base_id": args.knowledge_base_id,
    }

    if not repo_root.is_dir():
        report["status"] = "RAG_REPO_NOT_FOUND"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    if not corpus_dir.is_dir():
        report["status"] = "CORPUS_NOT_FOUND"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    pdf_paths = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_paths:
        report["status"] = "CORPUS_NOT_FOUND"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    report["status"] = "BUILDING"
    report["document_files"] = [path.name for path in pdf_paths]
    report["document_count"] = len(pdf_paths)
    report["corpus_bytes"] = sum(path.stat().st_size for path in pdf_paths)
    report["max_knowledge_base_chunks"] = args.max_chunks

    env_before = set(os.environ)
    os.environ["MAX_KNOWLEDGE_BASE_CHUNKS"] = str(args.max_chunks)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    backend_name = "backend.light_rag_core"
    try:
        backend = importlib.import_module(backend_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        report["status"] = "BUILD_FAILED"
        report["error"] = f"cannot import {backend_name}: {exc}"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    introduced = sorted(set(os.environ) - env_before)
    report["rag_env_side_effect_detected"] = bool(introduced)
    report["rag_env_introduced_key_names"] = introduced

    try:
        page_count, chunk_count = backend.build_knowledge_base(pdf_paths, args.knowledge_base_id)
    except Exception as exc:
        report["status"] = "BUILD_FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    index_path = Path(backend.get_index_storage_path(args.knowledge_base_id))
    ready = bool(backend.is_knowledge_base_ready(args.knowledge_base_id))

    report.update(
        {
            "status": "BUILT" if ready else "NOT_READY",
            "is_knowledge_base_ready": ready,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "data_dir": str(backend.get_data_dir(args.knowledge_base_id)),
            "index_path": str(index_path),
            "index_exists": index_path.is_file(),
            "index_bytes": index_path.stat().st_size if index_path.is_file() else 0,
            "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
        }
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

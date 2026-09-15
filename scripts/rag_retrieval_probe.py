"""Run real retrieval-only queries against the Industrial Knowledge RAG engine.

Calls ``light_rag_core.retrieve_docs`` directly. No HTTP request is made, no
``POST /ask`` is issued and no answer generation runs, so there is no LLM
anywhere in this path.

Every reported field comes from the RAG retrieval result itself. Nothing is
normalized, re-scored or converted: raw scores are printed as returned together
with the retrieval mode that produced them.

Usage::

    python scripts/rag_retrieval_probe.py \
        --rag-repo-root /path/to/industrial-knowledge-rag \
        --query "CompactLogix 5380 controller" \
        --query "PowerFlex 527 fault code" \
        --top-k 5

Environment variable values are never printed. If importing the RAG package
pulls the RAG ``.env`` into this process, only
``rag_env_side_effect_detected`` and the names of newly introduced keys are
reported.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

DEFAULT_KB_ID = "default"
MAX_PAGE_CHARS = 240


def truncate(value: str, limit: int = MAX_PAGE_CHARS) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-repo-root", required=True)
    parser.add_argument("--knowledge-base-id", default=DEFAULT_KB_ID)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        default=None,
        help="lexical | vector | hybrid. Defaults to the RAG project setting.",
    )
    parser.add_argument("--content-chars", type=int, default=MAX_PAGE_CHARS)
    return parser


def describe_hit(document: object, score: object) -> dict[str, object]:
    metadata = getattr(document, "metadata", None) or {}
    raw_page = metadata.get("page")
    raw_page_start = metadata.get("page_start")

    def as_int(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return {
        "document": Path(str(metadata.get("source", ""))).name or None,
        "page_index_zero_based": as_int(raw_page),
        "page_start_index_zero_based": as_int(raw_page_start),
        "section": metadata.get("section") or None,
        "subsection": metadata.get("subsection") or None,
        "chunk_id": metadata.get("chunk_id") or None,
        "manufacturer": metadata.get("manufacturer") or None,
        "equipment_model": metadata.get("equipment_model") or None,
        "knowledge_type": metadata.get("knowledge_type") or None,
        "raw_score": float(score) if isinstance(score, int | float) else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(args.rag_repo_root).expanduser().resolve()
    report: dict[str, object] = {
        "rag_repo_root": str(repo_root),
        "knowledge_base_id": args.knowledge_base_id,
        "top_k": args.top_k,
        "requested_mode": args.mode,
    }

    if not repo_root.is_dir():
        report["status"] = "RAG_REPO_NOT_FOUND"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    env_before = set(os.environ)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    backend_name = "backend.light_rag_core"
    try:
        backend = importlib.import_module(backend_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        report["status"] = "RETRIEVAL_FAILED"
        report["error"] = f"cannot import {backend_name}: {exc}"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    introduced = sorted(set(os.environ) - env_before)
    report["rag_env_side_effect_detected"] = bool(introduced)
    report["rag_env_introduced_key_names"] = introduced
    report["is_knowledge_base_ready"] = bool(
        backend.is_knowledge_base_ready(args.knowledge_base_id)
    )
    report["index_path"] = str(backend.get_index_storage_path(args.knowledge_base_id))

    results: list[dict[str, object]] = []
    for query in args.query:
        entry: dict[str, object] = {"query": query, "top_k": args.top_k}
        try:
            result = backend.retrieve_docs(
                query,
                k=args.top_k,
                knowledge_base_id=args.knowledge_base_id,
                retrieval_mode=args.mode,
            )
        except Exception as exc:
            entry["status"] = "RETRIEVAL_FAILED"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
            continue

        hits = []
        for document, score in result:
            hit = describe_hit(document, score)
            hit["content"] = truncate(
                str(getattr(document, "page_content", "")), args.content_chars
            )
            hits.append(hit)

        entry.update(
            {
                "status": "OK",
                "retrieval_mode": getattr(result, "retrieval_mode", None),
                "hit_count": len(hits),
                "hits": hits,
            }
        )
        results.append(entry)

    report["status"] = "OK"
    report["query_count"] = len(args.query)
    report["results"] = results

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

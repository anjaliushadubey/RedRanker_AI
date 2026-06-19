#!/usr/bin/env python3
"""Rank top-2000 candidates from prepared embedding/BM25 caches."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import (
    DEFAULT_BM25_CACHE_PATH,
    DEFAULT_DENSE_BACKEND,
    DEFAULT_ELIGIBLE_PATH,
    DEFAULT_QDRANT_PATH,
    DEFAULT_QUERY_EMBEDDINGS_PATH,
    DEFAULT_SECTION_EMBEDDINGS_DIR,
    DEFAULT_SECTION_EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME,
    DEFAULT_TOP_2000_CSV_PATH,
    DEFAULT_TOP_2000_JSONL_PATH,
    QDRANT_COLLECTION_NAME,
    RETRIEVAL_SECTION_WEIGHTS,
    TOP_2000_N,
)
from src.data_loader import load_candidates
from src.retrieval import retrieve_top_candidates_from_cache, section_embedding_cache_paths, write_retrieval_outputs


def resolve_cli_embedding_cache_dir(cache_dir: Path | None, legacy_cache_path: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir
    if legacy_cache_path is not None:
        return legacy_cache_path.parent if legacy_cache_path.suffix else legacy_cache_path
    return DEFAULT_SECTION_EMBEDDINGS_DIR


def build_top_2000(
    eligible_input: Path,
    csv_path: Path,
    jsonl_path: Path,
    top_n: int = TOP_2000_N,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    qdrant_collection: str = QDRANT_COLLECTION_NAME,
    embedding_cache_path: Path | None = DEFAULT_SECTION_EMBEDDINGS_PATH,
    embedding_cache_dir: Path | None = DEFAULT_SECTION_EMBEDDINGS_DIR,
    query_cache_path: Path | None = DEFAULT_QUERY_EMBEDDINGS_PATH,
    bm25_cache_path: Path | None = DEFAULT_BM25_CACHE_PATH,
    qdrant_path: Path | None = DEFAULT_QDRANT_PATH,
    dense_backend: str = DEFAULT_DENSE_BACKEND,
) -> None:
    started_at = time.perf_counter()
    eligible_candidates = list(load_candidates(eligible_input, loader="jsonl"))
    print(f"Loaded eligible candidates from {eligible_input}")

    results = retrieve_top_candidates_from_cache(
        eligible_candidates,
        top_n=top_n,
        model_name=embedding_model,
        collection_name=qdrant_collection,
        embedding_cache_path=embedding_cache_path,
        embedding_cache_dir=embedding_cache_dir,
        query_cache_path=query_cache_path,
        bm25_cache_path=bm25_cache_path,
        qdrant_path=qdrant_path,
        dense_backend=dense_backend,
    )
    write_retrieval_outputs(results, csv_path=csv_path, jsonl_path=jsonl_path)

    print(f"Eligible candidates: {len(eligible_candidates)}")
    print(f"Candidates sent to dense rerank: {len(eligible_candidates)}")
    print(f"Wrote top {min(top_n, len(results))} CSV to {csv_path}")
    print(f"Wrote top {min(top_n, len(results))} JSONL to {jsonl_path}")
    print(f"Embedding model: {embedding_model}")
    print(f"Embedding cache dir: {embedding_cache_dir}")
    cache_paths = section_embedding_cache_paths(embedding_cache_dir) or {}
    for section, cache_path in cache_paths.items():
        print(f"  {section}: {cache_path}")
    print(f"Query embedding cache: {query_cache_path}")
    print(f"BM25 cache: {bm25_cache_path}")
    print(f"Qdrant storage path: {qdrant_path}")
    print(f"Qdrant collection: {qdrant_collection}")
    print(f"Dense backend: {dense_backend}")
    weights_text = ", ".join(
        f"{section} {weight:.0%}" for section, weight in RETRIEVAL_SECTION_WEIGHTS.items()
    )
    print(f"Dense section weights: {weights_text}")
    print("Hybrid weights: dense 70%, BM25 30%")
    print(f"Elapsed seconds: {time.perf_counter() - started_at:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank top-2000 from prepared embedding/BM25 caches")
    parser.add_argument(
        "--eligible-input",
        type=Path,
        default=DEFAULT_ELIGIBLE_PATH,
        help="Start directly from an existing eligible_candidates.jsonl file.",
    )
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_TOP_2000_CSV_PATH)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_TOP_2000_JSONL_PATH)
    parser.add_argument("--top", type=int, default=TOP_2000_N)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Legacy combined-cache argument. Its parent directory is used for per-section caches.",
    )
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_EMBEDDINGS_PATH)
    parser.add_argument("--bm25-cache", type=Path, default=DEFAULT_BM25_CACHE_PATH)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-collection", default=QDRANT_COLLECTION_NAME)
    parser.add_argument(
        "--dense-backend",
        choices=["numpy", "qdrant"],
        default=DEFAULT_DENSE_BACKEND,
        help="Use exact NumPy matrix similarity by default; Qdrant is available for experiments.",
    )
    args = parser.parse_args()
    embedding_cache_dir = resolve_cli_embedding_cache_dir(args.embedding_cache_dir, args.embedding_cache)

    build_top_2000(
        eligible_input=args.eligible_input,
        csv_path=args.csv_out,
        jsonl_path=args.jsonl_out,
        top_n=args.top,
        embedding_model=args.embedding_model,
        qdrant_collection=args.qdrant_collection,
        embedding_cache_path=args.embedding_cache,
        embedding_cache_dir=embedding_cache_dir,
        query_cache_path=args.query_cache,
        bm25_cache_path=args.bm25_cache,
        qdrant_path=args.qdrant_path,
        dense_backend=args.dense_backend,
    )


if __name__ == "__main__":
    main()

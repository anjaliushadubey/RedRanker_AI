#!/usr/bin/env python3
"""Prepare all embedding and BM25 artifacts for top2000.py."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def configure_cpu_threads() -> int | None:
    threads = max(1, min(os.cpu_count() or 1, 12))
    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, threads // 2)))
        except RuntimeError:
            pass
        return threads
    except Exception:
        return None


CPU_THREADS = configure_cpu_threads()

from src.config import (
    DEFAULT_BM25_CACHE_PATH,
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_ELIGIBLE_PATH,
    DEFAULT_LOADER,
    DEFAULT_QUERY_EMBEDDINGS_PATH,
    DEFAULT_REJECTED_PATH,
    DEFAULT_SECTION_EMBEDDINGS_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_NAME,
    SECTION_EMBEDDING_MAX_SEQ_LENGTHS,
)
from src.data_loader import load_candidates
from src.rejector import hard_reject_reasons
from src.retrieval import prepare_retrieval_artifacts, section_embedding_cache_paths


def resolve_cli_embedding_cache_dir(cache_dir: Path | None, legacy_cache_path: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir
    if legacy_cache_path is not None:
        return legacy_cache_path.parent if legacy_cache_path.suffix else legacy_cache_path
    return DEFAULT_SECTION_EMBEDDINGS_DIR


def load_or_create_eligible_candidates(
    candidates_path: Path,
    eligible_input: Path | None,
    eligible_out: Path,
    rejected_out: Path,
    loader: str,
) -> tuple[list[dict], int, Counter[str], Counter[str]]:
    if eligible_input is not None:
        candidates = list(load_candidates(eligible_input, loader="jsonl"))
        return candidates, len(candidates), Counter(), Counter()

    eligible_candidates: list[dict] = []
    reject_reasons: Counter[str] = Counter()
    all_reason_codes: Counter[str] = Counter()
    total_candidates = 0
    with eligible_out.open("w", encoding="utf-8") as eligible_handle, rejected_out.open(
        "w", encoding="utf-8"
    ) as rejected_handle:
        for candidate in load_candidates(candidates_path, loader=loader):
            total_candidates += 1
            reasons = hard_reject_reasons(candidate)
            if reasons:
                primary_reason = reasons[0]
                reject_reasons[primary_reason or "unknown"] += 1
                all_reason_codes.update(reasons)
                rejected_handle.write(
                    json.dumps(
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "primary_reason": primary_reason,
                            "reason_codes": reasons,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            eligible_candidates.append(candidate)
            eligible_handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    return eligible_candidates, total_candidates, reject_reasons, all_reason_codes


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare section embeddings, JD embeddings, and BM25 cache")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--eligible-input", type=Path, default=None)
    parser.add_argument("--eligible-out", type=Path, default=DEFAULT_ELIGIBLE_PATH)
    parser.add_argument("--rejected-out", type=Path, default=DEFAULT_REJECTED_PATH)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--embedding-batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Legacy combined-cache argument. Its parent directory is used for per-section caches.",
    )
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_EMBEDDINGS_PATH)
    parser.add_argument("--bm25-cache", type=Path, default=DEFAULT_BM25_CACHE_PATH)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow SentenceTransformers to download the embedding model if it is not cached.",
    )
    parser.add_argument(
        "--loader",
        choices=["jsonl", "pandas"],
        default=DEFAULT_LOADER,
        help="Use jsonl for final fast streaming, or pandas for chunked analysis/debugging.",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Only write eligible/rejected JSONL files and skip embedding/BM25 preparation.",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()
    embedding_cache_dir = resolve_cli_embedding_cache_dir(args.embedding_cache_dir, args.embedding_cache)
    eligible_candidates, total_candidates, reject_reasons, all_reason_codes = load_or_create_eligible_candidates(
        candidates_path=args.candidates,
        eligible_input=args.eligible_input,
        eligible_out=args.eligible_out,
        rejected_out=args.rejected_out,
        loader=args.loader,
    )

    if not args.filter_only:
        prepare_retrieval_artifacts(
            candidates=eligible_candidates,
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
            local_files_only=not args.allow_model_download,
            embedding_cache_dir=embedding_cache_dir,
            query_cache_path=args.query_cache,
            bm25_cache_path=args.bm25_cache,
        )

    if args.eligible_input is not None:
        print(f"Loaded eligible candidates from {args.eligible_input}")
    else:
        print(f"Total candidates: {total_candidates}")
        print(f"Hard rejected: {sum(reject_reasons.values())}")
        if reject_reasons:
            print("Top rejection reasons:")
            for reason, count in reject_reasons.most_common(10):
                print(f"  {reason}: {count}")
            print("Fake AI-fit rejection reasons:")
            for reason in [
                "fake_ai_fit_non_target_genai_explorer",
                "ai_keywords_not_supported_by_career",
                "ai_productivity_usage_not_ai_engineering",
                "junior_non_target_ai_keyword_profile",
                "business_profile_ai_wrapper",
            ]:
                print(f"  {reason}: {all_reason_codes.get(reason, 0)}")
        print(f"Wrote eligible JSONL to {args.eligible_out}")
        print(f"Wrote rejected JSONL to {args.rejected_out}")
    print(f"Eligible candidates: {len(eligible_candidates)}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Embedding fallback max seq length: {EMBEDDING_MAX_SEQ_LENGTH}")
    print(
        "Section max seq lengths: "
        + ", ".join(f"{section}={length}" for section, length in SECTION_EMBEDDING_MAX_SEQ_LENGTHS.items())
    )
    print(f"TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}")
    print(f"Torch CPU threads: {CPU_THREADS if CPU_THREADS is not None else 'not configured'}")
    print(f"Embedding local files only: {not args.allow_model_download}")
    print(f"Embedding cache dir: {embedding_cache_dir}")
    cache_paths = section_embedding_cache_paths(embedding_cache_dir) or {}
    for section, cache_path in cache_paths.items():
        print(f"  {section}: {cache_path}")
    print(f"Query embedding cache: {args.query_cache}")
    print(f"BM25 cache: {args.bm25_cache}")
    print(f"Elapsed seconds: {time.perf_counter() - started_at:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a valid top-100 candidate submission CSV."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from src.config import DEFAULT_CANDIDATES_PATH, DEFAULT_LOADER, DEFAULT_OUTPUT_PATH, TOP_N
from src.data_loader import load_candidates
from src.reasoning import build_reasoning
from src.rejector import hard_reject
from src.scorer import score_candidate
from src.traps import trap_penalty
from src.writer import SubmissionRow, write_submission


def rank_candidates(
    candidates_path: Path,
    output_path: Path,
    top_n: int = TOP_N,
    loader: str = DEFAULT_LOADER,
) -> None:
    started_at = time.perf_counter()
    ranked_rows: list[SubmissionRow] = []
    total_candidates = 0
    reject_reasons: Counter[str] = Counter()

    for candidate in load_candidates(candidates_path, loader=loader):
        total_candidates += 1
        rejected, reason = hard_reject(candidate)
        if rejected:
            reject_reasons[reason or "unknown"] += 1
            continue

        base_score, score_details = score_candidate(candidate)
        penalty, trap_notes = trap_penalty(candidate)
        score = min(1.0, max(0.0, base_score - penalty))
        reasoning = build_reasoning(candidate, score_details, trap_notes)
        ranked_rows.append(
            SubmissionRow(
                candidate_id=candidate["candidate_id"],
                score=score,
                reasoning=reasoning,
            )
        )

    ranked_rows.sort(key=lambda row: (-row.score, row.candidate_id))
    write_submission(ranked_rows[:top_n], output_path)
    print(f"Total candidates: {total_candidates}")
    print(f"Hard rejected: {sum(reject_reasons.values())}")
    print(f"Eligible scored: {len(ranked_rows)}")
    if reject_reasons:
        print("Top rejection reasons:")
        for reason, count in reject_reasons.most_common(10):
            print(f"  {reason}: {count}")
    print(f"Wrote {min(top_n, len(ranked_rows))} rows to {output_path}")
    print(f"Loader: {loader}")
    print(f"Elapsed seconds: {time.perf_counter() - started_at:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates and write submission.csv")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top", type=int, default=TOP_N)
    parser.add_argument(
        "--loader",
        choices=["jsonl", "pandas"],
        default=DEFAULT_LOADER,
        help="Use jsonl for final fast streaming, or pandas for chunked analysis/debugging.",
    )
    args = parser.parse_args()

    rank_candidates(args.candidates, args.out, args.top, args.loader)


if __name__ == "__main__":
    main()

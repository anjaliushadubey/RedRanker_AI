#!/usr/bin/env python3
"""Generate a valid top-100 candidate submission CSV.

This is a deliberately simple baseline pipeline. It proves that the project can
load the candidate file, score every row, sort deterministically, write the
required CSV, and pass the provided validator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_CANDIDATES_PATH, DEFAULT_OUTPUT_PATH, TOP_N
from src.data_loader import load_candidates
from src.reasoning import build_reasoning
from src.scorer import score_candidate
from src.traps import trap_penalty
from src.writer import SubmissionRow, write_submission


def rank_candidates(candidates_path: Path, output_path: Path, top_n: int = TOP_N) -> None:
    ranked_rows: list[SubmissionRow] = []

    for candidate in load_candidates(candidates_path):
        base_score, score_details = score_candidate(candidate)
        penalty, trap_notes = trap_penalty(candidate)
        score = max(0.0, base_score - penalty)
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
    print(f"Wrote {min(top_n, len(ranked_rows))} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates and write submission.csv")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top", type=int, default=TOP_N)
    args = parser.parse_args()

    rank_candidates(args.candidates, args.out, args.top)


if __name__ == "__main__":
    main()

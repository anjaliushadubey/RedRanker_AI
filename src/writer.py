"""Submission CSV writer."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubmissionRow:
    candidate_id: str
    score: float
    reasoning: str

def write_submission(rows: list[SubmissionRow], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        previous_score: float | None = None
        for rank, row in enumerate(rows, start=1):
            score = round(row.score, 6)
            if previous_score is not None and score >= previous_score:
                score = max(0.0, previous_score - 0.000001)
            previous_score = score
            writer.writerow([row.candidate_id, rank, f"{score:.6f}", row.reasoning])

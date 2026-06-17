"""Candidate loading helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def load_candidates(path: Path) -> Iterator[dict]:
    """Yield candidates from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

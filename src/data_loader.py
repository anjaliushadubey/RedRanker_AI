"""Candidate loading helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_LOADER, PANDAS_CHUNK_SIZE


def load_candidates_jsonl(path: Path) -> Iterator[dict]:
    """Yield candidates from JSONL using low-overhead Python streaming."""
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_candidate_chunks(path: Path, chunksize: int = PANDAS_CHUNK_SIZE) -> Iterator[list[dict]]:
    """Yield candidate records from a JSONL file in Pandas chunks.

    The dataset is JSONL, not one giant JSON array. Chunked Pandas loading gives
    us DataFrame-based ingestion while avoiding a full-file memory load.
    """
    for chunk in pd.read_json(path, lines=True, chunksize=chunksize):
        yield chunk.to_dict(orient="records")


def load_candidates_pandas(path: Path) -> Iterator[dict]:
    """Yield candidates from JSONL using chunked Pandas ingestion."""
    for records in load_candidate_chunks(path):
        yield from records


def load_candidates(path: Path, loader: str = DEFAULT_LOADER) -> Iterator[dict]:
    """Yield candidates using the selected loader."""
    if loader == "jsonl":
        yield from load_candidates_jsonl(path)
    elif loader == "pandas":
        yield from load_candidates_pandas(path)
    else:
        raise ValueError(f"Unknown loader {loader!r}; expected 'jsonl' or 'pandas'.")

"""Top-2000 retrieval stage.

This module builds section-wise candidate text, caches SentenceTransformer
embeddings, computes exact NumPy dense similarity by default, optionally runs
Qdrant named-vector search, computes BM25 sparse scores over full candidate
text, and blends them into a 70/30 hybrid score.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.config import (
    DEFAULT_SECTION_EMBEDDINGS_DIR,
    DEFAULT_DENSE_BACKEND,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_VERSION,
    EMBEDDING_LOCAL_FILES_ONLY,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_NAME,
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    JD_QUERY_BUILDER_VERSION,
    JD_SECTION_MAX_SEQ_LENGTHS,
    JD_SECTION_QUERIES,
    JD_SECTION_WEIGHTS,
    JD_TO_CANDIDATE_SECTION_WEIGHTS,
    QDRANT_COLLECTION_NAME,
    QDRANT_UPSERT_BATCH_SIZE,
    SECTION_EMBEDDING_BATCH_SIZES,
    SECTION_EMBEDDING_MAX_SEQ_LENGTHS,
    TEXT_BUILDER_VERSION,
    RETRIEVAL_SECTION_MAX_CHARS,
    RETRIEVAL_SECTION_WEIGHTS,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
SPARSE_TEXT_SECTIONS = ["title", "summary", "career_history", "skills", "education"]
SUMMARY_SPARSE_MAX_CHARS = 300


@dataclass(frozen=True)
class RetrievalResult:
    candidate_id: str
    hybrid_score: float
    dense_score: float
    bm25_score: float
    section_scores: dict[str, float]
    candidate: dict


def candidate_sections(candidate: dict, clip: bool = True) -> dict[str, str]:
    profile = candidate.get("profile", {})
    title_parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
    ]
    summary_parts = [
        profile.get("summary", ""),
        profile.get("current_industry", ""),
        profile.get("location", ""),
        profile.get("country", ""),
    ]
    career_parts = []
    for job in candidate.get("career_history", []):
        career_parts.extend(
            str(job.get(key) or "")
            for key in ["title", "company", "industry", "description"]
        )
    skills_parts = []
    for skill in candidate.get("skills", []):
        skills_parts.extend(
            str(skill.get(key) or "")
            for key in ["name", "proficiency"]
        )
    education_parts = []
    for education in candidate.get("education", []):
        education_parts.extend(
            str(education.get(key) or "")
            for key in ["institution", "degree", "field_of_study", "tier"]
        )
    sections = {
        "title": clean_text(title_parts),
        "summary": clean_text(summary_parts),
        "career_history": clean_text(career_parts),
        "skills": clean_text(skills_parts),
        "education": clean_text(education_parts),
    }
    if not clip:
        return sections
    return {
        section: text[: RETRIEVAL_SECTION_MAX_CHARS.get(section, SUMMARY_SPARSE_MAX_CHARS)]
        for section, text in sections.items()
    }


def full_candidate_text(sections: dict[str, str]) -> str:
    return " ".join(sections.get(section, "") for section in SPARSE_TEXT_SECTIONS)


def clean_text(parts: list[str]) -> str:
    return " ".join(part for part in parts if part).lower()


def section_weighted_dense_scores(
    section_texts: dict[str, list[str]],
    candidate_ids: list[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    collection_name: str = QDRANT_COLLECTION_NAME,
    local_files_only: bool = EMBEDDING_LOCAL_FILES_ONLY,
    embedding_cache_path: Path | None = None,
    query_cache_path: Path | None = None,
    qdrant_path: Path | None = None,
    dense_backend: str = DEFAULT_DENSE_BACKEND,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return section_weighted_dense_scores_from_embeddings(
        section_texts=section_texts,
        candidate_ids=candidate_ids,
        model_name=model_name,
        batch_size=batch_size,
        collection_name=collection_name,
        local_files_only=local_files_only,
        embedding_cache_path=embedding_cache_path,
        query_cache_path=query_cache_path,
        qdrant_path=qdrant_path,
        dense_backend=dense_backend,
    )


def section_weighted_dense_scores_from_embeddings(
    section_texts: dict[str, list[str]],
    candidate_ids: list[str],
    model_name: str,
    batch_size: int,
    collection_name: str,
    local_files_only: bool,
    embedding_cache_path: Path | None,
    query_cache_path: Path | None,
    qdrant_path: Path | None,
    dense_backend: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    candidate_count = len(next(iter(section_texts.values()), []))
    if candidate_count == 0:
        return np.array([], dtype=float), {
            section: np.array([], dtype=float) for section in JD_SECTION_WEIGHTS
        }

    candidate_ids_digest = ordered_values_hash(candidate_ids)
    section_vectors = load_section_vector_cache(
        embedding_cache_path=embedding_cache_path,
        candidate_ids=candidate_ids,
        candidate_ids_digest=candidate_ids_digest,
        model_name=model_name,
    )
    should_save_section_cache = section_vectors is None
    query_vectors = load_query_vector_cache(
        query_cache_path=query_cache_path,
        model_name=model_name,
    )
    should_save_query_cache = query_vectors is None

    if section_vectors is None or query_vectors is None:
        section_vectors, query_vectors = build_missing_embeddings(
            section_texts=section_texts,
            section_vectors=section_vectors,
            query_vectors=query_vectors,
            model_name=model_name,
            batch_size=batch_size,
            local_files_only=local_files_only,
        )
        if should_save_query_cache:
            save_query_vector_cache(
                query_cache_path=query_cache_path,
                model_name=model_name,
                query_vectors=query_vectors,
            )
        if should_save_section_cache:
            save_section_vector_cache(
                embedding_cache_path=embedding_cache_path,
                candidate_ids=candidate_ids,
                candidate_ids_digest=candidate_ids_digest,
                model_name=model_name,
                section_vectors=section_vectors,
            )

    vector_size = int(next(iter(query_vectors.values())).shape[0])
    index_metadata = build_cache_metadata(
        model_name=model_name,
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=candidate_count,
        embedding_dimension=vector_size,
    )
    if dense_backend == "numpy":
        print("Using NumPy dense backend")
        return numpy_section_weighted_dense_scores(
            section_vectors=section_vectors,
            query_vectors=query_vectors,
        )
    if dense_backend != "qdrant":
        raise ValueError(f"Unknown dense backend: {dense_backend}")
    try:
        return qdrant_dense_scores_from_vectors(
            section_vectors=section_vectors,
            query_vectors=query_vectors,
            collection_name=collection_name,
            qdrant_path=qdrant_path,
            index_metadata=index_metadata,
        )
    except Exception as exc:
        print(f"Qdrant dense search failed; falling back to NumPy similarity. Reason: {exc}")
        return numpy_section_weighted_dense_scores(
            section_vectors=section_vectors,
            query_vectors=query_vectors,
        )


def section_names() -> list[str]:
    return list(RETRIEVAL_SECTION_WEIGHTS.keys())


def jd_section_names() -> list[str]:
    return list(JD_SECTION_WEIGHTS.keys())


def validate_jd_dense_config() -> None:
    candidate_sections = set(section_names())
    jd_sections = set(jd_section_names())
    missing_queries = jd_sections.difference(JD_SECTION_QUERIES)
    missing_mappings = jd_sections.difference(JD_TO_CANDIDATE_SECTION_WEIGHTS)
    if missing_queries:
        raise ValueError(f"Missing JD section queries: {', '.join(sorted(missing_queries))}")
    if missing_mappings:
        raise ValueError(f"Missing JD-to-candidate section mappings: {', '.join(sorted(missing_mappings))}")
    jd_weight_total = sum(float(weight) for weight in JD_SECTION_WEIGHTS.values())
    if abs(jd_weight_total - 1.0) > 1e-6:
        raise ValueError(f"JD section weights must sum to 1.0, got {jd_weight_total:.6f}")
    for jd_section, mapping in JD_TO_CANDIDATE_SECTION_WEIGHTS.items():
        unknown_sections = set(mapping).difference(candidate_sections)
        if unknown_sections:
            unknown = ", ".join(sorted(unknown_sections))
            raise ValueError(f"JD section {jd_section} maps to unknown candidate sections: {unknown}")
        mapping_total = sum(float(weight) for weight in mapping.values())
        if abs(mapping_total - 1.0) > 1e-6:
            raise ValueError(
                f"JD mapping weights for {jd_section} must sum to 1.0, got {mapping_total:.6f}"
            )


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ordered_values_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def query_text_hash() -> str:
    ordered_queries = {section: JD_SECTION_QUERIES[section] for section in jd_section_names()}
    return sha256_text(stable_json(ordered_queries))


def section_weights_hash() -> str:
    ordered_weights = {section: RETRIEVAL_SECTION_WEIGHTS[section] for section in section_names()}
    return sha256_text(stable_json(ordered_weights))


def jd_section_weights_hash() -> str:
    ordered_weights = {section: JD_SECTION_WEIGHTS[section] for section in jd_section_names()}
    return sha256_text(stable_json(ordered_weights))


def jd_candidate_mapping_hash() -> str:
    ordered_mapping = {
        section: JD_TO_CANDIDATE_SECTION_WEIGHTS[section]
        for section in jd_section_names()
    }
    return sha256_text(stable_json(ordered_mapping))


def section_limits_hash() -> str:
    ordered_limits = {section: RETRIEVAL_SECTION_MAX_CHARS[section] for section in section_names()}
    return sha256_text(stable_json(ordered_limits))


def section_max_seq_length(section: str) -> int:
    return SECTION_EMBEDDING_MAX_SEQ_LENGTHS.get(section, EMBEDDING_MAX_SEQ_LENGTH)


def jd_section_max_seq_length(section: str) -> int:
    return JD_SECTION_MAX_SEQ_LENGTHS.get(section, EMBEDDING_MAX_SEQ_LENGTH)


def section_max_seq_lengths_hash() -> str:
    ordered_lengths = {section: section_max_seq_length(section) for section in section_names()}
    return sha256_text(stable_json(ordered_lengths))


def jd_section_max_seq_lengths_hash() -> str:
    ordered_lengths = {section: jd_section_max_seq_length(section) for section in jd_section_names()}
    return sha256_text(stable_json(ordered_lengths))


def resolve_section_embedding_cache_dir(
    embedding_cache_dir: Path | None = None,
    legacy_embedding_cache_path: Path | None = None,
) -> Path | None:
    if embedding_cache_dir is not None:
        return embedding_cache_dir
    if legacy_embedding_cache_path is not None:
        return legacy_embedding_cache_path.parent if legacy_embedding_cache_path.suffix else legacy_embedding_cache_path
    return DEFAULT_SECTION_EMBEDDINGS_DIR


def section_embedding_cache_paths(cache_dir: Path | None) -> dict[str, Path] | None:
    if cache_dir is None:
        return None
    return {section: cache_dir / f"{section}_embeddings.npz" for section in section_names()}


def build_section_cache_metadata(
    model_name: str,
    candidate_ids_digest: str,
    candidate_count: int,
    section_name: str,
    section_text_hash: str,
    embedding_dimension: int,
) -> dict[str, object]:
    return {
        "cache_version": EMBEDDING_CACHE_VERSION,
        "text_builder_version": TEXT_BUILDER_VERSION,
        "model_name": model_name,
        "max_seq_length": section_max_seq_length(section_name),
        "candidate_ids_hash": candidate_ids_digest,
        "candidate_count": candidate_count,
        "section_name": section_name,
        "section_text_hash": section_text_hash,
        "embedding_dim": embedding_dimension,
    }


def build_cache_metadata(
    model_name: str,
    candidate_ids_digest: str | None,
    candidate_count: int | None,
    embedding_dimension: int,
) -> dict[str, object]:
    return {
        "cache_version": EMBEDDING_CACHE_VERSION,
        "text_builder_version": TEXT_BUILDER_VERSION,
        "model_name": model_name,
        "section_names": section_names(),
        "section_weights_hash": section_weights_hash(),
        "section_limits_hash": section_limits_hash(),
        "section_max_seq_lengths_hash": section_max_seq_lengths_hash(),
        "candidate_ids_hash": candidate_ids_digest,
        "candidate_count": candidate_count,
        "embedding_dimension": embedding_dimension,
        "embedding_max_seq_length": EMBEDDING_MAX_SEQ_LENGTH,
    }


def build_query_cache_metadata(
    model_name: str,
    embedding_dimension: int,
) -> dict[str, object]:
    return {
        "cache_version": EMBEDDING_CACHE_VERSION,
        "text_builder_version": TEXT_BUILDER_VERSION,
        "query_builder_version": JD_QUERY_BUILDER_VERSION,
        "model_name": model_name,
        "candidate_section_names": section_names(),
        "jd_section_names": jd_section_names(),
        "jd_section_weights_hash": jd_section_weights_hash(),
        "jd_to_candidate_section_weights_hash": jd_candidate_mapping_hash(),
        "jd_section_max_seq_lengths_hash": jd_section_max_seq_lengths_hash(),
        "query_text_hash": query_text_hash(),
        "embedding_dimension": embedding_dimension,
    }


def build_bm25_metadata(candidate_ids_digest: str, candidate_count: int) -> dict[str, object]:
    return {
        "cache_version": EMBEDDING_CACHE_VERSION,
        "text_builder_version": TEXT_BUILDER_VERSION,
        "section_names": section_names(),
        "jd_section_names": jd_section_names(),
        "section_limits_hash": section_limits_hash(),
        "query_text_hash": query_text_hash(),
        "candidate_ids_hash": candidate_ids_digest,
        "candidate_count": candidate_count,
        "bm25_query_hash": sha256_text(" ".join(JD_SECTION_QUERIES.values())),
    }


def cache_metadata_matches(actual: dict[str, object], expected: dict[str, object]) -> bool:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            return False
    return True


def load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, object] | None:
    if "metadata_json" not in data:
        return None
    return json.loads(str(data["metadata_json"].item()))


def load_section_vector_cache(
    embedding_cache_path: Path | None,
    candidate_ids: list[str],
    candidate_ids_digest: str,
    model_name: str,
) -> dict[str, np.ndarray] | None:
    if embedding_cache_path is None or not embedding_cache_path.exists():
        return None

    data = np.load(embedding_cache_path, allow_pickle=False)
    required_sections = section_names()
    if not all(f"{section}_vectors" in data for section in required_sections):
        print(f"Ignoring stale embedding cache at {embedding_cache_path}")
        return None

    first_vectors = np.asarray(data[f"{required_sections[0]}_vectors"], dtype=np.float32)
    if first_vectors.ndim != 2:
        print(f"Ignoring malformed embedding cache at {embedding_cache_path}")
        return None

    expected_metadata = build_cache_metadata(
        model_name=model_name,
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=len(candidate_ids),
        embedding_dimension=int(first_vectors.shape[1]),
    )
    cached_metadata = load_metadata(data)
    if cached_metadata is None or not cache_metadata_matches(cached_metadata, expected_metadata):
        print(f"Ignoring stale embedding cache at {embedding_cache_path}")
        return None

    section_vectors = {
        section: np.asarray(data[f"{section}_vectors"], dtype=np.float32)
        for section in required_sections
    }
    if any(vectors.shape != first_vectors.shape for vectors in section_vectors.values()):
        print(f"Ignoring malformed embedding cache at {embedding_cache_path}")
        return None

    print(f"Loaded section embeddings from {embedding_cache_path}")
    return section_vectors


def save_section_vector_cache(
    embedding_cache_path: Path | None,
    candidate_ids: list[str],
    candidate_ids_digest: str,
    model_name: str,
    section_vectors: dict[str, np.ndarray],
) -> None:
    if embedding_cache_path is None:
        return
    embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
    first_vectors = np.asarray(section_vectors[section_names()[0]], dtype=np.float32)
    metadata = build_cache_metadata(
        model_name=model_name,
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=len(candidate_ids),
        embedding_dimension=int(first_vectors.shape[1]),
    )
    payload: dict[str, np.ndarray] = {
        "metadata_json": np.array(stable_json(metadata)),
        "candidate_ids": np.array(candidate_ids, dtype=str),
    }
    for section, vectors in section_vectors.items():
        payload[f"{section}_vectors"] = np.asarray(vectors, dtype=np.float32)
    np.savez(embedding_cache_path, **payload)
    print(f"Saved section embeddings to {embedding_cache_path}")


def load_single_section_vector_cache(
    cache_path: Path | None,
    model_name: str,
    candidate_ids_digest: str,
    candidate_count: int,
    section_name: str,
    section_text_hash: str,
) -> np.ndarray | None:
    if cache_path is None or not cache_path.exists():
        return None

    data = np.load(cache_path, allow_pickle=False)
    if "vectors" not in data:
        print(f"Ignoring stale {section_name} embedding cache at {cache_path}")
        return None

    vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != candidate_count:
        print(f"Ignoring malformed {section_name} embedding cache at {cache_path}")
        return None

    expected_metadata = build_section_cache_metadata(
        model_name=model_name,
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=candidate_count,
        section_name=section_name,
        section_text_hash=section_text_hash,
        embedding_dimension=int(vectors.shape[1]),
    )
    cached_metadata = load_metadata(data)
    if cached_metadata is None or not cache_metadata_matches(cached_metadata, expected_metadata):
        print(f"Ignoring stale {section_name} embedding cache at {cache_path}")
        return None

    print(f"Loaded {section_name} embeddings from {cache_path}")
    return vectors


def save_single_section_vector_cache(
    cache_path: Path | None,
    vectors: np.ndarray,
    model_name: str,
    candidate_ids_digest: str,
    candidate_count: int,
    section_name: str,
    section_text_hash: str,
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    metadata = build_section_cache_metadata(
        model_name=model_name,
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=candidate_count,
        section_name=section_name,
        section_text_hash=section_text_hash,
        embedding_dimension=int(vectors.shape[1]),
    )
    np.savez(
        cache_path,
        metadata_json=np.array(stable_json(metadata)),
        vectors=vectors,
    )
    print(f"Saved {section_name} embeddings to {cache_path}")


def load_section_vector_caches(
    section_texts: dict[str, list[str]],
    candidate_ids_digest: str,
    candidate_count: int,
    model_name: str,
    embedding_cache_dir: Path | None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    vectors_by_section: dict[str, np.ndarray] = {}
    missing_sections: list[str] = []
    cache_paths = section_embedding_cache_paths(embedding_cache_dir)
    for section in section_names():
        section_text_hash = ordered_values_hash(section_texts[section])
        cache_path = cache_paths[section] if cache_paths is not None else None
        vectors = load_single_section_vector_cache(
            cache_path=cache_path,
            model_name=model_name,
            candidate_ids_digest=candidate_ids_digest,
            candidate_count=candidate_count,
            section_name=section,
            section_text_hash=section_text_hash,
        )
        if vectors is None:
            missing_sections.append(section)
        else:
            vectors_by_section[section] = vectors
    return vectors_by_section, missing_sections


def load_query_vector_cache(
    query_cache_path: Path | None,
    model_name: str,
) -> dict[str, np.ndarray] | None:
    validate_jd_dense_config()
    if query_cache_path is None or not query_cache_path.exists():
        return None

    data = np.load(query_cache_path, allow_pickle=False)
    required_sections = jd_section_names()
    if not all(f"{section}_query" in data for section in required_sections):
        print(f"Ignoring stale query embedding cache at {query_cache_path}")
        return None

    first_query = np.asarray(data[f"{required_sections[0]}_query"], dtype=np.float32)
    if first_query.ndim != 1:
        print(f"Ignoring malformed query embedding cache at {query_cache_path}")
        return None

    expected_metadata = build_query_cache_metadata(
        model_name=model_name,
        embedding_dimension=int(first_query.shape[0]),
    )
    cached_metadata = load_metadata(data)
    if cached_metadata is None or not cache_metadata_matches(cached_metadata, expected_metadata):
        print(f"Ignoring stale query embedding cache at {query_cache_path}")
        return None

    query_vectors = {
        section: np.asarray(data[f"{section}_query"], dtype=np.float32)
        for section in required_sections
    }
    print(f"Loaded JD query embeddings from {query_cache_path}")
    return query_vectors


def save_query_vector_cache(
    query_cache_path: Path | None,
    model_name: str,
    query_vectors: dict[str, np.ndarray],
) -> None:
    if query_cache_path is None:
        return
    query_cache_path.parent.mkdir(parents=True, exist_ok=True)
    validate_jd_dense_config()
    first_query = np.asarray(query_vectors[jd_section_names()[0]], dtype=np.float32)
    metadata = build_query_cache_metadata(
        model_name=model_name,
        embedding_dimension=int(first_query.shape[0]),
    )
    payload: dict[str, np.ndarray] = {
        "metadata_json": np.array(stable_json(metadata)),
    }
    for section, vector in query_vectors.items():
        payload[f"{section}_query"] = np.asarray(vector, dtype=np.float32)
    np.savez(query_cache_path, **payload)
    print(f"Saved JD query embeddings to {query_cache_path}")


def build_missing_embeddings(
    section_texts: dict[str, list[str]],
    section_vectors: dict[str, np.ndarray] | None,
    query_vectors: dict[str, np.ndarray] | None,
    model_name: str,
    batch_size: int,
    local_files_only: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    candidate_count = len(next(iter(section_texts.values()), []))
    encode_inputs: list[str] = []
    jobs: list[tuple[str, str, int | None]] = []

    if query_vectors is None:
        query_vectors = {}
        for section in jd_section_names():
            jobs.append(("query", section, None))
            encode_inputs.append(JD_SECTION_QUERIES[section])

    if section_vectors is None:
        for section in section_names():
            for index, text in enumerate(section_texts[section]):
                if text.strip():
                    jobs.append(("section", section, index))
                    encode_inputs.append(text)

    encoded = np.empty((0, 0), dtype=np.float32)
    if encode_inputs:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, local_files_only=local_files_only)
        model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
        encoded = encode_deduplicated_texts(model, encode_inputs, batch_size=batch_size)

    position = 0
    if query_vectors == {}:
        for section in jd_section_names():
            query_vectors[section] = encoded[position]
            position += 1

    embedding_dimension = infer_embedding_dimension(
        query_vectors=query_vectors,
        section_vectors=section_vectors,
        encoded=encoded,
    )

    if section_vectors is None:
        section_vectors = {
            section: np.zeros((candidate_count, embedding_dimension), dtype=np.float32)
            for section in section_names()
        }
        for job_kind, section, index in jobs[position:]:
            if job_kind == "section" and index is not None:
                section_vectors[section][index] = encoded[position]
            position += 1

    return section_vectors, query_vectors


def encode_deduplicated_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    unique_texts: list[str] = []
    text_to_index: dict[str, int] = {}
    inverse_indices: list[int] = []
    for text in texts:
        key = text.strip()
        if key not in text_to_index:
            text_to_index[key] = len(unique_texts)
            unique_texts.append(key)
        inverse_indices.append(text_to_index[key])

    unique_vectors = encode_texts(model, unique_texts, batch_size=batch_size)
    return unique_vectors[np.array(inverse_indices, dtype=int)]


def encode_section_texts(
    model: "SentenceTransformer",
    section_name: str,
    texts: list[str],
    embedding_dimension: int,
) -> np.ndarray:
    vectors = np.zeros((len(texts), embedding_dimension), dtype=np.float32)
    unique_texts: list[str] = []
    text_to_index: dict[str, int] = {}
    inverse_indices: list[int] = []
    non_empty_positions: list[int] = []

    for index, text in enumerate(texts):
        normalized = text.strip()
        if not normalized:
            continue
        if normalized not in text_to_index:
            text_to_index[normalized] = len(unique_texts)
            unique_texts.append(normalized)
        non_empty_positions.append(index)
        inverse_indices.append(text_to_index[normalized])

    if not unique_texts:
        print(f"Encoding {section_name}: all texts empty; using zero vectors")
        return vectors

    sorted_unique_indices = sorted(range(len(unique_texts)), key=lambda item: len(unique_texts[item]))
    sorted_unique_texts = [unique_texts[index] for index in sorted_unique_indices]
    batch_size = SECTION_EMBEDDING_BATCH_SIZES.get(section_name, EMBEDDING_BATCH_SIZE)
    model.max_seq_length = section_max_seq_length(section_name)
    print(
        f"Encoding {section_name}: {len(non_empty_positions)} non-empty rows, "
        f"{len(unique_texts)} unique texts, batch size {batch_size}, "
        f"max seq length {model.max_seq_length}"
    )
    sorted_vectors = encode_texts(model, sorted_unique_texts, batch_size=batch_size)
    unique_vectors = np.zeros((len(unique_texts), embedding_dimension), dtype=np.float32)
    for sorted_index, original_unique_index in enumerate(sorted_unique_indices):
        unique_vectors[original_unique_index] = sorted_vectors[sorted_index]

    mapped_vectors = unique_vectors[np.asarray(inverse_indices, dtype=int)]
    vectors[np.asarray(non_empty_positions, dtype=int)] = mapped_vectors
    return vectors


def encode_query_vectors(model: "SentenceTransformer") -> dict[str, np.ndarray]:
    validate_jd_dense_config()
    query_vectors: dict[str, np.ndarray] = {}
    for section in jd_section_names():
        model.max_seq_length = jd_section_max_seq_length(section)
        vector = encode_texts(
            model,
            [JD_SECTION_QUERIES[section]],
            batch_size=1,
        )[0]
        query_vectors[section] = vector
    return query_vectors


def load_sentence_transformer(model_name: str, local_files_only: bool) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
    return model


def infer_embedding_dimension(
    query_vectors: dict[str, np.ndarray] | None,
    section_vectors: dict[str, np.ndarray] | None,
    encoded: np.ndarray,
) -> int:
    if query_vectors:
        return int(next(iter(query_vectors.values())).shape[0])
    if section_vectors:
        return int(next(iter(section_vectors.values())).shape[1])
    if encoded.size:
        return int(encoded.shape[1])
    raise ValueError("Cannot infer embedding dimension from empty embeddings")


def qdrant_dense_scores_from_vectors(
    section_vectors: dict[str, np.ndarray],
    query_vectors: dict[str, np.ndarray],
    collection_name: str,
    qdrant_path: Path | None,
    index_metadata: dict[str, object],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    validate_jd_dense_config()
    client: QdrantClient | None = None
    try:
        client = make_qdrant_client(qdrant_path)
        reused_index = ensure_qdrant_collection(
            client=client,
            collection_name=collection_name,
            vector_size=int(index_metadata["embedding_dimension"]),
            section_vectors=section_vectors,
            qdrant_path=qdrant_path,
            index_metadata=index_metadata,
        )
        if reused_index:
            print("Reusing existing Qdrant named-vector index")
        else:
            print("Built Qdrant named-vector index")

        candidate_count = len(next(iter(section_vectors.values())))
        dense_total: np.ndarray | None = None
        jd_scores: dict[str, np.ndarray] = {}
        for jd_section, jd_weight in JD_SECTION_WEIGHTS.items():
            mapped_total: np.ndarray | None = None
            for candidate_section, candidate_weight in JD_TO_CANDIDATE_SECTION_WEIGHTS[jd_section].items():
                scores = qdrant_section_scores(
                    client=client,
                    collection_name=collection_name,
                    section=candidate_section,
                    query_vector=query_vectors[jd_section],
                    candidate_count=candidate_count,
                )
                normalized_scores = normalize_scores(scores)
                mapped_total = (
                    normalized_scores * candidate_weight
                    if mapped_total is None
                    else mapped_total + (normalized_scores * candidate_weight)
                )
            if mapped_total is None:
                mapped_total = np.zeros(candidate_count, dtype=float)
            normalized_jd_scores = normalize_scores(mapped_total)
            jd_scores[jd_section] = normalized_jd_scores * jd_weight
            dense_total = (
                normalized_jd_scores * jd_weight
                if dense_total is None
                else dense_total + (normalized_jd_scores * jd_weight)
            )
        if dense_total is None:
            dense_total = np.array([], dtype=float)
        return normalize_scores(dense_total), jd_scores
    finally:
        if client is not None:
            client.close()


def make_qdrant_client(qdrant_path: Path | None) -> QdrantClient:
    from qdrant_client import QdrantClient

    if qdrant_path is None:
        return QdrantClient(location=":memory:")
    qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


def ensure_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    section_vectors: dict[str, np.ndarray],
    qdrant_path: Path | None,
    index_metadata: dict[str, object],
) -> bool:
    metadata_path = qdrant_metadata_path(qdrant_path, collection_name)
    if (
        qdrant_path is not None
        and metadata_path.exists()
        and qdrant_collection_exists(client, collection_name)
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cache_metadata_matches(metadata, index_metadata):
            return True

    if qdrant_collection_exists(client, collection_name):
        client.delete_collection(collection_name=collection_name)

    from qdrant_client.models import Distance, VectorParams

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            section: VectorParams(size=vector_size, distance=Distance.COSINE)
            for section in RETRIEVAL_SECTION_WEIGHTS
        },
    )
    upsert_section_vectors(
        client=client,
        collection_name=collection_name,
        section_vectors=section_vectors,
    )
    if qdrant_path is not None:
        metadata_path.write_text(stable_json(index_metadata), encoding="utf-8")
    return False


def qdrant_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        return bool(client.collection_exists(collection_name=collection_name))
    except AttributeError:
        try:
            client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False


def qdrant_metadata_path(qdrant_path: Path | None, collection_name: str) -> Path:
    base_path = qdrant_path or Path(".")
    return base_path / f"{collection_name}.metadata.json"


def numpy_section_weighted_dense_scores(
    section_vectors: dict[str, np.ndarray],
    query_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    validate_jd_dense_config()
    dense_total: np.ndarray | None = None
    jd_scores: dict[str, np.ndarray] = {}
    candidate_count = len(next(iter(section_vectors.values()), []))
    for jd_section, jd_weight in JD_SECTION_WEIGHTS.items():
        mapped_total: np.ndarray | None = None
        for candidate_section, candidate_weight in JD_TO_CANDIDATE_SECTION_WEIGHTS[jd_section].items():
            scores = np.asarray(section_vectors[candidate_section] @ query_vectors[jd_section], dtype=float)
            normalized_scores = normalize_scores(scores)
            mapped_total = (
                normalized_scores * candidate_weight
                if mapped_total is None
                else mapped_total + (normalized_scores * candidate_weight)
            )
        if mapped_total is None:
            mapped_total = np.zeros(candidate_count, dtype=float)
        normalized_jd_scores = normalize_scores(mapped_total)
        jd_scores[jd_section] = normalized_jd_scores * jd_weight
        dense_total = (
            normalized_jd_scores * jd_weight
            if dense_total is None
            else dense_total + (normalized_jd_scores * jd_weight)
        )
    if dense_total is None:
        dense_total = np.array([], dtype=float)
    return normalize_scores(dense_total), jd_scores


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    normalized_texts = [text if text.strip() else " " for text in texts]
    vectors = model.encode(
        normalized_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def upsert_section_vectors(
    client: QdrantClient,
    collection_name: str,
    section_vectors: dict[str, np.ndarray],
) -> None:
    candidate_count = len(next(iter(section_vectors.values())))
    from qdrant_client.models import PointStruct

    for start in range(0, candidate_count, QDRANT_UPSERT_BATCH_SIZE):
        end = min(start + QDRANT_UPSERT_BATCH_SIZE, candidate_count)
        points = []
        for index in range(start, end):
            vectors = {
                section: values[index].tolist()
                for section, values in section_vectors.items()
            }
            points.append(PointStruct(id=index, vector=vectors, payload={"row_index": index}))
        client.upsert(collection_name=collection_name, points=points, wait=True)


def qdrant_section_scores(
    client: QdrantClient,
    collection_name: str,
    section: str,
    query_vector: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        using=section,
        limit=candidate_count,
        with_payload=False,
        with_vectors=False,
    )
    scores = np.zeros(candidate_count, dtype=float)
    for point in response.points:
        scores[int(point.id)] = float(point.score)
    return scores


def bm25_scores(texts: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokenized_docs = [tokenize(text) for text in texts]
    query_terms = tokenize(query)
    if not tokenized_docs:
        return np.array([], dtype=float)
    doc_freq: Counter[str] = Counter()
    doc_term_counts: list[Counter[str]] = []
    doc_lengths = []
    for tokens in tokenized_docs:
        counts = Counter(tokens)
        doc_term_counts.append(counts)
        doc_lengths.append(len(tokens))
        doc_freq.update(counts.keys())
    avgdl = sum(doc_lengths) / max(1, len(doc_lengths))
    doc_count = len(tokenized_docs)
    scores = []
    for counts, doc_len in zip(doc_term_counts, doc_lengths):
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if freq == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
            denom = freq + k1 * (1 - b + b * (doc_len / max(avgdl, 1e-9)))
            score += idf * ((freq * (k1 + 1)) / denom)
        scores.append(score)
    return normalize_scores(np.array(scores, dtype=float))


def cached_bm25_scores(
    texts: list[str],
    candidate_ids_digest: str,
    bm25_cache_path: Path | None,
) -> np.ndarray:
    metadata = build_bm25_metadata(
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=len(texts),
    )
    if bm25_cache_path is not None and bm25_cache_path.exists():
        data = np.load(bm25_cache_path, allow_pickle=False)
        cached_metadata = load_metadata(data)
        if (
            cached_metadata is not None
            and cache_metadata_matches(cached_metadata, metadata)
            and "bm25_scores" in data
        ):
            scores = np.asarray(data["bm25_scores"], dtype=float)
            if scores.shape == (len(texts),):
                print(f"Loaded BM25 scores from {bm25_cache_path}")
                return scores
        print(f"Ignoring stale BM25 cache at {bm25_cache_path}")

    scores = bm25_scores(texts, " ".join(JD_SECTION_QUERIES.values()))
    if bm25_cache_path is not None:
        bm25_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            bm25_cache_path,
            metadata_json=np.array(stable_json(metadata)),
            bm25_scores=np.asarray(scores, dtype=np.float32),
        )
        print(f"Saved BM25 scores to {bm25_cache_path}")
    return scores


def load_bm25_score_cache(
    candidate_ids_digest: str,
    candidate_count: int,
    bm25_cache_path: Path | None,
) -> np.ndarray | None:
    if bm25_cache_path is None or not bm25_cache_path.exists():
        return None
    metadata = build_bm25_metadata(
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=candidate_count,
    )
    data = np.load(bm25_cache_path, allow_pickle=False)
    cached_metadata = load_metadata(data)
    if (
        cached_metadata is not None
        and cache_metadata_matches(cached_metadata, metadata)
        and "bm25_scores" in data
    ):
        scores = np.asarray(data["bm25_scores"], dtype=float)
        if scores.shape == (candidate_count,):
            print(f"Loaded BM25 scores from {bm25_cache_path}")
            return scores
    print(f"Ignoring stale BM25 cache at {bm25_cache_path}")
    return None


def save_bm25_score_cache(
    texts: list[str],
    candidate_ids_digest: str,
    bm25_cache_path: Path | None,
) -> np.ndarray:
    scores = bm25_scores(texts, " ".join(JD_SECTION_QUERIES.values()))
    if bm25_cache_path is not None:
        bm25_cache_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = build_bm25_metadata(
            candidate_ids_digest=candidate_ids_digest,
            candidate_count=len(texts),
        )
        np.savez(
            bm25_cache_path,
            metadata_json=np.array(stable_json(metadata)),
            bm25_scores=np.asarray(scores, dtype=np.float32),
        )
        print(f"Saved BM25 scores to {bm25_cache_path}")
    return scores


def prepare_retrieval_artifacts(
    candidates: list[dict],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    local_files_only: bool = EMBEDDING_LOCAL_FILES_ONLY,
    embedding_cache_path: Path | None = None,
    embedding_cache_dir: Path | None = None,
    query_cache_path: Path | None = None,
    bm25_cache_path: Path | None = None,
) -> None:
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    candidate_ids_digest = ordered_values_hash(candidate_ids)
    candidate_count = len(candidates)
    resolved_cache_dir = resolve_section_embedding_cache_dir(
        embedding_cache_dir=embedding_cache_dir,
        legacy_embedding_cache_path=embedding_cache_path,
    )
    sections_by_candidate = [candidate_sections(candidate, clip=True) for candidate in candidates]
    section_texts = {
        section: [sections[section] for sections in sections_by_candidate]
        for section in RETRIEVAL_SECTION_WEIGHTS
    }

    section_vectors, missing_sections = load_section_vector_caches(
        section_texts=section_texts,
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=candidate_count,
        model_name=model_name,
        embedding_cache_dir=resolved_cache_dir,
    )
    query_vectors = load_query_vector_cache(
        query_cache_path=query_cache_path,
        model_name=model_name,
    )

    if missing_sections or query_vectors is None:
        model = load_sentence_transformer(model_name, local_files_only=local_files_only)
        if query_vectors is None:
            query_vectors = encode_query_vectors(model)
            if query_cache_path is not None:
                save_query_vector_cache(
                    query_cache_path=query_cache_path,
                    model_name=model_name,
                    query_vectors=query_vectors,
                )

        embedding_dimension = int(next(iter(query_vectors.values())).shape[0])
        cache_paths = section_embedding_cache_paths(resolved_cache_dir)
        for section in missing_sections:
            vectors = encode_section_texts(
                model=model,
                section_name=section,
                texts=section_texts[section],
                embedding_dimension=embedding_dimension,
            )
            section_vectors[section] = vectors
            if cache_paths is not None:
                save_single_section_vector_cache(
                    cache_path=cache_paths[section],
                    vectors=vectors,
                    model_name=model_name,
                    candidate_ids_digest=candidate_ids_digest,
                    candidate_count=candidate_count,
                    section_name=section,
                    section_text_hash=ordered_values_hash(section_texts[section]),
                )
    if load_bm25_score_cache(candidate_ids_digest, candidate_count, bm25_cache_path) is None:
        full_texts = [
            full_candidate_text(candidate_sections(candidate, clip=False))
            for candidate in candidates
        ]
        save_bm25_score_cache(
            texts=full_texts,
            candidate_ids_digest=candidate_ids_digest,
            bm25_cache_path=bm25_cache_path,
        )


def retrieve_top_candidates_from_cache(
    candidates: list[dict],
    top_n: int,
    model_name: str = EMBEDDING_MODEL_NAME,
    collection_name: str = QDRANT_COLLECTION_NAME,
    embedding_cache_path: Path | None = None,
    embedding_cache_dir: Path | None = None,
    query_cache_path: Path | None = None,
    bm25_cache_path: Path | None = None,
    qdrant_path: Path | None = None,
    dense_backend: str = DEFAULT_DENSE_BACKEND,
) -> list[RetrievalResult]:
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    candidate_ids_digest = ordered_values_hash(candidate_ids)
    candidate_count = len(candidates)
    resolved_cache_dir = resolve_section_embedding_cache_dir(
        embedding_cache_dir=embedding_cache_dir,
        legacy_embedding_cache_path=embedding_cache_path,
    )
    sections_by_candidate = [candidate_sections(candidate, clip=True) for candidate in candidates]
    section_texts = {
        section: [sections[section] for sections in sections_by_candidate]
        for section in RETRIEVAL_SECTION_WEIGHTS
    }
    section_vectors, missing_sections = load_section_vector_caches(
        section_texts=section_texts,
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=candidate_count,
        model_name=model_name,
        embedding_cache_dir=resolved_cache_dir,
    )
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise RuntimeError(f"Section embedding caches missing or stale for: {missing}. Run prepare_embeddings.py first.")

    query_vectors = load_query_vector_cache(
        query_cache_path=query_cache_path,
        model_name=model_name,
    )
    if query_vectors is None:
        raise RuntimeError("JD query embedding cache missing or stale. Run prepare_embeddings.py first.")

    bm25 = load_bm25_score_cache(
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=candidate_count,
        bm25_cache_path=bm25_cache_path,
    )
    if bm25 is None:
        raise RuntimeError("BM25 cache missing or stale. Run prepare_embeddings.py first.")

    vector_size = int(next(iter(query_vectors.values())).shape[0])
    index_metadata = build_cache_metadata(
        model_name=model_name,
        candidate_ids_digest=candidate_ids_digest,
        candidate_count=candidate_count,
        embedding_dimension=vector_size,
    )
    if dense_backend == "numpy":
        print("Using NumPy dense backend")
        dense_scores, section_scores = numpy_section_weighted_dense_scores(
            section_vectors=section_vectors,
            query_vectors=query_vectors,
        )
    elif dense_backend == "qdrant":
        try:
            dense_scores, section_scores = qdrant_dense_scores_from_vectors(
                section_vectors=section_vectors,
                query_vectors=query_vectors,
                collection_name=collection_name,
                qdrant_path=qdrant_path,
                index_metadata=index_metadata,
            )
        except Exception as exc:
            print(f"Qdrant dense search failed; falling back to NumPy similarity. Reason: {exc}")
            dense_scores, section_scores = numpy_section_weighted_dense_scores(
                section_vectors=section_vectors,
                query_vectors=query_vectors,
            )
    else:
        raise ValueError(f"Unknown dense backend: {dense_backend}")

    hybrid = (HYBRID_DENSE_WEIGHT * dense_scores) + (HYBRID_BM25_WEIGHT * bm25)
    results = []
    for index, candidate in enumerate(candidates):
        per_section = {
            section: float(scores[index])
            for section, scores in section_scores.items()
        }
        results.append(
            RetrievalResult(
                candidate_id=str(candidate["candidate_id"]),
                hybrid_score=float(hybrid[index]),
                dense_score=float(dense_scores[index]),
                bm25_score=float(bm25[index]),
                section_scores=per_section,
                candidate=candidate,
            )
        )
    results.sort(key=lambda result: (-result.hybrid_score, result.candidate_id))
    return results[:top_n]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    low = float(scores.min())
    high = float(scores.max())
    if high <= low:
        return np.zeros_like(scores, dtype=float)
    return (scores - low) / (high - low)


def retrieve_top_candidates(
    candidates: list[dict],
    top_n: int,
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    collection_name: str = QDRANT_COLLECTION_NAME,
    local_files_only: bool = EMBEDDING_LOCAL_FILES_ONLY,
    embedding_cache_path: Path | None = None,
    query_cache_path: Path | None = None,
    qdrant_path: Path | None = None,
    dense_backend: str = DEFAULT_DENSE_BACKEND,
    bm25_cache_path: Path | None = None,
) -> list[RetrievalResult]:
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    candidate_ids_digest = ordered_values_hash(candidate_ids)
    sections_by_candidate = [candidate_sections(candidate, clip=True) for candidate in candidates]
    section_texts = {
        section: [sections[section] for sections in sections_by_candidate]
        for section in RETRIEVAL_SECTION_WEIGHTS
    }
    dense_scores, section_scores = section_weighted_dense_scores(
        section_texts=section_texts,
        candidate_ids=candidate_ids,
        model_name=model_name,
        batch_size=batch_size,
        collection_name=collection_name,
        local_files_only=local_files_only,
        embedding_cache_path=embedding_cache_path,
        query_cache_path=query_cache_path,
        qdrant_path=qdrant_path,
        dense_backend=dense_backend,
    )
    full_texts = [
        full_candidate_text(candidate_sections(candidate, clip=False))
        for candidate in candidates
    ]
    bm25 = cached_bm25_scores(
        texts=full_texts,
        candidate_ids_digest=candidate_ids_digest,
        bm25_cache_path=bm25_cache_path,
    )
    hybrid = (HYBRID_DENSE_WEIGHT * dense_scores) + (HYBRID_BM25_WEIGHT * bm25)

    results = []
    for index, candidate in enumerate(candidates):
        per_section = {
            section: float(scores[index])
            for section, scores in section_scores.items()
        }
        results.append(
            RetrievalResult(
                candidate_id=str(candidate["candidate_id"]),
                hybrid_score=float(hybrid[index]),
                dense_score=float(dense_scores[index]),
                bm25_score=float(bm25[index]),
                section_scores=per_section,
                candidate=candidate,
            )
        )
    results.sort(key=lambda result: (-result.hybrid_score, result.candidate_id))
    return results[:top_n]


def write_retrieval_outputs(
    results: list[RetrievalResult],
    csv_path: Path,
    jsonl_path: Path,
) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        active_sections = jd_section_names()
        writer.writerow(
            [
                "candidate_id",
                "rank",
                "hybrid_score",
                "dense_score",
                "bm25_score",
            ]
            + [f"dense_{section}" for section in active_sections]
        )
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                [
                    result.candidate_id,
                    rank,
                    f"{result.hybrid_score:.6f}",
                    f"{result.dense_score:.6f}",
                    f"{result.bm25_score:.6f}",
                ]
                + [
                    f"{result.section_scores.get(section, 0.0):.6f}"
                    for section in active_sections
                ]
            )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for rank, result in enumerate(results, start=1):
            payload = {
                "rank": rank,
                "candidate_id": result.candidate_id,
                "hybrid_score": result.hybrid_score,
                "dense_score": result.dense_score,
                "bm25_score": result.bm25_score,
                "section_scores": result.section_scores,
                "candidate": result.candidate,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

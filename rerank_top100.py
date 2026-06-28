#!/usr/bin/env python3
"""Evidence-based final reranker from top-2000 retrieval output to submission.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


DEFAULT_TOP2000_PATH = Path("top_2000_candidates.jsonl")
DEFAULT_SUBMISSION_PATH = Path("submission.csv")
DEFAULT_DEBUG_PATH = Path("rerank_debug_top2000.csv")
DEFAULT_REJECTED_PATH = Path("rejected_candidates.jsonl")
DEFAULT_CANDIDATES_PATH = Path("candidates.jsonl")
DEFAULT_TOP100_JSONL_PATH = Path("top100.jsonl")
REFERENCE_DATE = date(2026, 6, 20)

TOKEN_CACHE: dict[str, re.Pattern] = {}


RANKING_RETRIEVAL_TERMS = [
    "ranking system",
    "search ranking",
    "search relevance",
    "retrieval system",
    "semantic search",
    "vector search",
    "hybrid retrieval",
    "hybrid search",
    "bm25",
    "elasticsearch",
    "opensearch",
    "solr",
    "faiss",
    "qdrant",
    "pinecone",
    "milvus",
    "weaviate",
    "llm reranking",
    "reranker",
    "cross encoder",
    "bi encoder",
    "learning to rank",
]

RECOMMENDATION_MATCHING_TERMS = [
    "recommendation system",
    "recommendation systems",
    "recommender system",
    "recommender systems",
    "candidate matching",
    "job matching",
    "matching system",
    "marketplace matching",
    "personalization",
]

EMBEDDING_VECTOR_TERMS = [
    "embeddings",
    "embedding pipeline",
    "semantic search",
    "vector search",
    "hybrid retrieval",
    "hybrid search",
    "faiss",
    "qdrant",
    "pinecone",
    "milvus",
    "weaviate",
    "sentence transformers",
    "sentence transformer",
]

PRODUCTION_TERMS = [
    "production ml",
    "deployed ml",
    "model serving",
    "ml pipeline",
    "feature pipeline",
    "python backend",
    "fastapi",
    "api latency",
    "latency",
    "monitoring",
    "scale",
    "backend api",
    "python service",
    "deployed",
    "shipped",
]

EVALUATION_TERMS = [
    "relevance labeling",
    "relevance labels",
    "human relevance judgments",
    "click-through data",
    "ndcg",
    "mrr",
    "map",
    "precision@k",
    "recall@k",
    "a/b testing",
    "a/b test",
    "offline evaluation",
    "online evaluation",
    "feedback loop",
]

PYTHON_DATA_ML_TERMS = [
    "python",
    "machine learning",
    "ml",
    "data pipeline",
    "data pipelines",
    "feature engineering",
    "feature pipeline",
    "spark",
    "kafka",
    "airflow",
    "backend",
    "api",
    "fastapi",
    "model serving",
]

DIRECT_SEARCH_EVALUATION_EVIDENCE_TERMS = [
    "learning-to-rank",
    "learning to rank",
    "ranking layer",
    "search relevance",
    "retrieval system",
    "semantic search",
    "vector search",
    "hybrid retrieval",
    "bm25",
    "elasticsearch",
    "opensearch",
    "faiss",
    "qdrant",
    "pinecone",
    "relevance labeling",
    "relevance labels",
    "human judgments",
    "human relevance judgments",
    "click-through data",
    "ndcg",
    "mrr",
    "map",
    "a/b testing",
    "offline evaluation",
    "online evaluation",
]

WEAK_RANKING_DEPTH_TERMS = [
    "lighter weight than ranking systems",
    "less mature than ranking systems",
    "still building depth",
    "want to grow into",
    "beyond the surface level",
    "not the core of my day",
    "production deployment was handled by the platform team",
    "didn't make it to production",
    "did not make it to production",
    "pure ml side of the work",
    "modeling and analysis side",
]

AI_SKILL_TERMS = [
    "rag",
    "langchain",
    "openai",
    "chatgpt",
    "llm",
    "llms",
    "embeddings",
    "vector search",
    "semantic search",
    "hybrid retrieval",
    "pinecone",
    "qdrant",
    "faiss",
    "sentence transformers",
    "hugging face transformers",
    "recommendation systems",
    "recommender systems",
    "information retrieval",
    "fine-tuning llms",
    "prompt engineering",
    "genai",
    "generative ai",
]

GENAI_EXPLORER_TERMS = [
    "online courses",
    "side projects",
    "exploring ai",
    "exploring genai",
    "generative ai explorer",
    "genai explorer",
    "chatgpt productivity",
    "ai-assisted content",
    "content creation",
    "drafting",
    "editing",
    "interested in transitioning",
    "learning modern ml",
    "curious about ai tools",
    "streamline workflows",
]

ADJACENT_TECH_TITLES = [
    "data engineer",
    "analytics engineer",
    "backend engineer",
    "software engineer",
    "platform engineer",
    "data scientist",
]

HANDS_ON_VERBS = [
    "built",
    "implemented",
    "developed",
    "designed",
    "deployed",
    "shipped",
    "optimized",
    "debugged",
    "integrated",
    "trained",
    "evaluated",
    "created",
    "coded",
    "migrated",
]

LEADERSHIP_VERBS = [
    "led",
    "managed",
    "owned",
    "oversaw",
    "coordinated",
    "guided",
    "mentored",
    "drove",
]

SEARCH_RETRIEVAL_REASON_TERMS = [
    "semantic search",
    "retrieval system",
    "vector search",
    "hybrid retrieval",
    "hybrid search",
    "bm25",
    "elasticsearch",
    "opensearch",
    "faiss",
    "qdrant",
    "pinecone",
    "query expansion",
    "nearest-neighbor retrieval",
    "nearest neighbor retrieval",
]

RANKING_REASON_TERMS = [
    "ranking layer",
    "learning-to-rank",
    "learning to rank",
    "search ranking",
    "ranking evaluation",
    "relevance labeling",
    "relevance labels",
    "click-through data",
    "human judgments",
    "human relevance judgments",
    "ndcg",
    "mrr",
    "map",
]

RECOMMENDATION_REASON_TERMS = [
    "recommendation system",
    "recommendation systems",
    "recommender system",
    "recommender systems",
    "personalization",
    "candidate matching",
    "job matching",
    "marketplace matching",
]

PRODUCTION_REASON_TERMS = [
    "production ml",
    "deployed ml",
    "model serving",
    "ml pipeline",
    "feature pipeline",
    "fastapi",
    "backend api",
    "python backend",
    "monitoring",
    "latency",
    "scale",
]

NLP_LLM_REASON_TERMS = [
    "nlp",
    "transformers",
    "sentence-transformers",
    "sentence transformers",
    "llm reranking",
    "rag system",
    "embeddings",
    "cross encoder",
    "cross-encoder",
    "bi encoder",
    "bi-encoder",
]

DATA_INFRA_REASON_TERMS = [
    "data engineer",
    "analytics engineer",
    "spark",
    "kafka",
    "airflow",
    "dbt",
    "snowflake",
    "data warehouse",
    "feature pipeline",
    "feature pipelines",
]

SEARCH_SYSTEM_TERMS = [
    "semantic search",
    "retrieval system",
    "vector search",
    "hybrid retrieval",
    "hybrid search",
    "query expansion",
    "nearest-neighbor retrieval",
    "nearest neighbor retrieval",
]

RANKING_SYSTEM_TERMS = [
    "ranking layer",
    "learning-to-rank",
    "learning to rank",
    "search ranking",
    "ranking evaluation",
    "relevance labeling",
    "relevance labels",
]

RECOMMENDATION_SYSTEM_TERMS = [
    "recommendation system",
    "recommendation systems",
    "recommender system",
    "recommender systems",
    "candidate matching",
    "job matching",
    "marketplace matching",
    "personalization",
]

PRODUCTION_SYSTEM_TERMS = [
    "production ml",
    "deployed ml",
    "model serving",
    "ml pipeline",
    "feature pipeline",
    "backend api",
    "python backend",
]

NLP_SYSTEM_TERMS = [
    "nlp",
    "embeddings",
    "sentence-transformers",
    "sentence transformers",
    "transformers",
    "llm reranking",
    "rag system",
    "cross encoder",
    "cross-encoder",
    "bi encoder",
    "bi-encoder",
]

TOOL_REASON_TERMS = [
    "bm25",
    "faiss",
    "qdrant",
    "pinecone",
    "elasticsearch",
    "opensearch",
    "sentence-transformers",
    "sentence transformers",
    "fastapi",
    "spark",
    "kafka",
    "airflow",
    "mlflow",
    "kubeflow",
]

EVALUATION_REASON_TERMS = [
    "ndcg",
    "mrr",
    "map",
    "a/b testing",
    "a/b test",
    "human judgments",
    "human relevance judgments",
    "relevance labeling",
    "relevance labels",
    "click-through data",
    "offline evaluation",
    "online evaluation",
]

DISPLAY_TERMS = {
    "bm25": "BM25",
    "faiss": "FAISS",
    "qdrant": "Qdrant",
    "pinecone": "Pinecone",
    "milvus": "Milvus",
    "weaviate": "Weaviate",
    "elasticsearch": "Elasticsearch",
    "opensearch": "OpenSearch",
    "fastapi": "FastAPI",
    "mlflow": "MLflow",
    "kubeflow": "Kubeflow",
    "ndcg": "NDCG",
    "mrr": "MRR",
    "map": "MAP",
    "a/b testing": "A/B testing",
    "a/b test": "A/B testing",
    "precision@k": "precision@k",
    "recall@k": "recall@k",
    "sentence-transformers": "sentence-transformers",
    "sentence transformers": "sentence-transformers",
    "llm reranking": "LLM reranking",
    "rag system": "RAG system",
    "nlp": "NLP",
    "ml pipeline": "ML pipeline",
    "production ml": "production ML",
    "deployed ml": "deployed ML",
    "backend api": "backend API",
    "python backend": "Python backend",
    "learning-to-rank": "learning-to-rank",
}

ALL_CAREER_REASON_TERMS = (
    SEARCH_RETRIEVAL_REASON_TERMS
    + RANKING_REASON_TERMS
    + RECOMMENDATION_REASON_TERMS
    + PRODUCTION_REASON_TERMS
    + NLP_LLM_REASON_TERMS
    + DATA_INFRA_REASON_TERMS
)


@dataclass
class RerankResult:
    candidate_id: str
    original_rank: int
    original_hybrid_score: float
    final_score: float
    components: dict[str, float]
    penalties: list[str]
    reason_codes: list[str]
    hireability_penalty: float
    hireability_reason_codes: list[str]
    evidence_realism_penalty: float
    evidence_realism_reason_codes: list[str]
    seniority_drift_penalty: float
    senior_ic_alignment_bonus: float
    seniority_alignment_reason_codes: list[str]
    primary_differentiator: str
    reasoning: str
    debug_flags: dict[str, object]
    row: dict


def normalize(value: object) -> str:
    return str(value or "").lower()


def term_in_text(text: str, term: str) -> bool:
    term = term.lower()
    if term.replace(".", "").replace("@", "").isalnum():
        if term not in text:
            return False
        pattern = TOKEN_CACHE.get(term)
        if pattern is None:
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
            TOKEN_CACHE[term] = pattern
        return pattern.search(text) is not None
    return term in text


def match_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term_in_text(text, term)]


def has_any(text: str, terms: list[str]) -> bool:
    return any(term_in_text(text, term) for term in terms)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def profile(candidate: dict) -> dict:
    return candidate.get("profile", {}) or {}


def signals(candidate: dict) -> dict:
    return candidate.get("redrob_signals", {}) or {}


def career_text(candidate: dict) -> str:
    return " ".join(
        " ".join(str(job.get(key) or "") for key in ["title", "company", "industry", "description"])
        for job in candidate.get("career_history", [])
    ).lower()


def summary_text(candidate: dict) -> str:
    p = profile(candidate)
    return " ".join(str(p.get(key) or "") for key in ["headline", "summary", "current_title", "current_industry"]).lower()


def skills_text(candidate: dict) -> str:
    return " ".join(str(skill.get("name") or "") for skill in candidate.get("skills", [])).lower()


def role_text(candidate: dict) -> str:
    p = profile(candidate)
    titles = [str(p.get("current_title") or "")]
    titles.extend(str(job.get("title") or "") for job in candidate.get("career_history", []))
    return " ".join(titles).lower()


def category_matches(text: str) -> dict[str, list[str]]:
    return {
        "ranking_retrieval": match_terms(text, RANKING_RETRIEVAL_TERMS),
        "recommendation_matching": match_terms(text, RECOMMENDATION_MATCHING_TERMS),
        "embedding_vector": match_terms(text, EMBEDDING_VECTOR_TERMS),
        "production": match_terms(text, PRODUCTION_TERMS),
        "evaluation": match_terms(text, EVALUATION_TERMS),
        "python_data_ml": match_terms(text, PYTHON_DATA_ML_TERMS),
    }


def unique_ordered(values: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if limit is not None and len(result) >= limit:
            break
    return result


def display_term(term: str) -> str:
    normalized = term.lower()
    if normalized in DISPLAY_TERMS:
        return DISPLAY_TERMS[normalized]
    return normalized


def format_terms(terms: list[str], limit: int = 3) -> str:
    clean = []
    seen = set()
    for term in terms:
        display = display_term(term)
        canonical = display.lower()
        canonical = canonical.replace("recommendation systems", "recommendation system")
        canonical = canonical.replace("recommender systems", "recommender system")
        canonical = canonical.replace("human relevance judgments", "human judgments")
        if canonical in seen:
            continue
        seen.add(canonical)
        clean.append(display)
        if len(clean) >= limit:
            break
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def join_phrases(phrases: list[str], limit: int = 3) -> str:
    clean = unique_ordered([phrase for phrase in phrases if phrase], limit)
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def word_count(text: str) -> int:
    return len(text.split())


def sentence_case_preserve(text: str) -> str:
    if not text:
        return text
    return text[:1].upper() + text[1:]


def career_sentences(candidate: dict) -> list[str]:
    sentences: list[str] = []
    for job in candidate.get("career_history", []):
        title = str(job.get("title") or "").strip()
        description = str(job.get("description") or "").strip()
        text = " ".join(part for part in [title, description] if part)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip().lower()
            if sentence:
                sentences.append(sentence)
    return sentences


@dataclass
class EvidenceRealismIndex:
    description_counts: Counter[str]
    fingerprint_candidate_counts: Counter[str]
    sentence_counts: Counter[str]
    sentence_companies: dict[str, set[str]]


def normalize_evidence_text(value: object) -> str:
    text = normalize(value)
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;,")


def career_description_fingerprint(description: object, company: object = "") -> str:
    text = normalize(description)
    company_tokens = {
        token for token in re.findall(r"[a-z0-9]+", normalize(company))
        if len(token) > 2
    }
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " ", text)
    text = re.sub(r"[^a-z0-9+#@]+", " ", text)
    tokens = [
        token for token in text.split()
        if token not in company_tokens and len(token) > 1
    ]
    return " ".join(tokens[:32])


def career_description_records(candidate: dict) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for job in candidate.get("career_history", []):
        description = str(job.get("description") or "")
        normalized = normalize_evidence_text(description)
        if not normalized:
            continue
        records.append({
            "normalized": normalized,
            "description": description,
            "company": str(job.get("company") or ""),
            "industry": str(job.get("industry") or ""),
            "duration_months": int(float(job.get("duration_months") or 0)),
            "fingerprint": career_description_fingerprint(description, job.get("company")),
        })
    return records


def build_evidence_realism_index(rows: list[dict]) -> EvidenceRealismIndex:
    description_counts: Counter[str] = Counter()
    fingerprint_candidate_counts: Counter[str] = Counter()
    sentence_counts: Counter[str] = Counter()
    sentence_companies: dict[str, set[str]] = defaultdict(set)
    fingerprint_candidates: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        candidate = row["candidate"]
        candidate_id = str(candidate.get("candidate_id") or row.get("candidate_id") or "")
        for record in career_description_records(candidate):
            description = str(record["normalized"])
            if len(description) >= 120 and has_any(description, ALL_CAREER_REASON_TERMS):
                description_counts[description] += 1
            fingerprint = str(record.get("fingerprint") or "")
            if len(fingerprint) >= 80 and has_any(fingerprint, ALL_CAREER_REASON_TERMS):
                fingerprint_candidates[fingerprint].add(candidate_id)
        for job in candidate.get("career_history", []):
            company = str(job.get("company") or "")
            text = " ".join(
                part for part in [
                    str(job.get("title") or ""),
                    str(job.get("description") or ""),
                ]
                if part
            )
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                normalized = normalize_evidence_text(sentence)
                if len(normalized) < 70 or not has_any(normalized, ALL_CAREER_REASON_TERMS):
                    continue
                sentence_counts[normalized] += 1
                if company:
                    sentence_companies[normalized].add(company.lower())

    for fingerprint, candidate_ids in fingerprint_candidates.items():
        fingerprint_candidate_counts[fingerprint] = len(candidate_ids)

    return EvidenceRealismIndex(
        description_counts=description_counts,
        fingerprint_candidate_counts=fingerprint_candidate_counts,
        sentence_counts=sentence_counts,
        sentence_companies=sentence_companies,
    )


def claimed_project_months(description: str) -> list[int]:
    months = []
    for match in re.finditer(r"\b(?:over|for|during|across|in)\s+(\d{1,2})\s+months?\b", normalize(description)):
        months.append(int(match.group(1)))
    return months


def has_duration_contradiction(candidate: dict) -> bool:
    for record in career_description_records(candidate):
        duration = int(record["duration_months"])
        if duration <= 0:
            continue
        for claimed_months in claimed_project_months(str(record["description"])):
            if claimed_months > duration + 1:
                return True
    return False


def has_domain_mismatch(candidate: dict) -> bool:
    compatible = [
        "e-commerce",
        "ecommerce",
        "commerce",
        "retail",
        "marketplace",
        "shopping",
        "internet",
        "consumer internet",
    ]
    incompatible = [
        "gaming",
        "sports",
        "paper products",
        "banking",
        "insurance",
        "manufacturing",
        "it services",
        "consulting",
    ]
    for record in career_description_records(candidate):
        description = str(record["normalized"])
        industry = normalize(record["industry"])
        company = normalize(record["company"])
        if not has_any(description, ["e-commerce search product", "ecommerce search product", "e-commerce search", "ecommerce search"]):
            continue
        if any(term in industry or term in company for term in compatible):
            continue
        if any(term in industry or term in company for term in incompatible):
            return True
    return False


def salary_range_inverted(candidate: dict) -> bool:
    salary = signals(candidate).get("expected_salary_range_inr_lpa") or {}
    if not isinstance(salary, dict):
        return False
    try:
        minimum = float(salary.get("min"))
        maximum = float(salary.get("max"))
    except (TypeError, ValueError):
        return False
    return minimum > maximum


def weak_hireability_signal_count(candidate: dict) -> int:
    s = signals(candidate)
    notice = float(s.get("notice_period_days") if s.get("notice_period_days") is not None else 0.0)
    response_rate = float(s.get("recruiter_response_rate") if s.get("recruiter_response_rate") is not None else 1.0)
    response_hours = float(s.get("avg_response_time_hours") if s.get("avg_response_time_hours") is not None else 0.0)
    verified_email = bool(s.get("verified_email"))
    github_activity = float(s.get("github_activity_score") if s.get("github_activity_score") is not None else 0.0)
    open_to_work = bool(s.get("open_to_work_flag"))
    weak_signals = [
        notice >= 90,
        response_rate < 0.60,
        response_hours > 48,
        not verified_email,
        github_activity == -1,
        not open_to_work,
    ]
    return sum(1 for item in weak_signals if item)


def weak_hireability_cluster_penalty(count: int) -> float:
    if count >= 4:
        return 0.055
    if count == 3:
        return 0.035
    if count == 2:
        return 0.015
    return 0.0


def repeated_template_stats(
    candidate: dict,
    index: EvidenceRealismIndex | None,
    records: list[dict[str, object]] | None = None,
) -> tuple[float, int, bool, list[str]]:
    if index is None:
        return 0.0, 0, False, []

    penalty = 0.0
    max_repeated_count = 0
    reason_codes: list[str] = []
    if records is None:
        records = career_description_records(candidate)
    fingerprints = [
        str(record.get("fingerprint") or "")
        for record in records
    ]
    fingerprint_counts = Counter(fp for fp in fingerprints if fp)

    same_candidate_repeated = any(
        own_count > 1 and len(fingerprint) >= 80 and has_any(fingerprint, ALL_CAREER_REASON_TERMS)
        for fingerprint, own_count in fingerprint_counts.items()
    )
    if same_candidate_repeated:
        penalty += 0.02
        reason_codes.append("same_candidate_repeated_role_template")

    repeated_penalty = 0.0
    for fingerprint in fingerprint_counts:
        if len(fingerprint) < 80 or not has_any(fingerprint, ALL_CAREER_REASON_TERMS):
            continue
        corpus_count = index.fingerprint_candidate_counts.get(fingerprint, 0)
        max_repeated_count = max(max_repeated_count, corpus_count)
        if 2 <= corpus_count <= 3:
            repeated_penalty += 0.01
        elif 4 <= corpus_count <= 8:
            repeated_penalty += 0.025
        elif corpus_count > 8:
            repeated_penalty += 0.04
    repeated_penalty = min(0.05, repeated_penalty)
    if repeated_penalty:
        penalty += repeated_penalty
        reason_codes.append("repeated_career_template")

    return penalty, max_repeated_count, same_candidate_repeated, reason_codes


def evidence_realism_penalty(
    candidate: dict,
    global_template_stats: EvidenceRealismIndex | None = None,
    strong_technical: bool = False,
    weak_hireability_count: int | None = None,
    records: list[dict[str, object]] | None = None,
    repeated_template_details: tuple[float, int, bool, list[str]] | None = None,
) -> tuple[float, list[str]]:
    penalty = 0.0
    reason_codes: list[str] = []
    index = global_template_stats
    weak_count = weak_hireability_count
    if weak_count is None:
        weak_count = weak_hireability_signal_count(candidate)

    if salary_range_inverted(candidate):
        penalty += 0.04 if weak_count >= 3 else 0.025
        reason_codes.append("salary_range_inverted")

    cluster_penalty = weak_hireability_cluster_penalty(weak_count)
    if cluster_penalty:
        penalty += cluster_penalty
        reason_codes.append("weak_hireability_cluster")

    if records is None:
        records = career_description_records(candidate)
    if repeated_template_details is None:
        repeated_template_details = repeated_template_stats(candidate, index, records=records)
    repeated_penalty, _, _, repeated_reasons = repeated_template_details
    penalty += repeated_penalty
    reason_codes.extend(repeated_reasons)

    description_counts = Counter(str(record["normalized"]) for record in records)

    if index is not None:
        for description, own_count in description_counts.items():
            corpus_count = index.description_counts.get(description, 0)
            if own_count > 1:
                penalty += 0.02
                reason_codes.append("same_candidate_repeated_role_template")
            if corpus_count >= 3 and has_any(description, ALL_CAREER_REASON_TERMS):
                penalty += 0.01 + min(0.02, 0.003 * (corpus_count - 3))
                reason_codes.append("repeated_career_template")

    repeated_sentence_hits = 0
    if index is not None:
        for sentence in career_sentences(candidate):
            normalized = normalize_evidence_text(sentence)
            if len(normalized) < 70 or not has_any(normalized, ALL_CAREER_REASON_TERMS):
                continue
            corpus_count = index.sentence_counts.get(normalized, 0)
            company_count = len(index.sentence_companies.get(normalized, set()))
            if corpus_count >= 4 and company_count >= 3:
                repeated_sentence_hits += 1

    if repeated_sentence_hits:
        penalty += min(0.04, 0.012 * repeated_sentence_hits)
        reason_codes.append("repeated_career_template")

    if has_duration_contradiction(candidate):
        penalty += 0.04
        reason_codes.append("project_duration_exceeds_role_duration")

    if has_domain_mismatch(candidate):
        penalty += 0.018
        reason_codes.append("company_domain_description_mismatch")

    unique_support = 0
    if extract_impact_signals(candidate):
        unique_support += 1
    if concise_eval(candidate):
        unique_support += 1
    if has_any(career_text(candidate), ["latency", "monitoring", "model serving", "feature pipeline", "pipeline", "api"]):
        unique_support += 1
    if reason_codes and unique_support >= 2:
        penalty *= 0.75

    hard_realism_issue = any(
        code in reason_codes
        for code in [
            "project_duration_exceeds_role_duration",
            "salary_range_inverted",
            "weak_hireability_cluster",
        ]
    )
    profile_consistency_only = all(
        code in {
            "repeated_career_template",
            "same_candidate_repeated_role_template",
            "company_domain_description_mismatch",
        }
        for code in reason_codes
    )
    has_same_candidate_template = "same_candidate_repeated_role_template" in reason_codes
    if strong_technical and unique_support >= 2 and profile_consistency_only and has_same_candidate_template:
        cap = 0.04
    elif strong_technical and unique_support >= 2 and profile_consistency_only:
        cap = 0.02
    elif strong_technical and unique_support >= 2 and not hard_realism_issue:
        cap = 0.04
    else:
        cap = 0.10 if strong_technical else 0.12
    return min(penalty, cap), unique_ordered(reason_codes, 10)


def same_sentence_evidence(candidate: dict, terms: list[str], verbs: list[str]) -> tuple[str, list[str]] | None:
    for sentence in career_sentences(candidate):
        term_hits = match_terms(sentence, terms)
        verb_hits = match_terms(sentence, verbs)
        if term_hits and verb_hits:
            return verb_hits[0], unique_ordered(term_hits, 3)
    return None


def evidence_phrase(candidate: dict, terms: list[str], fallback: str = "career history shows") -> str:
    career = career_text(candidate)
    term_hits = unique_ordered(match_terms(career, terms), 3)
    if not term_hits:
        return ""

    hands_on = same_sentence_evidence(candidate, terms, HANDS_ON_VERBS)
    if hands_on:
        verb, hits = hands_on
        metric_hits = {"ndcg", "mrr", "map", "precision@k", "recall@k", "a/b testing", "a/b test"}
        if all(hit in metric_hits for hit in hits):
            return f"career history includes {format_terms(hits)}"
        operational_hits = {"latency", "monitoring", "scale"}
        if all(hit in operational_hits for hit in hits):
            return f"career history mentions {format_terms(hits)}"
        return f"{verb} {format_terms(hits)}"

    leadership = same_sentence_evidence(candidate, terms, LEADERSHIP_VERBS)
    if leadership:
        _, hits = leadership
        return f"shows ownership of {format_terms(hits)}"

    return f"{fallback} {format_terms(term_hits)}"


def extract_impact_signals(candidate: dict) -> list[str]:
    career = career_text(candidate)
    raw = re.findall(
        r"(?:~?\d+(?:\.\d+)?\s?(?:%|k|m|million|lakh|crore)\b(?:[- ][a-z]+){0,3})",
        career,
    )
    cleaned = []
    for item in raw:
        text = item.strip(" .,;:")
        text = text.replace("~", "~")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*k\b", r"\1K", text)
        text = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*m\b", r"\1M", text)
        if text and text not in cleaned:
            cleaned.append(text.replace(" documents", " documents").replace(" over", " over"))
    return cleaned[:2]


def strongest_availability_signal(candidate: dict) -> str:
    s = signals(candidate)
    pieces = []
    response_rate = s.get("recruiter_response_rate")
    interview = s.get("interview_completion_rate")
    notice = s.get("notice_period_days")

    if s.get("open_to_work_flag") and notice is not None and float(notice) <= 30:
        pieces.append(f"open to work with {int(float(notice))}-day notice")
    if response_rate is not None and float(response_rate) >= 0.6:
        pieces.append(f"{float(response_rate):.2f} recruiter response rate")
    if interview is not None and float(interview) >= 0.8:
        pieces.append(f"{float(interview):.2f} interview completion")

    if not pieces:
        return ""
    return "Availability adds confidence: " + ", ".join(pieces[:2]) + "."


def penalty_reasoning(reason_codes: list[str]) -> str:
    if "adjacent_technical_no_retrieval_penalty" in reason_codes:
        return "Weaker on direct ranking/retrieval ownership, so this is treated as adjacent."
    if "unsupported_ai_skills_penalty" in reason_codes:
        return "Skills outweigh career evidence, so the rank is discounted."
    if "genai_explorer_penalty" in reason_codes:
        return "GenAI interest outweighs proved AI engineering depth."
    if "leadership_only_penalty" in reason_codes:
        return "Leadership evidence outweighs hands-on implementation evidence."
    if "very_long_notice_penalty" in reason_codes or "long_notice_penalty" in reason_codes:
        return "Notice period reduces immediate hireability."
    return ""


def candidate_profile_type(candidate: dict, components: dict[str, float]) -> str:
    career = career_text(candidate)
    roles = role_text(candidate)
    years = float(profile(candidate).get("years_of_experience") or 0.0)
    has_core = has_any(career, SEARCH_RETRIEVAL_REASON_TERMS + RANKING_REASON_TERMS + RECOMMENDATION_REASON_TERMS)
    has_adjacent = has_any(roles + " " + career, DATA_INFRA_REASON_TERMS)
    has_hands_on = has_any(career, HANDS_ON_VERBS)

    if has_adjacent and not has_core:
        return "adjacent_data_infra_candidate"

    profile_scores = {
        "search_retrieval_specialist": 2.0 * len(match_terms(career, SEARCH_RETRIEVAL_REASON_TERMS))
        + len(match_terms(career, TOOL_REASON_TERMS)),
        "ranking_ltr_engineer": 2.0 * len(match_terms(career, RANKING_REASON_TERMS))
        + 1.5 * len(match_terms(career, EVALUATION_REASON_TERMS)),
        "recommendation_matching_engineer": 2.0 * len(match_terms(career, RECOMMENDATION_REASON_TERMS)),
        "production_ml_engineer": 1.5 * len(match_terms(career, PRODUCTION_REASON_TERMS))
        + (1.0 if has_hands_on else 0.0),
        "nlp_llm_engineer": 1.5 * len(match_terms(career, NLP_LLM_REASON_TERMS)),
        "senior_hands_on_ic": 1.0
        if 5 <= years <= 9 and has_hands_on and components.get("hands_on_depth_score", 0.0) >= 0.75
        else 0.0,
    }
    best_type, best_score = max(profile_scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return "general_ml_fit"
    return best_type


def action_evidence(candidate: dict, terms: list[str]) -> tuple[str, list[str], bool]:
    hands_on = same_sentence_evidence(candidate, terms, HANDS_ON_VERBS)
    if hands_on:
        verb, hits = hands_on
        return verb, hits, True
    leadership = same_sentence_evidence(candidate, terms, LEADERSHIP_VERBS)
    if leadership:
        verb, hits = leadership
        return verb, hits, False
    hits = match_terms(career_text(candidate), terms)
    return "shows", hits, False


def action_for_profile(candidate: dict, primary_terms: list[str], fallback_terms: list[str] | None = None) -> tuple[str, list[str], bool]:
    verb, hits, hands_on = action_evidence(candidate, primary_terms)
    if verb != "shows":
        return verb, hits, hands_on
    if fallback_terms:
        fallback_verb, fallback_hits, fallback_hands_on = action_evidence(candidate, fallback_terms)
        if fallback_verb != "shows":
            return fallback_verb, fallback_hits or hits, fallback_hands_on

    career = career_text(candidate)
    broad_hands_on = match_terms(career, HANDS_ON_VERBS)
    if broad_hands_on:
        return broad_hands_on[0], hits, True
    broad_leadership = match_terms(career, LEADERSHIP_VERBS)
    if broad_leadership:
        return broad_leadership[0], hits, False
    return verb, hits, hands_on


def action_word(verb: str, hands_on: bool) -> str:
    if verb == "shows":
        return "Shows"
    if not hands_on and verb in LEADERSHIP_VERBS:
        return {
            "led": "Led",
            "managed": "Managed",
            "owned": "Owned",
            "oversaw": "Oversaw",
            "coordinated": "Coordinated",
            "guided": "Guided",
            "mentored": "Mentored",
            "drove": "Drove",
        }.get(verb, sentence_case_preserve(verb))
    return {
        "built": "Built",
        "implemented": "Implemented",
        "developed": "Developed",
        "designed": "Designed",
        "deployed": "Deployed",
        "shipped": "Shipped",
        "optimized": "Optimized",
        "debugged": "Debugged",
        "integrated": "Integrated",
        "trained": "Trained",
        "evaluated": "Evaluated",
        "created": "Created",
        "coded": "Coded",
        "migrated": "Migrated",
    }.get(verb, sentence_case_preserve(verb))


def system_label(hits: list[str], fallback: str) -> str:
    useful = [hit for hit in hits if hit not in TOOL_REASON_TERMS and hit not in EVALUATION_REASON_TERMS]
    if useful and all(hit == "nlp" for hit in useful):
        useful = []
    formatted = format_terms(useful, 2)
    return formatted or fallback


def candidate_variant(candidate: dict, options: list[str]) -> str:
    candidate_id = str(candidate.get("candidate_id") or "")
    index = sum(ord(char) for char in candidate_id) % len(options)
    return options[index]


def system_label_variant(candidate: dict, hits: list[str], fallbacks: list[str]) -> str:
    useful = [hit for hit in hits if hit not in TOOL_REASON_TERMS and hit not in EVALUATION_REASON_TERMS]
    if useful and all(hit == "nlp" for hit in useful):
        useful = []
    formatted = format_terms(useful, 2)
    return formatted or candidate_variant(candidate, fallbacks)


def tool_clause(career: str) -> str:
    tools = format_terms(match_terms(career, TOOL_REASON_TERMS), 3)
    if not tools:
        return ""
    return f" using {tools}"


def evaluation_clause(career: str) -> str:
    evals = format_terms(match_terms(career, EVALUATION_REASON_TERMS), 2)
    if not evals:
        return ""
    return f" with {evals}"


def impact_clause(candidate: dict) -> str:
    impacts = extract_impact_signals(candidate)
    if not impacts:
        return ""
    return f" and {impacts[0]} scale/impact evidence"


def support_clause(candidate: dict, include_tools: bool = True, include_eval: bool = True, include_impact: bool = True) -> str:
    career = career_text(candidate)
    tools = format_terms(match_terms(career, TOOL_REASON_TERMS), 3) if include_tools else ""
    evals = format_terms(match_terms(career, EVALUATION_REASON_TERMS), 2) if include_eval else ""
    impacts = extract_impact_signals(candidate) if include_impact else []

    clause = ""
    if tools:
        clause += f" using {tools}"
    if evals:
        clause += f", evaluated with {evals}" if clause else f", evaluated with {evals}"
    if impacts:
        clause += f", with {impacts[0]} scale/impact evidence" if clause else f", with {impacts[0]} scale/impact evidence"
    return clause


def jd_match_clause(profile_type: str, candidate: dict | None = None) -> str:
    chooser = candidate or {"candidate_id": profile_type}
    if profile_type == "ranking_ltr_engineer":
        return candidate_variant(chooser, [
            "direct evidence for Redrob's ranking and relevance-evaluation work",
            "the kind of ranking/evaluation ownership Redrob needs",
            "useful proof for improving Redrob's BM25/rule-based ranking stack",
        ])
    if profile_type == "recommendation_matching_engineer":
        return candidate_variant(chooser, [
            "a close match for Redrob's candidate-role matching needs",
            "relevant to Redrob's recruiter-candidate matching problem",
            "good evidence for matching and recommendation work at Redrob",
        ])
    if profile_type == "production_ml_engineer":
        return candidate_variant(chooser, [
            "useful for Redrob's applied ML production stack",
            "valuable for shipping reliable ML services at Redrob",
            "practical support for Redrob's production intelligence layer",
        ])
    if profile_type == "nlp_llm_engineer":
        return candidate_variant(chooser, [
            "relevant to Redrob's embedding and LLM-assisted retrieval roadmap",
            "useful for Redrob's NLP/retrieval layer",
            "applicable to Redrob's semantic matching roadmap",
        ])
    if profile_type == "adjacent_data_infra_candidate":
        return candidate_variant(chooser, [
            "useful infrastructure depth, though not direct search-ranking ownership",
            "adjacent ML infrastructure value, with weaker retrieval ownership",
            "helpful data-platform depth but less direct search-ranking evidence",
        ])
    return candidate_variant(chooser, [
        "directly matching Redrob's retrieval and ranking-improvement roadmap",
        "well aligned with Redrob's search and matching upgrade path",
        "strongly tied to Redrob's retrieval-first intelligence layer",
    ])


def shorten_reasoning(reasoning: str) -> str:
    if word_count(reasoning) <= 38:
        return reasoning
    sentences = re.split(r"(?<=[.!?])\s+", reasoning)
    trimmed = " ".join(sentences[:2]).strip()
    if 20 <= word_count(trimmed) <= 38:
        return trimmed
    words = reasoning.split()
    return " ".join(words[:38]).rstrip(" ,;:") + "."


def has_career(candidate: dict, terms: list[str]) -> bool:
    return has_any(career_text(candidate), terms)


def has_hands_on_career(candidate: dict) -> bool:
    return has_any(career_text(candidate), HANDS_ON_VERBS)


def determine_primary_differentiator(candidate: dict, component_scores: dict[str, float] | None = None) -> str:
    career = career_text(candidate)
    roles = role_text(candidate)
    years = float(profile(candidate).get("years_of_experience") or 0.0)
    has_ltr = has_any(career, ["learning-to-rank", "learning to rank", "ranking layer", "hand-tuned scoring"])
    has_bm25_migration = (
        has_any(career, ["semantic search", "embedding-based retrieval", "vector search"])
        and has_any(career, ["bm25", "elasticsearch", "opensearch", "keyword-search", "keyword search"])
    )
    has_retrieval = has_any(career, SEARCH_SYSTEM_TERMS + SEARCH_RETRIEVAL_REASON_TERMS)
    has_eval = has_any(career, EVALUATION_REASON_TERMS)
    has_recommendation = has_any(career, RECOMMENDATION_SYSTEM_TERMS + RECOMMENDATION_REASON_TERMS)
    has_candidate_matching = has_any(career, ["candidate matching", "job matching", "marketplace matching"])
    has_dense_sparse = (
        has_any(career, ["hybrid retrieval", "hybrid search", "dense retrieval", "sparse retrieval"])
        or (has_any(career, ["bm25", "elasticsearch", "opensearch"]) and has_any(career, ["semantic search", "vector search", "faiss", "qdrant", "pinecone", "embeddings"]))
    )
    has_production = has_any(career, PRODUCTION_SYSTEM_TERMS + PRODUCTION_REASON_TERMS)
    has_infra = has_any(roles + " " + career, DATA_INFRA_REASON_TERMS)
    has_scale = bool(extract_impact_signals(candidate)) or has_any(career, ["large scale", "scale", "latency"])
    hands_on = has_hands_on_career(candidate)

    if has_ltr:
        return "learning_to_rank_ownership"
    if has_bm25_migration:
        return "bm25_to_semantic_migration"
    if has_retrieval and hands_on:
        if has_scale:
            return "large_scale_search_system"
        return "hands_on_retrieval_builder"
    if has_eval and has_any(career, ["relevance labeling", "relevance labels", "human judgments", "human relevance judgments", "click-through data"]):
        return "relevance_labeling_eval_depth"
    if has_scale and has_retrieval:
        return "large_scale_search_system"
    if has_recommendation:
        return "recommendation_matching_ownership"
    if has_candidate_matching:
        return "candidate_job_matching_relevance"
    if has_dense_sparse:
        return "dense_sparse_hybrid_experience"
    if has_production:
        return "production_ml_infrastructure"
    if has_eval:
        return "strong_evaluation_mindset"
    if 5 <= years <= 9 and hands_on and (component_scores or {}).get("hands_on_depth_score", 0.0) >= 0.75:
        return "senior_hands_on_ic"
    if has_infra:
        return "adjacent_but_strong_ml_infra"
    if has_any(career, NLP_SYSTEM_TERMS + NLP_LLM_REASON_TERMS):
        return "nlp_embedding_experience"
    return "general_ml_pipeline_experience"


def differentiator_opening(candidate: dict, differentiator: str) -> str:
    openings = {
        "learning_to_rank_ownership": [
            "Strongest fit is ranking ownership",
            "Main reason to rank them is LTR ownership",
            "The clearest signal is ranking-system ownership",
            "Best evidence is learning-to-rank work",
            "Primary advantage is search-ranking ownership",
            "Most relevant proof is ranking-layer evolution",
            "Top signal is hands-on ranking work",
            "The ranking case is the clearest here",
            "Ranking-layer evolution is the main proof",
            "Learning-to-rank ownership drives this rank",
            "The strongest career proof is LTR migration",
            "Search-ranking ownership is the useful edge",
            "Direct ranking work is the main differentiator",
            "The LTR evidence is more persuasive than keywords",
            "Ranking-quality ownership anchors this profile",
            "The hand-tuned-to-LTR path is the key signal",
            "This profile is best differentiated by ranking ownership",
            "The direct ranking-system proof matters most",
            "LTR delivery is the main reason to shortlist",
            "Ranking workflow ownership separates this profile",
        ],
        "bm25_to_semantic_migration": [
            "Best signal is BM25-to-semantic-search migration",
            "Most useful proof is replacing keyword search with semantic retrieval",
            "The strongest plus point is semantic search over a BM25 baseline",
            "Primary differentiator is dense retrieval on top of keyword search",
            "The standout evidence is keyword-to-embedding retrieval migration",
            "Best Redrob match is the BM25-to-vector-search transition",
            "The clearest retrieval signal is semantic migration from BM25",
            "Strongest evidence is moving search beyond keyword scoring",
            "Keyword-to-vector retrieval work is the main proof",
            "The BM25 baseline migration is the best evidence",
            "Semantic retrieval over keyword search is the key signal",
            "The clearest plus point is BM25 replacement work",
            "Search modernization is the strongest differentiator",
            "The retrieval upgrade story is the main reason",
            "Dense retrieval layered over BM25 is the useful proof",
            "The semantic-search migration is the strongest signal",
        ],
        "hands_on_retrieval_builder": [
            "Best signal is hands-on retrieval building",
            "Primary edge is direct retrieval implementation",
            "The strongest evidence is search-system implementation",
            "Most relevant proof is building retrieval workflows",
           "The retrieval case is backed by actual implementation",
            "Best fit comes from actual search implementation",
            "The clearest plus point is retrieval engineering",
            "Strongest point is hands-on search/retrieval work",
            "Direct retrieval building is the main proof",
            "The practical search implementation matters most",
            "Hands-on search-system work drives the rank",
            "The retrieval builder signal is the differentiator",
            "Implementation depth in retrieval is the useful edge",
            "Search implementation is the strongest career proof",
            "The concrete proof is retrieval delivery",
            "Hands-on retrieval work separates this profile",
        ],
        "relevance_labeling_eval_depth": [
            "Evaluation depth is the main advantage",
            "The strongest plus point is relevance measurement",
            "Best evidence is relevance-labeling depth",
            "Primary differentiator is ranking-quality evaluation",
            "Most valuable signal is human/click judgment workflow",
            "The evaluation case is more credible than keyword overlap",
            "Best proof is search-quality measurement work",
            "The clearest signal is relevance evaluation depth",
        ],
        "large_scale_search_system": [
            "Strongest advantage is production retrieval scale",
            "Best signal is search work at meaningful scale",
            "Primary differentiator is large-scale retrieval",
            "The scale of the search system is the useful proof",
            "Most relevant edge is retrieval at production scale",
            "Best evidence is scaled semantic-search work",
            "The strongest case is production-scale search",
            "Large-scale retrieval is the clearest plus point",
        ],
        "recommendation_matching_ownership": [
            "Best match is recommendation-system ownership",
            "Primary fit is recommender or matching ownership",
            "The strongest plus point is recommendation work",
            "Most relevant evidence is personalization/matching experience",
            "Best signal is production recommendation ownership",
            "The matching case is the clearest here",
            "Strongest recruiter-relevant proof is recommendation work",
            "Primary differentiator is matching-system experience",
        ],
        "candidate_job_matching_relevance": [
            "Best signal is candidate/job matching relevance",
            "Primary advantage is marketplace matching experience",
            "The clearest Redrob fit is matching-system work",
            "Most relevant proof is candidate-role matching exposure",
            "The matching problem maps closely to their career evidence",
            "Best evidence is matching workflow ownership",
        ],
        "production_ml_infrastructure": [
            "Primary strength is production ML infrastructure",
            "Best reason to keep them high is ML systems delivery",
            "The strongest edge is production ML ownership",
            "Most useful proof is model/pipeline infrastructure",
            "Primary differentiator is applied ML systems work",
            "The production case is more concrete than the search case",
            "Best evidence is ML pipeline ownership",
            "The useful signal is production-grade ML execution",
        ],
        "strong_evaluation_mindset": [
            "Primary differentiator is evaluation mindset",
            "Best signal is explicit ranking-quality measurement",
            "The useful edge is evaluation discipline",
            "Most relevant proof is offline/online evaluation exposure",
            "The evaluation evidence makes the profile credible",
            "Best reason to rank them is measurement depth",
        ],
        "dense_sparse_hybrid_experience": [
            "Primary advantage is dense/sparse retrieval exposure",
            "Best evidence is hybrid retrieval experience",
            "The strongest retrieval clue is combining sparse and dense search",
            "Most relevant plus point is hybrid search familiarity",
            "Dense-plus-sparse retrieval is the useful signal",
            "The hybrid-search evidence is the differentiator",
        ],
        "senior_hands_on_ic": [
            "Primary signal is senior hands-on execution",
            "Best proof is senior IC implementation depth",
            "The useful edge is hands-on senior ownership",
            "Most relevant signal is senior builder experience",
            "The senior IC case is practical, not managerial only",
            "Hands-on senior delivery is the main proof",
            "The senior signal is implementation rather than oversight",
            "Senior IC execution is the useful differentiator",
            "The profile works because the seniority is still hands-on",
            "Practical senior engineering evidence drives the rank",
            "The key advantage is senior-level building",
            "Senior implementation depth is the main reason",
            "The strongest senior signal is coding-adjacent delivery",
            "Builder-style seniority is the useful proof",
            "The IC case is more concrete than the management case",
            "Hands-on ownership makes the seniority credible",
            "The rank comes from senior execution evidence",
            "Senior applied-ML delivery is the clearest proof",
            "Implementation-heavy seniority separates this profile",
            "The senior profile remains close to the code",
            "Direct system-building keeps this senior profile in scope",
            "The strongest point is senior technical execution",
            "Hands-on senior ML work is the differentiator",
            "The evidence reads like senior IC delivery",
            "Practical ownership, not title alone, supports the rank",
        ],
        "adjacent_but_strong_ml_infra": [
            "Useful adjacent profile",
            "The case here is ML/data infrastructure depth",
            "Best reason to keep them is pipeline strength",
            "Primary value is adjacent ML platform experience",
            "The profile is adjacent but technically useful",
            "Most useful signal is data/feature infrastructure",
        ],
        "nlp_embedding_experience": [
            "Primary signal is NLP and embedding experience",
            "Best evidence is language-model retrieval exposure",
            "The useful plus point is embedding-oriented ML work",
            "Most relevant clue is NLP systems experience",
            "The NLP evidence supports retrieval-adjacent fit",
        ],
        "general_ml_pipeline_experience": [
            "Primary reason is applied ML pipeline evidence",
            "Best available signal is career-backed ML systems work",
            "The profile is kept for applied ML implementation",
            "Most relevant proof is technical ML delivery",
            "The useful evidence is practical ML engineering",
        ],
    }
    return candidate_variant(candidate, openings.get(differentiator, openings["general_ml_pipeline_experience"]))


def concise_tools(candidate: dict, limit: int = 2) -> str:
    return format_terms(match_terms(career_text(candidate), TOOL_REASON_TERMS), limit)


def concise_eval(candidate: dict, limit: int = 2) -> str:
    return format_terms(match_terms(career_text(candidate), EVALUATION_REASON_TERMS), limit)


def concise_impact(candidate: dict) -> str:
    impacts = extract_impact_signals(candidate)
    return impacts[0] if impacts else ""


def exceptional_hireability_note(candidate: dict) -> str:
    s = signals(candidate)
    response_rate = float(s.get("recruiter_response_rate") or 0.0)
    interview = float(s.get("interview_completion_rate") if s.get("interview_completion_rate") is not None else -1)
    notice = float(s.get("notice_period_days") if s.get("notice_period_days") is not None else 999)
    verified = bool(s.get("verified_email")) and bool(s.get("verified_phone"))

    if notice <= 15 and s.get("open_to_work_flag"):
        return candidate_variant(candidate, [
            f"Availability is unusually clean with {int(notice)}-day notice.",
            f"Short {int(notice)}-day notice makes outreach practical.",
            f"Open-to-work status and {int(notice)}-day notice help hireability.",
        ])
    if response_rate >= 0.85:
        return candidate_variant(candidate, [
            f"Recruiter response rate is unusually high at {response_rate:.2f}.",
            f"Strong {response_rate:.2f} recruiter response rate improves reachability.",
            f"Outreach risk is lower with a {response_rate:.2f} recruiter response rate.",
        ])
    if interview >= 0.90:
        return candidate_variant(candidate, [
            f"Interview follow-through is unusually reliable at {interview:.2f}.",
            f"A {interview:.2f} interview completion rate strengthens process reliability.",
            f"High interview completion ({interview:.2f}) makes the profile easier to progress.",
        ])
    if s.get("open_to_work_flag") and notice <= 30 and verified:
        return candidate_variant(candidate, [
            "Open-to-work status, short notice, and verified contacts make outreach practical.",
            "Short notice plus verified contact channels improves immediate hireability.",
            "Verified contacts and sub-30-day notice make the process easier to start.",
        ])
    return ""


def differentiator_sentence(candidate: dict, differentiator: str, reason_codes: list[str]) -> str:
    opening = differentiator_opening(candidate, differentiator)
    tools = concise_tools(candidate)
    evals = concise_eval(candidate)
    impact = concise_impact(candidate)
    penalty = penalty_reasoning(reason_codes)

    if differentiator == "learning_to_rank_ownership":
        detail = "evolved a hand-tuned scoring/ranking layer toward learning-to-rank"
        support = []
        if evals:
            support.append(evals)
        if has_career(candidate, ["feature pipeline", "feature pipelines"]):
            support.append("feature-pipeline work")
        support_text = f", with {join_phrases(support, 3)}" if support else ""
        tail = candidate_variant(candidate, [
            "matching Redrob's ranking-quality improvement problem",
            "useful for Redrob's v2 ranking work",
            "close to Redrob's need for measurable search quality",
           "practical evidence for moving beyond hand-tuned scoring",
            "directly relevant to improving recruiter-facing search quality",
            "practical evidence for moving beyond hand-tuned scoring",
        ])
        return f"{opening}: {detail}{support_text}, {tail}."

    if differentiator == "bm25_to_semantic_migration":
        verb, _, hands_on = action_for_profile(candidate, SEARCH_SYSTEM_TERMS + SEARCH_RETRIEVAL_REASON_TERMS)
        action = action_word(verb, hands_on)
        support = []
        if tools:
            support.append(f"using {tools}")
        if evals:
            support.append(f"validated with {evals}")
        support_text = f" {', '.join(support)}" if support else ""
        return f"{opening}: {action.lower()} semantic retrieval over a BM25/Elasticsearch-style baseline{support_text}."

    if differentiator == "hands_on_retrieval_builder":
        system = system_label(match_terms(career_text(candidate), SEARCH_SYSTEM_TERMS), "retrieval/search system")
        tool_text = f" using {tools}" if tools else ""
        eval_text = f", backed by {evals}" if evals else ""
        return f"{opening}: implemented {system}{tool_text}{eval_text}, matching the retrieval-heavy part of Redrob's role."

    if differentiator == "relevance_labeling_eval_depth":
        eval_text = evals or "relevance labels and human/click feedback"
        return f"{opening}: designed relevance-labeling or judgment workflows with {eval_text}, close to Redrob's need to measure ranking quality."

    if differentiator == "large_scale_search_system":
        system = system_label(match_terms(career_text(candidate), SEARCH_SYSTEM_TERMS), "semantic-search/retrieval system")
        scale_text = f" at {impact} scale" if impact else " with production scale signals"
        tool_text = f" using {tools}" if tools else ""
        return f"{opening}: {system}{tool_text}{scale_text}, useful for a recruiter-facing search and matching stack."

    if differentiator == "recommendation_matching_ownership":
        eval_text = f" with {evals}" if evals else ""
        return f"{opening}: shipped or owned recommendation/personalization work{eval_text}, relevant to ranking candidates and roles rather than generic AI keyword matching."

    if differentiator == "candidate_job_matching_relevance":
        return f"{opening}: career evidence maps to matching marketplaces or candidate-role workflows, which is closer to Redrob's product problem than generic model building."

    if differentiator == "dense_sparse_hybrid_experience":
        tool_text = f" across {tools}" if tools else ""
        return f"{opening}: combines keyword/BM25-style search with vector or semantic retrieval{tool_text}, a practical base for Redrob's hybrid retrieval roadmap."

    if differentiator == "production_ml_infrastructure":
        system = system_label(match_terms(career_text(candidate), PRODUCTION_SYSTEM_TERMS), "production ML/feature-pipeline infrastructure")
        extra = f" with {tools}" if tools else ""
        return f"{opening}: built {system}{extra}; less search-specific than top retrieval profiles, but useful for shipping Redrob's applied intelligence layer."

    if differentiator == "strong_evaluation_mindset":
        eval_text = evals or "offline/online evaluation"
        return f"{opening}: explicit {eval_text} evidence shows they can judge ranking quality, not only build models or list AI tools."

    if differentiator == "senior_hands_on_ic":
        return f"{opening}: career history shows implementation verbs around ML/retrieval systems, keeping the profile closer to a coding IC than a pure manager."

    if differentiator == "adjacent_but_strong_ml_infra":
        infra = format_terms(match_terms(career_text(candidate), DATA_INFRA_REASON_TERMS + PRODUCTION_SYSTEM_TERMS), 3)
        return f"{opening}: strong {infra or 'ML/data infrastructure'} and feature-pipeline exposure, but ranked below candidates with direct search-ranking or retrieval ownership."

    if differentiator == "nlp_embedding_experience":
        tool_text = f" with {tools}" if tools else ""
        return f"{opening}: career history shows NLP/embedding work{tool_text}, useful for semantic matching but weaker than direct ranking-system ownership."

    base = "career history shows applied ML or pipeline work"
    if penalty:
        return f"{opening}: {base}, but {penalty[0].lower() + penalty[1:]}"
    return f"{opening}: {base}, enough for the shortlist but less differentiated than direct retrieval or ranking ownership."


def candidate_differentiators(candidate: dict, component_scores: dict[str, float] | None = None) -> list[str]:
    career = career_text(candidate)
    roles = role_text(candidate)
    years = float(profile(candidate).get("years_of_experience") or 0.0)
    diffs: list[str] = []

    has_ltr = has_any(career, ["learning-to-rank", "learning to rank", "ranking layer", "hand-tuned scoring"])
    has_bm25_migration = (
        has_any(career, ["semantic search", "embedding-based retrieval", "vector search"])
        and has_any(career, ["bm25", "elasticsearch", "opensearch", "keyword-search", "keyword search"])
    )
    has_retrieval = has_any(career, SEARCH_SYSTEM_TERMS + SEARCH_RETRIEVAL_REASON_TERMS)
    has_eval = has_any(career, EVALUATION_REASON_TERMS)
    has_relevance_eval = has_eval and has_any(
        career,
        ["relevance labeling", "relevance labels", "human judgments", "human relevance judgments", "click-through data"],
    )
    has_recommendation = has_any(career, RECOMMENDATION_SYSTEM_TERMS + RECOMMENDATION_REASON_TERMS)
    has_candidate_matching = has_any(career, ["candidate matching", "job matching", "marketplace matching"])
    has_dense_sparse = (
        has_any(career, ["hybrid retrieval", "hybrid search", "dense retrieval", "sparse retrieval"])
        or (has_any(career, ["bm25", "elasticsearch", "opensearch"]) and has_any(career, ["semantic search", "vector search", "faiss", "qdrant", "pinecone", "embeddings"]))
    )
    has_production = has_any(career, PRODUCTION_SYSTEM_TERMS + PRODUCTION_REASON_TERMS)
    has_infra = has_any(roles + " " + career, DATA_INFRA_REASON_TERMS)
    has_scale = bool(extract_impact_signals(candidate)) or has_any(career, ["large scale", "scale", "latency"])
    hands_on = has_hands_on_career(candidate)

    ordered_checks = [
        ("learning_to_rank_ownership", has_ltr),
        ("bm25_to_semantic_migration", has_bm25_migration),
        ("hands_on_retrieval_builder", has_retrieval and hands_on),
        ("relevance_labeling_eval_depth", has_relevance_eval),
        ("large_scale_search_system", has_scale and has_retrieval),
        ("recommendation_matching_ownership", has_recommendation),
        ("candidate_job_matching_relevance", has_candidate_matching),
        ("dense_sparse_hybrid_experience", has_dense_sparse),
        ("production_ml_infrastructure", has_production),
        ("strong_evaluation_mindset", has_eval),
        ("senior_hands_on_ic", 5 <= years <= 9 and hands_on and (component_scores or {}).get("hands_on_depth_score", 0.0) >= 0.75),
        ("adjacent_but_strong_ml_infra", has_infra),
        ("nlp_embedding_experience", has_any(career, NLP_SYSTEM_TERMS + NLP_LLM_REASON_TERMS)),
    ]
    for name, present in ordered_checks:
        if present and name not in diffs:
            diffs.append(name)
    if not diffs:
        diffs.append("general_ml_pipeline_experience")
    return diffs


def differentiator_label(diff: str) -> str:
    labels = {
        "learning_to_rank_ownership": "learning-to-rank ownership",
        "bm25_to_semantic_migration": "BM25-to-semantic migration",
        "hands_on_retrieval_builder": "hands-on retrieval building",
        "relevance_labeling_eval_depth": "relevance-labeling/evaluation depth",
        "large_scale_search_system": "large-scale search evidence",
        "recommendation_matching_ownership": "recommendation or matching ownership",
        "candidate_job_matching_relevance": "candidate-job matching relevance",
        "dense_sparse_hybrid_experience": "dense/sparse retrieval experience",
        "production_ml_infrastructure": "production ML infrastructure",
        "strong_evaluation_mindset": "ranking-quality measurement",
        "senior_hands_on_ic": "senior hands-on IC evidence",
        "adjacent_but_strong_ml_infra": "adjacent ML/data infrastructure",
        "nlp_embedding_experience": "NLP/embedding experience",
        "general_ml_pipeline_experience": "applied ML pipeline evidence",
    }
    return labels.get(diff, diff.replace("_", " "))


def differentiator_proof(candidate: dict, diff: str) -> str:
    career = career_text(candidate)
    tools = concise_tools(candidate, 2)
    evals = concise_eval(candidate, 2)
    impact = concise_impact(candidate)

    if diff == "learning_to_rank_ownership":
        support = []
        if evals:
            support.append(evals)
        if has_career(candidate, ["feature pipeline", "feature pipelines"]):
            support.append("feature-pipeline work")
        if impact:
            support.append(f"{impact} impact/scale evidence")
        return "moved a hand-tuned scoring layer toward learning-to-rank" + (f" with {join_phrases(support, 3)}" if support else "")
    if diff == "bm25_to_semantic_migration":
        support_parts = []
        if tools:
            support_parts.append(f"using {tools}")
        if evals:
            support_parts.append(f"validated with {evals}")
        if impact:
            support_parts.append(f"{impact} scale/impact evidence")
        return "shifted BM25/Elasticsearch-style search toward semantic retrieval" + (f" {', '.join(support_parts)}" if support_parts else "")
    if diff == "hands_on_retrieval_builder":
        system = system_label(match_terms(career, SEARCH_SYSTEM_TERMS), "retrieval/search systems")
        support_parts = []
        if tools:
            support_parts.append(f"using {tools}")
        if impact:
            support_parts.append(f"with {impact} scale/impact evidence")
        return f"implemented {system}" + (f" {', '.join(support_parts)}" if support_parts else "")
    if diff == "relevance_labeling_eval_depth":
        return f"built relevance/evaluation workflows around {evals or 'human judgments and click feedback'}"
    if diff == "large_scale_search_system":
        system = system_label(match_terms(career, SEARCH_SYSTEM_TERMS), "semantic retrieval")
        return f"worked on {system}" + (f" at {impact} scale" if impact else " with production-scale signals")
    if diff == "recommendation_matching_ownership":
        p = profile(candidate)
        title = str(p.get("current_title") or "candidate")
        years = float(p.get("years_of_experience") or 0.0)
        return f"{title} with {years:.1f} years owned recommendation/personalization work" + (f" measured with {evals}" if evals else "")
    if diff == "candidate_job_matching_relevance":
        return "has marketplace or candidate-role matching evidence close to Redrob's product problem"
    if diff == "dense_sparse_hybrid_experience":
        return "combines keyword/BM25 search with vector or semantic retrieval" + (f" across {tools}" if tools else "")
    if diff == "production_ml_infrastructure":
        system = system_label(match_terms(career, PRODUCTION_SYSTEM_TERMS), "production ML or feature-pipeline infrastructure")
        return f"built {system}" + (f" with {tools}" if tools else "")
    if diff == "strong_evaluation_mindset":
        return f"has explicit {evals or 'offline/online evaluation'} evidence for measuring ranking quality"
    if diff == "senior_hands_on_ic":
        p = profile(candidate)
        title = str(p.get("current_title") or "senior technical profile")
        years = float(p.get("years_of_experience") or 0.0)
        return f"{title} with {years:.1f} years has hands-on ML/retrieval delivery rather than pure management"
    if diff == "adjacent_but_strong_ml_infra":
        infra = format_terms(match_terms(career, DATA_INFRA_REASON_TERMS + PRODUCTION_SYSTEM_TERMS), 3)
        return f"brings {infra or 'ML/data infrastructure'} and feature-pipeline exposure"
    if diff == "nlp_embedding_experience":
        return "shows NLP/embedding work" + (f" with {tools}" if tools else "")
    p = profile(candidate)
    title = str(p.get("current_title") or "candidate")
    years = float(p.get("years_of_experience") or 0.0)
    return f"{title} with {years:.1f} years has applied ML pipeline evidence from career history"


def rank_tier(rank: int) -> str:
    if rank <= 5:
        return "elite"
    if rank <= 20:
        return "high"
    if rank <= 60:
        return "solid"
    return "cutoff"


def rank_reason_caveat(result: RerankResult, diffs: list[str], rank: int) -> str:
    candidate = result.row["candidate"]
    notice = float(signals(candidate).get("notice_period_days") or 0)
    primary = result.primary_differentiator
    if result.penalties:
        penalty = result.penalties[0].lower()
        if "adjacent" in penalty:
            return phrase_variant(candidate, "penalty-adjacent", [
                "good adjacent fit, but lacks the ranking-specific depth of higher ranks",
                "stronger on ML systems than search ranking",
                "included for production ML depth, not pure retrieval ownership",
                "relevant, but not as complete as candidates with BM25/retrieval/evaluation evidence",
                "less direct than the top search/retrieval profiles",
            ])
        if "notice" in penalty:
            return phrase_variant(candidate, "penalty-notice", [
                "long notice makes the process less immediate",
                "hireability is weaker because of the notice period",
                "availability is less clean than similarly strong profiles",
                "notice period is the main practical concern",
                "joining timeline is less attractive than shorter-notice candidates",
            ])
        return penalty
    if rank > 80:
        if primary in {"production_ml_infrastructure", "adjacent_but_strong_ml_infra", "nlp_embedding_experience", "general_ml_pipeline_experience"}:
            return phrase_variant(candidate, "cutoff-adjacent", [
                "less direct than the top search/retrieval profiles",
                "stronger on ML systems than search ranking",
                "included for production ML depth, not pure retrieval ownership",
                "good adjacent fit, but lacks the ranking-specific depth of higher ranks",
                "final-cut profile with useful systems depth rather than complete retrieval ownership",
                "NLP or ML systems proof is clearer than ranking ownership",
            ])
        if notice > 90:
            return phrase_variant(candidate, "cutoff-notice", [
                "notice period makes the process less immediate",
                "hireability is less clean than similarly strong candidates",
                "longer joining timeline keeps the profile near the cutoff",
                "availability is the main reason this sits lower",
            ])
        return phrase_variant(candidate, "cutoff-core", [
            "search-ranking proof is narrower than specialist profiles",
            "less complete than candidates with BM25, retrieval, and evaluation evidence",
            "kept for relevant systems work, but retrieval depth is thinner",
            "final-cut candidate with useful relevance signals, not a complete search owner",
            "ranking/retrieval evidence is present but less complete than higher profiles",
        ])
    if rank > 20:
        if "learning_to_rank_ownership" not in diffs and "bm25_to_semantic_migration" not in diffs:
            return phrase_variant(candidate, "mid-caveat", [
                "less direct than the top search/retrieval profiles",
                "stronger on ML systems than search ranking",
                "recommendation or production depth is clearer than retrieval ownership",
                "evaluation depth is lighter than the highest-ranked search profiles",
                "relevant, but not as complete as candidates with BM25/retrieval/evaluation evidence",
                "good applied-ML fit with less explicit ranking-system ownership",
            ])
    return ""


def career_context(candidate: dict) -> tuple[str, float]:
    p = profile(candidate)
    title = str(p.get("current_title") or "Applied ML profile")
    years = float(p.get("years_of_experience") or 0.0)
    if title.lower().startswith("junior "):
        title = title[7:].strip() or "Applied ML profile"
    return title, years


def phrase_variant(candidate: dict, key: str, variants: list[str]) -> str:
    seed = f"{candidate.get('candidate_id', '')}-{key}"
    index = sum((idx + 1) * ord(char) for idx, char in enumerate(seed)) % len(variants)
    return variants[index]


def hireability_tradeoff_note(result: RerankResult) -> str:
    if result.hireability_penalty < 0.035:
        return ""
    candidate = result.row["candidate"]
    codes = set(result.hireability_reason_codes)
    s = signals(candidate)
    notice = int(float(s.get("notice_period_days") or 0))

    if "notice_120_plus_not_open" in codes:
        return phrase_variant(candidate, "hire-note-120-not-open", [
            f"Strong technical fit, but {notice}-day notice and not-open status weaken immediate hireability.",
            f"Technical proof is convincing; {notice}-day notice plus not-open status is the practical concern.",
            f"Search evidence remains valuable, but {notice}-day notice and not-open status push the profile lower.",
        ])
    if "notice_120_plus" in codes:
        return phrase_variant(candidate, "hire-note-120", [
            f"Strong technical fit, but {notice}-day notice weakens immediate hireability.",
            f"Technical proof is convincing; joining timeline is the main practical concern at {notice} days.",
            f"{notice}-day notice keeps the profile below similarly strong, faster-moving candidates.",
        ])
    if "notice_90_plus" in codes:
        return phrase_variant(candidate, "hire-note-90", [
            f"Ranked lower mainly due to {notice}-day notice.",
            f"Technical fit is convincing; {notice}-day notice is the main practical tradeoff.",
            f"{notice}-day notice keeps the profile behind similar candidates with cleaner availability.",
        ])
    if {"low_recruiter_response", "very_slow_response_low_response_rate", "unverified_contacts_low_response"} & codes:
        return phrase_variant(candidate, "hire-note-reach", [
            "Strong search evidence, but reachability signals are weaker than nearby candidates.",
            "Technical fit is solid; recruiter response and contact signals make outreach riskier.",
            "Good technical proof, but weak response/contact signals reduce immediate hiring confidence.",
        ])
    if "weak_interview_completion" in codes:
        return phrase_variant(candidate, "hire-note-interview", [
            "Technical fit is useful, but interview follow-through is weaker than nearby candidates.",
            "Search/ML evidence remains relevant; process reliability is the practical concern.",
            "Good systems evidence, but low interview completion keeps the profile lower.",
        ])
    return phrase_variant(candidate, "hire-note-general", [
        "Technical fit is useful, but hireability signals are weaker than nearby candidates.",
        "Good technical evidence, with practical availability risk keeping the profile lower.",
        "Strong enough technically, though Redrob engagement signals are less clean.",
    ])


def realism_tradeoff_note(result: RerankResult) -> str:
    if result.evidence_realism_penalty < 0.04 and result.seniority_drift_penalty < 0.04:
        return ""
    candidate = result.row["candidate"]
    codes = set(result.evidence_realism_reason_codes + result.seniority_alignment_reason_codes)
    if "recent_leadership_heavy" in codes or "recent_leadership_ownership_with_weak_hands_on" in codes:
        return phrase_variant(candidate, "realism-leadership", [
            "Recent role wording leans leadership-heavy.",
            "Recent ownership language is less hands-on.",
            "Recent role evidence is less IC-clean.",
        ])
    if "tech_lead_aspiration_soft_risk" in codes or "tech_lead_or_architecture_drift_without_recent_hands_on" in codes:
        return phrase_variant(candidate, "realism-techlead", [
            "Senior-IC alignment is slightly less clean.",
            "Tech-lead direction is a small caveat.",
            "Recent direction is less purely IC.",
        ])
    if "weak_hireability_cluster" in codes:
        return phrase_variant(candidate, "realism-hire-cluster", [
            "Hireability signals are less clean.",
            "Engagement and availability are weaker.",
            "Practical outreach signals are less clean.",
        ])
    if any(
        code in codes
        for code in [
            "salary_range_inverted",
            "repeated_career_template",
            "same_candidate_repeated_role_template",
            "company_domain_description_mismatch",
        ]
    ):
        return phrase_variant(candidate, "realism-profile-consistency", [
            "Profile consistency is a minor caveat.",
            "Profile consistency is less clean.",
            "Some profile consistency signals are weaker.",
        ])
    return ""


def append_reasoning_note(reasoning: str, note: str, max_words: int = 38) -> str:
    if not note:
        return reasoning
    combined = f"{reasoning.rstrip('.;')}; {note}"
    if word_count(combined) <= max_words:
        return combined
    lead = re.split(r";\s+|(?<=[.!?])\s+", reasoning.strip())[0].strip()
    combined = f"{lead.rstrip('.;')}; {note}"
    if word_count(combined) <= max_words:
        return combined
    note_words = note.split()
    remaining = max_words - word_count(lead) - 1
    if remaining >= 8:
        short_note = " ".join(note_words[:remaining]).rstrip(" ,;:") + "."
        return f"{lead.rstrip('.;')}; {short_note}"
    return shorten_reasoning(reasoning)


def top_evidence_sentence(candidate: dict, diffs: list[str]) -> str:
    tools = concise_tools(candidate, 2)
    evals = concise_eval(candidate, 2)
    impact = concise_impact(candidate)
    secondary = set(diffs[1:3])

    if "bm25_to_semantic_migration" in diffs:
        templates = [
            "Developed semantic retrieval over a BM25/Elasticsearch-style baseline",
            "Moved keyword-style search toward semantic retrieval",
            "Modernized BM25-backed search with vector retrieval",
            "Built semantic search on top of an Elasticsearch/BM25 baseline",
            "Migrated search beyond BM25-style scoring",
            "Extended keyword search into embedding-based retrieval",
            "Reworked keyword retrieval into semantic search",
            "Added vector retrieval over a BM25-style baseline",
            "Converted baseline keyword search into semantic retrieval",
            "Upgraded Elasticsearch/BM25 search with embedding retrieval",
            "Built embedding retrieval over keyword-search foundations",
            "Developed vector search against an existing BM25 baseline",
            "Improved keyword search by adding semantic retrieval",
            "Shifted search quality work from BM25 toward embeddings",
            "Introduced semantic retrieval alongside BM25-style search",
            "Moved search beyond lexical matching into vector retrieval",
        ]
        sentence = phrase_variant(candidate, "bm25", templates)
        support = []
        if tools:
            support.append(phrase_variant(candidate, "bm25-tools", [
                f"with {tools}",
                f"using {tools} for the retrieval layer",
                f"where {tools} supported the search stack",
            ]))
        if evals:
            support.append(phrase_variant(candidate, "bm25-eval", [
                f"validated with {evals}",
                f"checked through {evals}",
                f"measured with {evals}",
            ]))
        if impact:
            support.append(f"{impact} scale/impact")
        if "learning_to_rank_ownership" in secondary:
            support.append("plus learning-to-rank ownership")
        return sentence + (f", {join_phrases(support, 3)}" if support else "") + "."

    if "learning_to_rank_ownership" in diffs:
        templates = [
            "Owned the ranking-layer move from hand-tuned scoring toward learning-to-rank",
            "Evolved a manual scoring layer into learning-to-rank",
            "Built ranking-quality infrastructure around learning-to-rank",
            "Improved a search ranking layer with learning-to-rank work",
            "Drove ranking-system evolution beyond rule-based scoring",
            "Developed learning-to-rank work around an existing scoring layer",
            "Refined search scoring through learning-to-rank ownership",
            "Moved search quality work from rules toward learning-to-rank",
            "Built evaluation-backed learning-to-rank workflows",
            "Designed feature and labeling workflows for learning-to-rank",
            "Created ranking infrastructure around learning-to-rank",
            "Improved relevance scoring through learning-to-rank work",
            "Delivered ranking-layer evolution from rules to LTR",
            "Guided ranking-quality work toward learning-to-rank",
            "Strengthened relevance scoring with learning-to-rank work",
            "Produced LTR-style ranking improvements over rule-based scoring",
            "Created search-ranking workflows with relevance labels",
            "Refactored manual ranking toward learning-to-rank methods",
            "Built ranking evaluation around LTR-style scoring",
        ]
        sentence = phrase_variant(candidate, "ltr", templates)
        support = []
        if evals:
            support.append(evals)
        if has_career(candidate, ["feature pipeline", "feature pipelines"]):
            support.append("feature-pipeline work")
        if impact:
            support.append(f"{impact} impact/scale")
        result = sentence + (f" with {join_phrases(support, 3)}" if support else "") + "."
        if word_count(result) < 22:
            result = result.rstrip(".") + "; " + phrase_variant(candidate, "ltr-short", [
                "directly useful for ranking-quality measurement and relevance improvement.",
                "good proof for improving search relevance beyond hand rules.",
                "valuable for measurable ranking improvements in a matching product.",
                "stronger than generic ML exposure because it touches ranking quality.",
                "especially relevant to moving from rule scoring toward measured ranking.",
            ])
        return result

    if "hands_on_retrieval_builder" in diffs:
        system = system_label(match_terms(career_text(candidate), SEARCH_SYSTEM_TERMS), "retrieval/search systems")
        templates = [
            f"Implemented {system}",
            f"Built {system}",
            f"Developed {system}",
            f"Shipped {system}",
            f"Designed {system}",
        ]
        sentence = phrase_variant(candidate, "retrieval", templates)
        support = []
        if tools:
            support.append(f"using {tools}")
        if evals:
            support.append(f"evaluated with {evals}")
        if impact:
            support.append(f"{impact} scale/impact")
        support_text = f", {join_phrases(support, 3)}" if support else ""
        return f"{sentence}{support_text}; " + phrase_variant(candidate, "retrieval-tail", [
            "directly relevant to retrieval-heavy candidate matching.",
            "good proof for production search-quality work.",
            "useful for matching systems where retrieval quality drives recruiter results.",
            "stronger search-system signal than generic AI application work.",
            "maps naturally to candidate discovery and relevance ranking.",
            "shows concrete retrieval ownership from career history.",
        ])

    if "recommendation_matching_ownership" in diffs:
        templates = [
            "Owned recommendation and personalization work",
            "Shipped recommendation-system work",
            "Delivered recommender-system ownership",
            "Built personalization workflows for recommendation systems",
            "Owned matching-oriented recommendation delivery",
            "Delivered recommendation work tied to product matching",
            "Shipped personalization features with evaluation support",
            "Handled recommender-system work for product ranking",
            "Developed recommendation workflows with matching relevance",
            "Owned recommendation delivery with ranking adjacency",
            "Worked on personalization systems measured in production",
            "Delivered matching-relevant recommender work",
        ]
        sentence = phrase_variant(candidate, "rec", templates)
        if evals and "measured" in sentence.lower():
            eval_text = f", evaluated through {evals}"
        elif evals:
            eval_text = f", measured with {evals}"
        else:
            eval_text = ""
        return sentence + eval_text + phrase_variant(candidate, "rec-tail", [
            ", useful for candidate-role matching.",
            ", a good bridge into recruiter-candidate relevance.",
            ", relevant to matching quality rather than generic AI work.",
            ", helpful for ranking people or items by fit.",
            ", giving practical matching-system experience.",
        ])

    if "production_ml_infrastructure" in diffs:
        system = system_label(match_terms(career_text(candidate), PRODUCTION_SYSTEM_TERMS), "production ML/feature-pipeline infrastructure")
        templates = [
            f"Built {system}",
            f"Delivered {system}",
            f"Owned {system}",
            f"Developed {system}",
            f"Shipped {system}",
        ]
        sentence = phrase_variant(candidate, "prod", templates)
        return sentence + (f" with {tools}" if tools else "") + "; " + phrase_variant(candidate, "prod-tail", [
            "stronger on ML delivery than pure search ranking.",
            "included for production ML depth, not pure retrieval ownership.",
            "useful where model serving and reliability matter.",
            "practical for applied AI systems with backend constraints.",
            "less direct than the top retrieval profiles, but operationally valuable.",
        ])

    if "adjacent_but_strong_ml_infra" in diffs:
        infra = format_terms(match_terms(career_text(candidate), DATA_INFRA_REASON_TERMS + PRODUCTION_SYSTEM_TERMS), 3)
        title, years = career_context(candidate)
        templates = [
            f"Built adjacent ML/data infrastructure around {infra or 'feature pipelines'}",
            f"{title} brings {infra or 'ML/data infrastructure'} across {years:.1f} years",
            f"Production AI support comes from {infra or 'feature-pipeline'} infrastructure",
            f"Applied ML infrastructure is the useful signal from this {years:.1f}-year profile",
            f"Feature-pipeline and data-system work make this {years:.1f}-year profile useful",
            f"ML infrastructure depth is the clearest reason to keep this {title} profile",
        ]
        return phrase_variant(candidate, "adjacent-final", templates) + phrase_variant(candidate, "adjacent-tail", [
            ", with less direct search-ranking ownership than stronger retrieval candidates.",
            ", stronger on ML systems than search ranking.",
            ", included for production ML depth rather than pure retrieval ownership.",
            ", good adjacent fit, but missing deeper ranking-specific proof.",
            ", relevant but less complete than BM25/retrieval/evaluation profiles.",
        ])

    if "nlp_embedding_experience" in diffs:
        title, years = career_context(candidate)
        lead = phrase_variant(candidate, "nlp-final", [
            "Developed NLP/embedding work",
            "Built NLP-oriented ML workflows",
            "Worked on embedding-oriented ML systems",
            "Delivered NLP systems experience",
            "Handled language-model or embedding workflows",
            "Applied NLP workflows to semantic matching",
            "Contributed embedding-focused ML work",
            "Supported language-model retrieval workflows",
            "Worked across NLP and embedding systems",
            "Delivered embedding-adjacent ML work",
            "Built language-processing ML workflows",
            "Handled NLP systems with retrieval relevance",
        ])
        return f"{lead}{f' with {tools}' if tools else ''}; " + phrase_variant(candidate, "nlp-tail", [
            f"{title} with {years:.1f} years has semantic-matching value, but lighter ranking ownership.",
            "embedding/NLP proof is clearer than search-ranking ownership.",
            "good NLP adjacency, but less complete than direct retrieval builders.",
            "relevant to semantic matching while still weaker on ranking-system ownership.",
            "useful for language-heavy matching, not a pure search-ranking specialist.",
        ])

    title, years = career_context(candidate)
    templates = [
        f"Applied ML pipeline work is the useful signal from this {years:.1f}-year {title} profile",
        f"{title} brings {years:.1f} years of applied ML pipeline work",
        f"Production AI support comes from {years:.1f} years of applied ML pipeline work",
        f"Applied ML delivery is present in this {years:.1f}-year {title} profile",
        f"ML pipeline work makes this {years:.1f}-year {title} profile useful",
        f"{years:.1f} years of ML pipeline work gives this {title} profile some value",
        f"Production ML delivery is the clearest signal in this {title} profile",
        f"This {title} profile is supported by applied ML pipeline work",
        f"Applied model/pipeline delivery keeps this {years:.1f}-year profile useful",
        f"ML systems delivery, rather than search ownership, is the useful signal here",
        f"The useful evidence is applied ML delivery from a {years:.1f}-year profile",
    ]
    return phrase_variant(candidate, "general-final", templates) + phrase_variant(candidate, "general-tail", [
        ", with direct retrieval evidence still limited.",
        ", stronger on ML systems than search ranking.",
        ", included for production ML depth rather than complete retrieval ownership.",
        ", useful as an adjacent profile, not a top retrieval specialist.",
        ", relevant but less complete than BM25/retrieval/evaluation profiles.",
    ])


def build_ranked_reasoning(result: RerankResult, rank: int) -> str:
    candidate = result.row["candidate"]
    diffs = candidate_differentiators(candidate, result.components)
    primary = result.primary_differentiator
    if primary not in diffs:
        diffs.insert(0, primary)
    tier = rank_tier(rank)
    caveat = rank_reason_caveat(result, diffs, rank)
    hireability_tradeoff = hireability_tradeoff_note(result)
    realism_tradeoff = realism_tradeoff_note(result)
    hireability = exceptional_hireability_note(candidate)

    reasoning = top_evidence_sentence(candidate, diffs)
    if hireability_tradeoff and (rank <= 60 or result.hireability_penalty >= 0.07):
        reasoning = append_reasoning_note(reasoning, hireability_tradeoff)
    elif realism_tradeoff and (rank <= 60 or result.evidence_realism_penalty >= 0.05 or result.seniority_drift_penalty >= 0.04):
        reasoning = append_reasoning_note(reasoning, realism_tradeoff)
    elif tier in {"solid", "cutoff"} and caveat:
        reasoning = f"{reasoning.rstrip('.;')} ; {caveat}."

    if not hireability_tradeoff and hireability and rank <= 40 and word_count(reasoning) <= 25:
        reasoning = f"{reasoning} {hireability}"

    reasoning = re.sub(r"\s+", " ", reasoning).replace(" ,", ",").replace(" ;", ";").strip()
    if word_count(reasoning) < 20:
        evals = concise_eval(candidate, 2)
        tools = concise_tools(candidate, 2)
        if evals and evals.lower() not in reasoning.lower():
            reasoning = f"{reasoning} Evaluation evidence includes {evals}."
        elif tools and tools.lower() not in reasoning.lower():
            reasoning = f"{reasoning} Tool evidence includes {tools}."
        else:
            addition = phrase_variant(candidate, "short-final", [
                "Useful for recruiter-facing search workflows.",
                "Practical for applied ML delivery.",
                "Helpful for ranking-quality improvement work.",
                "Adds practical ML systems depth.",
                "Relevant to applied retrieval systems.",
                "Supports production matching-system work.",
                "Adds useful systems depth for candidate discovery.",
            ])
            reasoning = f"{reasoning} {addition}"
    if word_count(reasoning) < 20:
        reasoning = f"{reasoning} Useful for retrieval-heavy matching work."
    return shorten_reasoning(reasoning)


def opening_for(profile_type: str, candidate: dict) -> str:
    variants = {
        "search_retrieval_specialist": [
            "Excellent retrieval fit",
            "Search/retrieval depth stands out",
            "Retrieval-system evidence looks strong",
            "Search stack fit is clear",
            "Semantic-search experience is relevant",
            "Hybrid retrieval background matches",
            "Information-retrieval profile looks strong",
            "Search relevance evidence stands out",
            "Retrieval implementation signal is convincing",
            "Dense/sparse search exposure is relevant",
            "Search engineering evidence is convincing",
            "Retrieval-heavy profile fits the mandate",
        ],
        "ranking_ltr_engineer": [
            "Strong ranking candidate",
            "Ranking-system ownership stands out",
            "Learning-to-rank evidence looks strong",
            "Relevance-evaluation depth is useful",
            "Ranking layer experience fits well",
            "Search-ranking background is compelling",
            "Evaluation-backed ranking work is visible",
            "Relevance workflow experience looks strong",
            "Ranking and labeling evidence matches",
            "LTR-style profile fits the role",
            "Ranking improvement signal is clear",
            "Search-quality ownership is relevant",
        ],
        "recommendation_matching_engineer": [
            "Strong matching/recommendation fit",
            "Recommendation-system evidence is relevant",
            "Matching-system background fits Redrob",
            "Personalization experience is useful",
            "Candidate-matching relevance is clear",
            "Recommendation ownership stands out",
            "Marketplace matching signal is useful",
            "Recommender-system profile looks strong",
            "Matching product experience fits",
            "Recommendation depth supports the rank",
            "Matching/ranking adjacency is valuable",
            "Recommendation work maps well",
        ],
        "production_ml_engineer": [
            "Production ML systems fit",
            "Applied ML ownership is clear",
            "Production engineering depth helps",
            "ML serving experience is relevant",
            "Feature-pipeline background is useful",
            "Backend ML signal is convincing",
            "Operational ML experience fits",
            "Production deployment evidence matters",
            "ML platform depth supports the rank",
            "Systems-minded ML profile fits",
            "Production-readiness signal is convincing",
            "Applied ML implementation depth shows",
        ],
        "nlp_llm_engineer": [
            "Good NLP/retrieval signal",
            "NLP systems background is relevant",
            "Transformer retrieval exposure helps",
            "LLM retrieval evidence is useful",
            "Embedding/NLP experience fits",
            "Applied NLP depth supports the rank",
            "NLP implementation signal is visible",
            "Retrieval-oriented NLP work stands out",
            "NLP-plus-search evidence is useful",
            "Language-model systems exposure helps",
            "NLP engineering signal is convincing",
            "Embedding workflow experience fits",
        ],
        "adjacent_data_infra_candidate": [
            "Adjacent data-infra fit",
            "Data platform strength is useful",
            "Infrastructure-adjacent profile remains relevant",
            "Feature-pipeline background helps",
            "Data engineering depth is valuable",
            "Analytics infrastructure signal is useful",
            "Pipeline-heavy profile is adjacent",
            "ML infrastructure background helps",
            "Data systems experience supports the rank",
            "Adjacent technical profile is credible",
            "Infrastructure experience keeps them in scope",
            "Data workflow depth is useful",
        ],
        "senior_hands_on_ic": [
            "Hands-on senior IC signal",
            "Senior builder profile fits",
            "Implementation depth stands out",
            "Hands-on ownership is visible",
            "Senior applied-ML profile looks strong",
            "Builder-oriented senior profile fits",
            "Practical system-building evidence helps",
            "Senior execution signal is clear",
            "Hands-on production experience matters",
            "Direct implementation evidence supports the rank",
            "Senior IC fit is credible",
            "Builder mindset shows in the career record",
        ],
        "general_ml_fit": [
            "Applied ML fit is present",
            "Career evidence keeps this profile relevant",
            "ML systems signal supports the rank",
            "Technical career evidence is usable",
            "Profile has enough applied ML signal",
            "Career record shows relevant ML exposure",
            "Applied engineering evidence is present",
            "Technical fit is defensible",
            "ML implementation signal is visible",
            "Profile remains in scope on evidence",
            "Career-backed ML relevance is present",
            "Engineering evidence supports selection",
        ],
    }
    choices = variants[profile_type]
    candidate_id = str(candidate.get("candidate_id") or "")
    index = sum(ord(char) for char in candidate_id) % len(choices)
    return choices[index]


def build_candidate_reasoning(candidate: dict, component_scores: dict[str, float], reason_codes: list[str]) -> str:
    differentiator = determine_primary_differentiator(candidate, component_scores)
    diffs = candidate_differentiators(candidate, component_scores)
    if differentiator not in diffs:
        diffs.insert(0, differentiator)
    reasoning = top_evidence_sentence(candidate, diffs)
    reasoning = re.sub(r"\s+", " ", reasoning).replace(" ,", ",").strip()
    if word_count(reasoning) < 20:
        evals = concise_eval(candidate, 2)
        tools = concise_tools(candidate, 2)
        if evals and evals.lower() not in reasoning.lower():
            reasoning = f"{reasoning} Evaluation evidence includes {evals}."
        elif tools and tools.lower() not in reasoning.lower():
            reasoning = f"{reasoning} Tool evidence includes {tools}."
        else:
            reasoning = f"{reasoning} Useful for applied ranking/retrieval work."
    if word_count(reasoning) < 20:
        reasoning = f"{reasoning} This is useful for the ranking/retrieval role."
    return shorten_reasoning(reasoning)


def career_evidence_score(candidate: dict) -> float:
    career = career_text(candidate)
    summary = summary_text(candidate)
    skills = skills_text(candidate)
    categories = category_matches(career)
    has_core = bool(categories["ranking_retrieval"] or categories["recommendation_matching"])
    category_count = sum(1 for hits in categories.values() if hits)
    has_production_or_eval = bool(categories["production"] or categories["evaluation"])

    if has_core and has_production_or_eval and category_count >= 3:
        return 1.0
    if has_core and category_count >= 2:
        return 0.85
    if categories["production"] and categories["python_data_ml"]:
        return 0.75
    if has_any(career, ["data pipeline", "data pipelines", "analytics", "machine learning", "feature engineering", "backend", "spark", "airflow"]):
        return 0.45
    if has_any(summary + " " + skills, AI_SKILL_TERMS):
        return 0.20
    return 0.0


def jd_pillar_score(candidate: dict) -> float:
    career = career_text(candidate)
    summary = summary_text(candidate)
    skills = skills_text(candidate)
    pillars = [
        RANKING_RETRIEVAL_TERMS,
        RECOMMENDATION_MATCHING_TERMS,
        EMBEDDING_VECTOR_TERMS,
        PRODUCTION_TERMS,
        EVALUATION_TERMS,
        PYTHON_DATA_ML_TERMS,
    ]
    scores = []
    for terms in pillars:
        if has_any(career, terms):
            scores.append(1.0)
        elif has_any(summary, terms):
            scores.append(0.5)
        elif has_any(skills, terms):
            scores.append(0.25)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def hands_on_depth_score(candidate: dict) -> float:
    career = career_text(candidate)
    core = has_any(career, RANKING_RETRIEVAL_TERMS + RECOMMENDATION_MATCHING_TERMS + EMBEDDING_VECTOR_TERMS)
    hands_on = has_any(career, HANDS_ON_VERBS)
    leadership = has_any(career, LEADERSHIP_VERBS)
    production = has_any(career, PRODUCTION_TERMS + PYTHON_DATA_ML_TERMS)

    if core and hands_on:
        return 1.0
    if core and leadership and hands_on:
        return 1.0
    if production and hands_on:
        return 0.75
    if core and leadership:
        return 0.58
    if hands_on:
        return 0.45
    if leadership:
        return 0.32
    return 0.10


def seniority_fit_score(candidate: dict, career_score: float, hands_on_score: float) -> float:
    years = float(profile(candidate).get("years_of_experience") or 0.0)
    if 5 <= years <= 9:
        return 1.0
    if 4 <= years < 5:
        return 0.75
    if 9 < years <= 12:
        return 0.85
    if 3 <= years < 4:
        return 0.55
    if years > 12:
        if career_score >= 0.9 and hands_on_score >= 0.85:
            return 0.85
        return 0.65
    return 0.20


def redrob_score(candidate: dict) -> float:
    s = signals(candidate)
    score = 0.45
    response_rate = float(s.get("recruiter_response_rate") or 0.0)
    response_hours = float(s.get("avg_response_time_hours") or 999.0)
    notice = float(s.get("notice_period_days") or 180.0)
    interview = float(s.get("interview_completion_rate") if s.get("interview_completion_rate") is not None else -1)
    offer = float(s.get("offer_acceptance_rate") if s.get("offer_acceptance_rate") is not None else -1)

    if s.get("open_to_work_flag"):
        score += 0.12
    else:
        score -= 0.08
    if response_rate >= 0.5:
        score += 0.14
    elif response_rate < 0.2:
        score -= 0.16
    if response_hours <= 48:
        score += 0.10
    elif response_hours > 168:
        score -= 0.12
    if notice <= 30:
        score += 0.10
    elif notice > 90:
        score -= 0.10
    if interview >= 0.7:
        score += 0.12
    elif 0 <= interview < 0.3:
        score -= 0.12
    if offer >= 0.5 or offer == -1:
        score += 0.06
    if s.get("verified_email") or s.get("verified_phone"):
        score += 0.08
    if s.get("willing_to_relocate"):
        score += 0.06

    last_active = parse_date(s.get("last_active_date"))
    if last_active and (REFERENCE_DATE - last_active).days > 180:
        score -= 0.08
    return clamp(score)


def location_availability_score(candidate: dict) -> float:
    p = profile(candidate)
    s = signals(candidate)
    country = normalize(p.get("country"))
    location = normalize(p.get("location"))
    work_mode = normalize(s.get("preferred_work_mode"))
    willing = bool(s.get("willing_to_relocate"))
    welcome = ["pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "ncr", "gurgaon", "gurugram"]

    if country and country != "india":
        return 0.55 if willing else 0.40
    if any(city in location for city in welcome):
        return 1.0
    if willing:
        return 0.85
    if work_mode in {"hybrid", "onsite"}:
        return 0.70
    return 0.75


def parse_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def penalty_factors(candidate: dict, career_score: float, hands_on_score: float) -> tuple[float, list[str], list[str]]:
    factor = 1.0
    penalties: list[str] = []
    reason_codes: list[str] = []
    career = career_text(candidate)
    summary = summary_text(candidate)
    skills = skills_text(candidate)
    roles = role_text(candidate)
    ai_skill_count = len(set(match_terms(skills, AI_SKILL_TERMS)))
    career_ai_count = len(set(match_terms(career, RANKING_RETRIEVAL_TERMS + RECOMMENDATION_MATCHING_TERMS + EMBEDDING_VECTOR_TERMS)))

    if ai_skill_count >= 5 and career_ai_count <= 1:
        factor *= 0.55
        penalties.append("unsupported AI skills")
        reason_codes.append("unsupported_ai_skills_penalty")

    if career_score <= 0.45 and has_any(summary + " " + skills, AI_SKILL_TERMS) and not has_any(career, ALL_CAREER_REASON_TERMS):
        factor *= 0.72
        penalties.append("weak direct JD evidence in career history")
        reason_codes.append("weak_direct_jd_career_evidence_penalty")

    if (
        career_score >= 0.75
        and has_any(career + " " + summary, WEAK_RANKING_DEPTH_TERMS)
        and not has_any(career, DIRECT_SEARCH_EVALUATION_EVIDENCE_TERMS)
    ):
        factor *= 0.70
        penalties.append("self-described weak ranking/retrieval depth")
        reason_codes.append("weak_ranking_depth_language_penalty")

    if has_any(summary, GENAI_EXPLORER_TERMS) and career_score <= 0.45:
        factor *= 0.50
        penalties.append("GenAI explorer language with weak career evidence")
        reason_codes.append("genai_explorer_penalty")

    is_adjacent = has_any(roles, ADJACENT_TECH_TITLES)
    has_retrieval_ownership = has_any(career, RANKING_RETRIEVAL_TERMS + RECOMMENDATION_MATCHING_TERMS)
    if is_adjacent and not has_retrieval_ownership:
        factor *= 0.80
        penalties.append("adjacent technical profile without retrieval/ranking ownership")
        reason_codes.append("adjacent_technical_no_retrieval_penalty")

    leadership = has_any(career, LEADERSHIP_VERBS)
    hands_on = has_any(career, HANDS_ON_VERBS)
    core = has_any(career, RANKING_RETRIEVAL_TERMS + RECOMMENDATION_MATCHING_TERMS + EMBEDDING_VECTOR_TERMS)
    if core and leadership and not hands_on:
        factor *= 0.88
        penalties.append("leadership-only retrieval evidence")
        reason_codes.append("leadership_only_penalty")

    return factor, penalties, reason_codes


def strong_technical_candidate(career_score: float, pillar_score: float, primary_differentiator: str) -> bool:
    return (
        career_score >= 0.75
        or pillar_score >= 0.75
        or primary_differentiator in {
            "learning_to_rank_ownership",
            "bm25_to_semantic_migration",
            "hands_on_retrieval_builder",
            "large_scale_search_system",
            "relevance_labeling_eval_depth",
        }
    )


def hireability_risk(candidate: dict, strong_technical: bool = False) -> tuple[float, list[str]]:
    s = signals(candidate)
    p = profile(candidate)
    penalty = 0.0
    reason_codes: list[str] = []

    notice = float(s.get("notice_period_days") if s.get("notice_period_days") is not None else 0)
    open_to_work = bool(s.get("open_to_work_flag"))
    response_rate = float(s.get("recruiter_response_rate") if s.get("recruiter_response_rate") is not None else 0.0)
    response_hours = float(s.get("avg_response_time_hours") if s.get("avg_response_time_hours") is not None else 999.0)
    interview = float(s.get("interview_completion_rate") if s.get("interview_completion_rate") is not None else -1.0)
    verified_email = bool(s.get("verified_email"))
    verified_phone = bool(s.get("verified_phone"))
    github_activity = float(s.get("github_activity_score") if s.get("github_activity_score") is not None else -1.0)
    country = normalize(p.get("country"))
    preferred_work_mode = normalize(s.get("preferred_work_mode"))
    willing_to_relocate = bool(s.get("willing_to_relocate"))

    if notice >= 120:
        if not open_to_work:
            penalty += 0.14
            reason_codes.append("notice_120_plus_not_open")
        else:
            penalty += 0.09
            reason_codes.append("notice_120_plus")
    elif notice >= 90:
        penalty += 0.04
        reason_codes.append("notice_90_plus")
        if not open_to_work:
            penalty += 0.03
            reason_codes.append("notice_90_plus_not_open")
    elif notice >= 60:
        penalty += 0.015
        reason_codes.append("notice_60")

    if not open_to_work:
        if notice >= 90:
            penalty += 0.02
            reason_codes.append("not_open_long_notice")
        elif strong_technical:
            penalty += 0.01
            reason_codes.append("not_open_small_risk")
        else:
            penalty += 0.02
            reason_codes.append("not_open")

    if response_rate < 0.30:
        penalty += 0.045
        reason_codes.append("low_recruiter_response")
    elif response_rate < 0.50:
        penalty += 0.018
        reason_codes.append("below_average_recruiter_response")

    if response_hours > 168 and response_rate < 0.50:
        penalty += 0.04
        reason_codes.append("very_slow_response_low_response_rate")
    elif response_hours > 72:
        penalty += 0.015
        reason_codes.append("slow_response_time")

    if 0 <= interview < 0.50:
        penalty += 0.05
        reason_codes.append("weak_interview_completion")
    elif 0.50 <= interview < 0.70:
        penalty += 0.02
        reason_codes.append("below_average_interview_completion")

    if not verified_email and not verified_phone:
        if response_rate < 0.50:
            penalty += 0.065
            reason_codes.append("unverified_contacts_low_response")
        else:
            penalty += 0.04
            reason_codes.append("unverified_contacts")
    elif not verified_email or not verified_phone:
        penalty += 0.005
        reason_codes.append("partially_verified_contact")

    if github_activity == -1:
        penalty += 0.005
        reason_codes.append("github_activity_unknown")
    elif github_activity == 0 and not strong_technical:
        penalty += 0.01
        reason_codes.append("github_activity_zero_weak_technical")

    if country and country != "india":
        if not willing_to_relocate:
            penalty += 0.025 if strong_technical else 0.04
            reason_codes.append("outside_india_not_relocating")
    if preferred_work_mode == "remote" and not willing_to_relocate:
        penalty += 0.01
        reason_codes.append("remote_only_not_relocating")

    cap = 0.15 if strong_technical else 0.20
    return min(penalty, cap), unique_ordered(reason_codes, 12)


SENIORITY_DRIFT_TERMS = [
    "tech-lead roles",
    "tech lead roles",
    "tech-lead role",
    "tech lead role",
    "architecture role",
    "architecture roles",
    "architecture track",
    "moved into architecture",
    "architecture/tech lead",
    "architectural roadmap",
    "roadmap",
    "strategy",
    "team lead",
    "leadership role",
    "management track",
]

RECENT_LEADERSHIP_OWNERSHIP_TERMS = LEADERSHIP_VERBS + [
    "ownership",
    "owner",
    "tech lead",
    "team lead",
    "architect",
    "architecture",
    "roadmap",
    "strategy",
    "stakeholder",
    "stakeholders",
    "planning",
    "reviews",
    "reviewed",
    "people management",
]

STRONG_RECENT_HANDS_ON_VERBS = [
    "built",
    "implemented",
    "developed",
    "designed",
    "deployed",
    "shipped",
    "optimized",
    "debugged",
    "integrated",
    "coded",
    "migrated",
]

CURRENT_ROLE_BUILDER_VERBS = [
    "built",
    "implemented",
    "developed",
    "deployed",
    "shipped",
    "coded",
    "migrated",
]

RECENT_SYSTEM_CONTEXT_TERMS = (
    RANKING_RETRIEVAL_TERMS
    + RECOMMENDATION_MATCHING_TERMS
    + EMBEDDING_VECTOR_TERMS
    + EVALUATION_TERMS
    + PRODUCTION_TERMS
    + [
        "ranking pipeline",
        "search system",
        "evaluation framework",
        "feature pipeline",
        "embedding pipeline",
        "api",
        "model serving",
        "deployment",
        "retrieval",
        "search",
        "ranking",
        "pipeline",
        "service",
    ]
)


def recent_career_text(candidate: dict, months: int = 18) -> str:
    threshold = (REFERENCE_DATE.year * 12 + REFERENCE_DATE.month) - months
    parts: list[str] = []
    for job in candidate.get("career_history", []):
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) or REFERENCE_DATE
        if not end:
            continue
        end_month = end.year * 12 + end.month
        is_current = bool(job.get("is_current"))
        if is_current or end_month >= threshold:
            parts.extend(
                str(job.get(key) or "")
                for key in ["title", "industry", "description", "company"]
            )
    return normalize(" ".join(parts))


def counted_recent_hands_on_verbs(text: str) -> list[str]:
    verbs = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        normalized = normalize(sentence)
        for verb in STRONG_RECENT_HANDS_ON_VERBS:
            if verb == "designed":
                if term_in_text(normalized, verb) and has_any(normalized, RECENT_SYSTEM_CONTEXT_TERMS):
                    verbs.append(verb)
                continue
            if term_in_text(normalized, verb):
                verbs.append(verb)
    return verbs


def recent_hands_on_score(candidate: dict) -> tuple[float, list[str]]:
    recent = recent_career_text(candidate, months=18)
    if not recent:
        return 0.0, []
    verbs = unique_ordered(counted_recent_hands_on_verbs(recent))
    score = min(1.0, len(verbs) / 3.0)
    reason_codes = ["recent_hands_on_evidence_present"] if score >= 0.34 else []
    return score, reason_codes


def recent_leadership_score(candidate: dict) -> float:
    recent = recent_career_text(candidate, months=18)
    if not recent:
        return 0.0
    hits = unique_ordered(match_terms(recent, RECENT_LEADERSHIP_OWNERSHIP_TERMS))
    return min(1.0, len(hits) / 3.0)


def current_or_latest_career_text(candidate: dict) -> str:
    current_jobs = [
        job for job in candidate.get("career_history", [])
        if bool(job.get("is_current"))
    ]
    if not current_jobs:
        jobs = list(candidate.get("career_history", []))
        if jobs:
            current_jobs = [
                max(
                    jobs,
                    key=lambda job: (
                        parse_date(job.get("end_date"))
                        or parse_date(job.get("start_date"))
                        or REFERENCE_DATE
                    ),
                )
            ]
    parts: list[str] = []
    for job in current_jobs:
        parts.extend(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        )
    return normalize(" ".join(parts))


def has_recent_hands_on_production(candidate: dict) -> bool:
    recent = recent_career_text(candidate, months=18)
    if not recent:
        return False
    hands_score, _ = recent_hands_on_score(candidate)
    has_hands_on = hands_score > 0.0
    has_production_or_core = has_any(
        recent,
        PRODUCTION_TERMS
        + RANKING_RETRIEVAL_TERMS
        + RECOMMENDATION_MATCHING_TERMS
        + EMBEDDING_VECTOR_TERMS
        + EVALUATION_TERMS,
    )
    return has_hands_on and has_production_or_core


def has_recent_strong_hands_on_production(candidate: dict) -> bool:
    recent = recent_career_text(candidate, months=18)
    if not recent:
        return False
    hands_score, _ = recent_hands_on_score(candidate)
    has_hands_on = hands_score >= 0.66
    has_production_or_core = has_any(
        recent,
        PRODUCTION_TERMS
        + RANKING_RETRIEVAL_TERMS
        + RECOMMENDATION_MATCHING_TERMS
        + EMBEDDING_VECTOR_TERMS
        + EVALUATION_TERMS,
    )
    return has_hands_on and has_production_or_core


def has_current_builder_evidence(candidate: dict) -> bool:
    current = current_or_latest_career_text(candidate)
    if not current:
        return False
    has_builder_verb = has_any(current, CURRENT_ROLE_BUILDER_VERBS)
    has_production_or_core = has_any(
        current,
        PRODUCTION_TERMS
        + RANKING_RETRIEVAL_TERMS
        + RECOMMENDATION_MATCHING_TERMS
        + EMBEDDING_VECTOR_TERMS
        + EVALUATION_TERMS,
    )
    return has_builder_verb and has_production_or_core


def seniority_alignment_adjustment(
    candidate: dict,
    strong_technical: bool,
    hireability_penalty: float,
    evidence_realism_penalty: float,
) -> tuple[float, float, list[str]]:
    profile_text = normalize(
        " ".join(
            str(profile(candidate).get(key) or "")
            for key in ["headline", "summary", "current_title"]
        )
    )
    if not strong_technical:
        return 0.0, 0.0, []

    reason_codes: list[str] = []
    drift_penalty = 0.0
    senior_ic_bonus = 0.0
    has_drift_language = has_any(profile_text, SENIORITY_DRIFT_TERMS)
    explicit_tech_lead_roles = has_any(profile_text, ["tech-lead roles", "tech lead roles"])
    hands_score, hands_reason_codes = recent_hands_on_score(candidate)
    leadership_score = recent_leadership_score(candidate)
    recent_leadership = leadership_score >= 0.34
    recent_hands_on = has_recent_hands_on_production(candidate)
    recent_strong_hands_on = has_recent_strong_hands_on_production(candidate)
    current_builder_evidence = has_current_builder_evidence(candidate)

    penalty_options: list[float] = []

    if recent_leadership:
        if recent_strong_hands_on:
            penalty_options.append(0.025)
            reason_codes.append("recent_leadership_with_hands_on_builder_evidence")
            reason_codes.extend(hands_reason_codes)
        elif hands_score >= 0.34:
            penalty_options.append(0.010)
            reason_codes.append("recent_leadership_heavy")
            reason_codes.extend(hands_reason_codes)
        else:
            penalty_options.append(0.040)
            reason_codes.append("recent_leadership_heavy")

    if has_drift_language:
        if explicit_tech_lead_roles:
            if current_builder_evidence:
                reason_codes.append("tech_lead_aspiration_offset_by_current_builder_evidence")
            else:
                penalty_options.append(0.050 if not recent_hands_on else 0.015)
                reason_codes.append("tech_lead_aspiration_soft_risk")
        elif recent_hands_on:
            penalty_options.append(0.015)
            reason_codes.append("architecture_or_strategy_language_with_recent_hands_on")
        else:
            penalty_options.append(0.050)
            reason_codes.append("tech_lead_or_architecture_drift_without_recent_hands_on")

    if penalty_options:
        drift_penalty = max(penalty_options)

    explicit_senior_ic = has_any(profile_text, ["senior ic", "senior individual contributor"])
    if (
        explicit_senior_ic
        and recent_hands_on
        and not has_drift_language
        and hireability_penalty == 0
        and evidence_realism_penalty <= 0.02
    ):
        senior_ic_bonus = 0.035
        reason_codes.append("explicit_senior_ic_hands_on_alignment")

    return drift_penalty, senior_ic_bonus, reason_codes


def candidate_debug_flags(
    candidate: dict,
    realism_index: EvidenceRealismIndex | None,
    nontechnical_penalty_total: float = 0.0,
    weak_count: int | None = None,
    repeated_template_details: tuple[float, int, bool, list[str]] | None = None,
    hands_score: float | None = None,
    leadership_score: float | None = None,
) -> dict[str, object]:
    if repeated_template_details is None:
        repeated_template_details = repeated_template_stats(candidate, realism_index)
    repeated_penalty, repeated_count, same_repeated, _ = repeated_template_details
    if weak_count is None:
        weak_count = weak_hireability_signal_count(candidate)
    if hands_score is None:
        hands_score, _ = recent_hands_on_score(candidate)
    if leadership_score is None:
        leadership_score = recent_leadership_score(candidate)
    profile_text = normalize(
        " ".join(
            str(profile(candidate).get(key) or "")
            for key in ["headline", "summary", "current_title"]
        )
    )
    tech_lead_risk = has_any(profile_text, ["tech-lead roles", "tech lead roles", "architecture roles"])
    return {
        "salary_range_inverted": salary_range_inverted(candidate),
        "weak_hireability_signal_count": weak_count,
        "repeated_career_template_count": repeated_count,
        "same_candidate_repeated_template": same_repeated,
        "company_domain_description_mismatch": has_domain_mismatch(candidate),
        "recent_hands_on_score": hands_score,
        "recent_leadership_score": leadership_score,
        "recent_leadership_heavy": leadership_score >= 0.67 and hands_score < 0.67,
        "tech_lead_aspiration_soft_risk": tech_lead_risk,
        "repeated_career_template_penalty": repeated_penalty,
        "nontechnical_penalty_total": nontechnical_penalty_total,
    }


def rerank_row(row: dict, realism_index: EvidenceRealismIndex) -> RerankResult:
    candidate = row["candidate"]
    original_hybrid = float(row.get("hybrid_score") or 0.0)
    career_score = career_evidence_score(candidate)
    pillar_score = jd_pillar_score(candidate)
    hands_score = hands_on_depth_score(candidate)
    seniority_score = seniority_fit_score(candidate, career_score, hands_score)
    behavior_score = redrob_score(candidate)
    location_score = location_availability_score(candidate)

    base = (
        0.25 * original_hybrid
        + 0.25 * career_score
        + 0.20 * pillar_score
        + 0.10 * hands_score
        + 0.08 * seniority_score
        + 0.07 * behavior_score
        + 0.05 * location_score
    )
    penalty_factor, penalties, reason_codes = penalty_factors(candidate, career_score, hands_score)
    components = {
        "career_evidence_score": career_score,
        "jd_pillar_score": pillar_score,
        "hands_on_depth_score": hands_score,
        "seniority_fit_score": seniority_score,
        "redrob_score": behavior_score,
        "location_availability_score": location_score,
        "penalty_factor": penalty_factor,
    }
    primary = determine_primary_differentiator(candidate, components)
    strong_technical = strong_technical_candidate(career_score, pillar_score, primary)
    hireability_penalty, hireability_reason_codes = hireability_risk(candidate, strong_technical)
    weak_count = weak_hireability_signal_count(candidate)
    description_records = career_description_records(candidate)
    repeated_details = repeated_template_stats(candidate, realism_index, records=description_records)
    realism_penalty, realism_reason_codes = evidence_realism_penalty(
        candidate,
        realism_index,
        strong_technical,
        weak_hireability_count=weak_count,
        records=description_records,
        repeated_template_details=repeated_details,
    )
    seniority_drift_penalty, senior_ic_bonus, seniority_alignment_reason_codes = seniority_alignment_adjustment(
        candidate,
        strong_technical,
        hireability_penalty,
        realism_penalty,
    )
    nontechnical_penalty_raw = hireability_penalty + realism_penalty + seniority_drift_penalty
    nontechnical_penalty_cap = 0.18 if strong_technical else 0.30
    nontechnical_penalty_total = min(nontechnical_penalty_raw, nontechnical_penalty_cap)
    final = clamp(
        (base * penalty_factor)
        - nontechnical_penalty_total
        + senior_ic_bonus
    )
    components["hireability_penalty"] = hireability_penalty
    components["evidence_realism_penalty"] = realism_penalty
    components["seniority_drift_penalty"] = seniority_drift_penalty
    components["senior_ic_alignment_bonus"] = senior_ic_bonus
    components["nontechnical_penalty_total"] = nontechnical_penalty_total
    debug_flags = candidate_debug_flags(
        candidate,
        realism_index,
        nontechnical_penalty_total=nontechnical_penalty_total,
        weak_count=weak_count,
        repeated_template_details=repeated_details,
    )
    return RerankResult(
        candidate_id=str(candidate["candidate_id"]),
        original_rank=int(row.get("rank") or 0),
        original_hybrid_score=original_hybrid,
        final_score=final,
        components=components,
        penalties=penalties,
        reason_codes=reason_codes,
        hireability_penalty=hireability_penalty,
        hireability_reason_codes=hireability_reason_codes,
        evidence_realism_penalty=realism_penalty,
        evidence_realism_reason_codes=realism_reason_codes,
        seniority_drift_penalty=seniority_drift_penalty,
        senior_ic_alignment_bonus=senior_ic_bonus,
        seniority_alignment_reason_codes=seniority_alignment_reason_codes,
        primary_differentiator=primary,
        reasoning=build_candidate_reasoning(candidate, components, reason_codes),
        debug_flags=debug_flags,
        row=row,
    )


def load_top2000(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_rejected_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rejected = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            cid = payload.get("candidate_id")
            if cid:
                rejected.add(str(cid))
    return rejected


def ranked_output_rows(results: list[RerankResult]) -> list[tuple[int, RerankResult, float]]:
    rows: list[tuple[int, RerankResult, float]] = []
    previous_score: float | None = None
    for rank, result in enumerate(results, start=1):
        score = round(result.final_score, 6)
        if previous_score is not None and score >= previous_score:
            score = max(0.0, previous_score - 0.000001)
        previous_score = score
        rows.append((rank, result, score))
    return rows


def write_submission(results: list[RerankResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, result, score in ranked_output_rows(results):
            writer.writerow([result.candidate_id, rank, f"{score:.6f}", result.reasoning])


def write_top100_jsonl(results: list[RerankResult], path: Path, candidates_path: Path) -> None:
    target_ids = [result.candidate_id for result in results]
    target_set = set(target_ids)
    raw_by_id: dict[str, str] = {}

    with candidates_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            payload = json.loads(line)
            candidate_id = str(payload.get("candidate_id") or "")
            if candidate_id in target_set:
                raw_by_id[candidate_id] = line.rstrip("\r\n")
                if len(raw_by_id) == len(target_set):
                    break

    missing = [candidate_id for candidate_id in target_ids if candidate_id not in raw_by_id]
    if missing:
        raise ValueError(f"Top100 candidates missing from {candidates_path}: {missing[:5]}")

    with path.open("w", encoding="utf-8") as handle:
        for candidate_id in target_ids:
            handle.write(raw_by_id[candidate_id] + "\n")


def write_debug(results: list[RerankResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "candidate_id",
            "original_rank",
            "original_hybrid_score",
            "final_score",
            "primary_differentiator",
            "career_evidence_score",
            "jd_pillar_score",
            "hands_on_depth_score",
            "seniority_fit_score",
            "redrob_score",
            "location_availability_score",
            "penalty_factor",
            "hireability_penalty",
            "evidence_realism_penalty",
            "seniority_drift_penalty",
            "senior_ic_alignment_bonus",
            "nontechnical_penalty_total",
            "penalties",
            "reason_codes",
            "hireability_reason_codes",
            "evidence_realism_reason_codes",
            "seniority_alignment_reason_codes",
            "salary_range_inverted",
            "weak_hireability_signal_count",
            "repeated_career_template_count",
            "same_candidate_repeated_template",
            "company_domain_description_mismatch",
            "recent_hands_on_score",
            "recent_leadership_score",
            "recent_leadership_heavy",
            "tech_lead_aspiration_soft_risk",
            "notice_period_days",
            "open_to_work_flag",
            "recruiter_response_rate",
            "avg_response_time_hours",
            "interview_completion_rate",
            "verified_email",
            "verified_phone",
            "github_activity_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            candidate = result.row["candidate"]
            s = signals(candidate)
            row = {
                "candidate_id": result.candidate_id,
                "original_rank": result.original_rank,
                "original_hybrid_score": f"{result.original_hybrid_score:.6f}",
                "final_score": f"{result.final_score:.6f}",
                "primary_differentiator": result.primary_differentiator,
                "hireability_penalty": f"{result.hireability_penalty:.6f}",
                "evidence_realism_penalty": f"{result.evidence_realism_penalty:.6f}",
                "seniority_drift_penalty": f"{result.seniority_drift_penalty:.6f}",
                "senior_ic_alignment_bonus": f"{result.senior_ic_alignment_bonus:.6f}",
                "nontechnical_penalty_total": f"{float(result.debug_flags.get('nontechnical_penalty_total', 0.0)):.6f}",
                "penalties": ";".join(result.penalties),
                "reason_codes": ";".join(result.reason_codes),
                "hireability_reason_codes": ";".join(result.hireability_reason_codes),
                "evidence_realism_reason_codes": ";".join(result.evidence_realism_reason_codes),
                "seniority_alignment_reason_codes": ";".join(result.seniority_alignment_reason_codes),
                "salary_range_inverted": result.debug_flags.get("salary_range_inverted", False),
                "weak_hireability_signal_count": result.debug_flags.get("weak_hireability_signal_count", 0),
                "repeated_career_template_count": result.debug_flags.get("repeated_career_template_count", 0),
                "same_candidate_repeated_template": result.debug_flags.get("same_candidate_repeated_template", False),
                "company_domain_description_mismatch": result.debug_flags.get("company_domain_description_mismatch", False),
                "recent_hands_on_score": f"{float(result.debug_flags.get('recent_hands_on_score', 0.0)):.6f}",
                "recent_leadership_score": f"{float(result.debug_flags.get('recent_leadership_score', 0.0)):.6f}",
                "recent_leadership_heavy": result.debug_flags.get("recent_leadership_heavy", False),
                "tech_lead_aspiration_soft_risk": result.debug_flags.get("tech_lead_aspiration_soft_risk", False),
                "notice_period_days": s.get("notice_period_days", ""),
                "open_to_work_flag": s.get("open_to_work_flag", ""),
                "recruiter_response_rate": s.get("recruiter_response_rate", ""),
                "avg_response_time_hours": s.get("avg_response_time_hours", ""),
                "interview_completion_rate": s.get("interview_completion_rate", ""),
                "verified_email": s.get("verified_email", ""),
                "verified_phone": s.get("verified_phone", ""),
                "github_activity_score": s.get("github_activity_score", ""),
            }
            for key, value in result.components.items():
                row[key] = f"{value:.6f}"
            writer.writerow(row)


def validate_submission(results: list[RerankResult], rejected_ids: set[str]) -> None:
    if len(results) != 100:
        raise ValueError(f"Expected exactly 100 results, found {len(results)}")
    candidate_ids = [result.candidate_id for result in results]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate_id in final top100")
    if any(cid in rejected_ids for cid in candidate_ids):
        raise ValueError("Hard rejected candidate found in final top100")
    if any(not result.reasoning.strip() for result in results):
        raise ValueError("Missing reasoning in final top100")
    for result in results:
        words = word_count(result.reasoning)
        if words < 20 or words > 38:
            raise ValueError(f"Reasoning length outside 20-38 words for {result.candidate_id}: {words}")
    if any(math.isnan(result.final_score) for result in results):
        raise ValueError("NaN score in final top100")
    for result in results:
        penalty_values = [
            result.hireability_penalty,
            result.evidence_realism_penalty,
            result.seniority_drift_penalty,
            float(result.debug_flags.get("nontechnical_penalty_total", 0.0)),
        ]
        if any(math.isnan(value) for value in penalty_values):
            raise ValueError(f"NaN penalty in final top100 for {result.candidate_id}")
    for previous, current in zip(results, results[1:]):
        if previous.final_score < current.final_score:
            raise ValueError("Scores are not sorted descending")
    prefixes: dict[str, int] = {}
    starts: dict[str, int] = {}
    for result in results:
        prefix = result.reasoning.split(":", 1)[0].strip().lower()
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
        start = " ".join(result.reasoning.lower().split()[:3])
        starts[start] = starts.get(start, 0) + 1
    repeated = {prefix: count for prefix, count in prefixes.items() if count > 5}
    if repeated:
        raise ValueError(f"Reasoning prefix repeated too often: {repeated}")
    repeated_starts = {start: count for start, count in starts.items() if count > 5}
    if repeated_starts:
        raise ValueError(f"Reasoning start repeated too often: {repeated_starts}")
    banned_phrases = [
        "Availability adds confidence",
        "Selection is based on career-history evidence",
        "giving the kind of",
        "well aligned with Redrob",
        "strongly tied to Redrob",
        "This keeps the selection tied",
        "stands out",
        "is strong",
        "is relevant",
        "fits well",
        "signal is clear",
        "background matches",
        "stays high",
        "remains in the top 100",
        "The rank reflects",
        "Exceptional fit",
        "High-confidence fit",
        "Solid fit through",
        "Useful but narrower fit through",
        "Strong fit through",
        "primary differentiator",
        "career evidence",
        "keyword-only",
        "not only skill keywords",
        "implementation verbs",
        "layered evidence",
        "less layered evidence",
        "career text uses implementation verbs",
        "This keeps the profile relevant",
        "The strongest signal comes from shipped work",
        "Junior ML Engineer",
        "The work is still useful",
        "The profile has shipped-system signal",
    ]
    for result in results:
        if re.search(r"\bRank\s+\d+\b", result.reasoning):
            raise ValueError(f"Literal rank number found in reasoning for {result.candidate_id}")
        for phrase in banned_phrases:
            if phrase.lower() in result.reasoning.lower():
                raise ValueError(f"Banned repeated/filler phrase in {result.candidate_id}: {phrase}")
    limited_phrases = [
        "useful for ranking-quality evaluation and search relevance work",
        "weaker link to Redrob's ranking upgrade than retrieval-heavy profiles",
        "useful for semantic matching though direct ranking ownership is weaker",
        "search/retrieval proof is thinner than specialist profiles",
        "useful for production AI delivery work",
    ]
    for phrase in limited_phrases:
        count = sum(1 for result in results if phrase.lower() in result.reasoning.lower())
        if count > 5:
            raise ValueError(f"Repeated tradeoff phrase too frequent ({count}): {phrase}")
    hireability_rows = [
        result for result in results
        if any(marker in result.reasoning for marker in [
            "Availability is unusually clean",
            "recruiter response rate",
            "Interview follow-through",
            "Open-to-work status",
            "Short notice",
            "verified contact",
        ])
    ]
    if len(hireability_rows) > 25:
        raise ValueError(f"Hireability mentioned too often: {len(hireability_rows)} rows")
    for result in results[:50]:
        candidate = result.row["candidate"]
        profile_type = candidate_profile_type(candidate, result.components)
        if profile_type == "adjacent_data_infra_candidate":
            continue
        if not has_any(career_text(candidate), ALL_CAREER_REASON_TERMS):
            raise ValueError(f"Top50 candidate lacks career-history evidence in reasoning terms: {result.candidate_id}")


def validate_debug_columns(path: Path) -> None:
    required = {
        "evidence_realism_penalty",
        "evidence_realism_reason_codes",
        "salary_range_inverted",
        "weak_hireability_signal_count",
        "repeated_career_template_count",
        "same_candidate_repeated_template",
        "company_domain_description_mismatch",
        "recent_hands_on_score",
        "recent_leadership_score",
        "recent_leadership_heavy",
        "tech_lead_aspiration_soft_risk",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"Debug CSV missing required columns: {missing}")


def rerank(
    top2000_path: Path,
    rejected_path: Path,
    candidates_path: Path,
    submission_path: Path,
    debug_path: Path,
    top100_jsonl_path: Path,
) -> None:
    rejected_ids = load_rejected_ids(rejected_path)
    rows = load_top2000(top2000_path)
    realism_index = build_evidence_realism_index(rows)
    results = [
        rerank_row(row, realism_index)
        for row in rows
        if str(row.get("candidate_id") or row.get("candidate", {}).get("candidate_id")) not in rejected_ids
    ]
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))
    selected = results[:100]
    for rank, result in enumerate(selected, start=1):
        result.reasoning = build_ranked_reasoning(result, rank)
    validate_submission(selected, rejected_ids)
    write_submission(selected, submission_path)
    write_top100_jsonl(selected, top100_jsonl_path, candidates_path)
    write_debug(results, debug_path)
    validate_debug_columns(debug_path)
    print(f"Loaded top2000 rows: {len(rows)}")
    print(f"Reranked candidates: {len(results)}")
    print(f"Wrote submission: {submission_path}")
    print(f"Wrote top100 JSONL from {candidates_path}: {top100_jsonl_path}")
    print(f"Wrote debug CSV: {debug_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-based rerank from top2000 to final top100 submission")
    parser.add_argument("--input", type=Path, default=DEFAULT_TOP2000_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--debug-out", type=Path, default=DEFAULT_DEBUG_PATH)
    parser.add_argument("--top100-jsonl", type=Path, default=DEFAULT_TOP100_JSONL_PATH)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED_PATH)
    args = parser.parse_args()
    rerank(args.input, args.rejected, args.candidates, args.out, args.debug_out, args.top100_jsonl)


if __name__ == "__main__":
    main()

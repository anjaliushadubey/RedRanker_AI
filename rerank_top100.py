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
EXPECTED_TOP4_ORDER = [
    "CAND_0081846",
    "CAND_0046525",
    "CAND_0071974",
    "CAND_0062247",
]

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
    career_template_penalty: float
    career_template_reason_codes: list[str]
    seniority_drift_penalty: float
    senior_ic_alignment_bonus: float
    seniority_alignment_reason_codes: list[str]
    top5_template_guardrail_penalty: float
    top5_template_guardrail_reason: str
    top5_guardrail_applied: bool
    strong_unique_current_role_evidence: bool
    unique_current_role_signal_count: int
    final_calibration_penalty: float
    final_calibration_reason_codes: list[str]
    jd_alignment_boost: float
    jd_alignment_penalty: float
    jd_alignment_reason_codes: list[str]
    notice_availability_penalty: float
    behavioral_availability_penalty: float
    final_jd_adjusted_score: float
    recruiter_facing_matching_evidence: bool
    production_eval_ownership: bool
    top10_repeated_evidence_guardrail_penalty: float
    top10_repeated_evidence_guardrail_reason: str
    top10_repeated_evidence_guardrail_applied: bool
    top10_not_open_guardrail_penalty: float
    top10_not_open_guardrail_reason: str
    top10_not_open_guardrail_applied: bool
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
    fingerprint_role_counts: Counter[str]
    semantic_candidate_counts: Counter[str]
    semantic_role_counts: Counter[str]
    sentence_counts: Counter[str]
    sentence_companies: dict[str, set[str]]


TEMPLATE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "this",
    "to",
    "with",
}

TEMPLATE_TOOL_PLACEHOLDERS = {
    "faiss": "vector_index",
    "hnsw": "ann_index",
    "pinecone": "vector_db",
    "qdrant": "vector_db",
    "milvus": "vector_db",
    "weaviate": "vector_db",
    "elasticsearch": "keyword_search",
    "opensearch": "keyword_search",
    "solr": "keyword_search",
    "bm25": "keyword_search",
    "sentence-transformers": "embedding_model",
    "sentence": "embedding_model",
    "transformers": "embedding_model",
    "bge": "embedding_model",
    "openai": "llm_api",
    "gpt": "llm_model",
    "xgboost": "ltr_model",
}

TEMPLATE_SIGNAL_TERMS = {
    "ranking",
    "ranker",
    "retrieval",
    "search",
    "semantic",
    "vector",
    "hybrid",
    "recommendation",
    "recommender",
    "matching",
    "personalization",
    "learning",
    "rank",
    "relevance",
    "labeling",
    "judgments",
    "click",
    "feature",
    "pipeline",
    "embedding",
    "reranker",
    "chatbot",
    "rag",
    "support",
    "content",
    "candidate",
    "job",
    "latency",
    "monitoring",
    "index",
    "refresh",
    "rollback",
    "versioning",
    "evaluation",
    "ndcg",
    "mrr",
    "map",
    "qps",
    "p95",
    "engagement",
}


def normalize_evidence_text(value: object) -> str:
    text = normalize(value)
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;,")


def template_tokens(description: object, company: object = "") -> list[str]:
    text = normalize(description)
    company_tokens = {
        token for token in re.findall(r"[a-z0-9]+", normalize(company))
        if len(token) > 2
    }
    text = re.sub(r"\b\d+(?:\.\d+)?\s?(?:k|m|million|lakh|crore|%|ms|qps)\b", " ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " ", text)
    text = re.sub(r"[^a-z0-9+#@]+", " ", text)
    tokens: list[str] = []
    for token in text.split():
        if token in company_tokens or token in TEMPLATE_STOPWORDS or len(token) <= 1:
            continue
        tokens.append(TEMPLATE_TOOL_PLACEHOLDERS.get(token, token))
    return tokens


def exact_template_fingerprint(description: object, company: object = "") -> str:
    return " ".join(template_tokens(description, company)[:45])


def semantic_template_signature(description: object, company: object = "") -> str:
    tokens = template_tokens(description, company)
    signal_tokens = [
        token for token in tokens
        if token in TEMPLATE_SIGNAL_TERMS or token in set(TEMPLATE_TOOL_PLACEHOLDERS.values())
    ]
    if len(signal_tokens) < 8:
        signal_tokens = tokens[:35]
    return " ".join(signal_tokens[:50])


def career_description_fingerprint(description: object, company: object = "") -> str:
    return exact_template_fingerprint(description, company)


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
            "title": str(job.get("title") or ""),
            "start_date": job.get("start_date"),
            "end_date": job.get("end_date"),
            "is_current": bool(job.get("is_current")),
            "duration_months": int(float(job.get("duration_months") or 0)),
            "fingerprint": career_description_fingerprint(description, job.get("company")),
            "exact_template_fingerprint": exact_template_fingerprint(description, job.get("company")),
            "semantic_template_signature": semantic_template_signature(description, job.get("company")),
        })
    return records


def build_evidence_realism_index(rows: list[dict]) -> EvidenceRealismIndex:
    description_counts: Counter[str] = Counter()
    fingerprint_candidate_counts: Counter[str] = Counter()
    fingerprint_role_counts: Counter[str] = Counter()
    semantic_candidate_counts: Counter[str] = Counter()
    semantic_role_counts: Counter[str] = Counter()
    sentence_counts: Counter[str] = Counter()
    sentence_companies: dict[str, set[str]] = defaultdict(set)
    fingerprint_candidates: dict[str, set[str]] = defaultdict(set)
    semantic_candidates: dict[str, set[str]] = defaultdict(set)

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
                fingerprint_role_counts[fingerprint] += 1
            semantic = str(record.get("semantic_template_signature") or "")
            if len(semantic) >= 60 and has_any(semantic, ALL_CAREER_REASON_TERMS):
                semantic_candidates[semantic].add(candidate_id)
                semantic_role_counts[semantic] += 1
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
    for semantic, candidate_ids in semantic_candidates.items():
        semantic_candidate_counts[semantic] = len(candidate_ids)

    return EvidenceRealismIndex(
        description_counts=description_counts,
        fingerprint_candidate_counts=fingerprint_candidate_counts,
        fingerprint_role_counts=fingerprint_role_counts,
        semantic_candidate_counts=semantic_candidate_counts,
        semantic_role_counts=semantic_role_counts,
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


POSTGRAD_DEGREE_TERMS = [
    "m.e",
    "me",
    "m.tech",
    "mtech",
    "m.s",
    "ms",
    "m.sc",
    "msc",
    "master",
    "masters",
    "mba",
    "pgdm",
]

DOCTORAL_DEGREE_TERMS = [
    "phd",
    "ph.d",
    "doctor",
    "doctorate",
]


def education_timeline_inconsistency(candidate: dict) -> tuple[bool, list[str]]:
    """Return education anomalies that matter only when other realism signals exist."""
    reasons: list[str] = []
    education_records = []
    for education in candidate.get("education", []) or []:
        start = education.get("start_year")
        end = education.get("end_year")
        degree = normalize(education.get("degree"))
        institution = normalize(education.get("institution"))
        try:
            start_year = int(start) if start is not None else None
            end_year = int(end) if end is not None else None
        except (TypeError, ValueError):
            continue
        education_records.append(
            {
                "start_year": start_year,
                "end_year": end_year,
                "degree": degree,
                "institution": institution,
            }
        )
        if start_year and end_year and end_year < start_year:
            reasons.append("education_end_before_start")
            continue
        if not start_year or not end_year:
            continue
        duration_years = end_year - start_year + 1
        is_doctoral = has_any(degree, DOCTORAL_DEGREE_TERMS)
        is_taught_postgrad = has_any(degree, POSTGRAD_DEGREE_TERMS) and not is_doctoral
        if is_taught_postgrad and duration_years > 3:
            reasons.append("postgraduate_degree_unusually_long")

    dated_education = sorted(
        [record for record in education_records if record["start_year"] and record["end_year"]],
        key=lambda record: (int(record["start_year"]), int(record["end_year"])),
    )
    career_starts = [
        parse_date(job.get("start_date"))
        for job in candidate.get("career_history", []) or []
        if parse_date(job.get("start_date"))
    ]
    first_career_year = min((value.year for value in career_starts), default=None)
    for previous, current in zip(dated_education, dated_education[1:]):
        previous_end = int(previous["end_year"])
        current_start = int(current["start_year"])
        gap_years = current_start - previous_end
        has_recorded_work_before_next_degree = first_career_year is not None and first_career_year <= current_start
        if gap_years > 4 and not has_recorded_work_before_next_degree:
            reasons.append("large_unexplained_education_gap")
            break

    return bool(reasons), unique_ordered(reasons, 6)


def compound_realism_top100_risk(result: RerankResult) -> tuple[bool, list[str]]:
    if result.strong_unique_current_role_evidence or result.unique_current_role_signal_count >= 3:
        return False, []
    candidate = result.row["candidate"]
    education_bad, education_reasons = education_timeline_inconsistency(candidate)
    if not education_bad:
        return False, []
    if not bool(result.debug_flags.get("company_domain_description_mismatch", False)):
        return False, []
    if not bool(result.debug_flags.get("current_role_template_heavy", False)):
        return False, []
    max_template_count = int(result.debug_flags.get("max_template_candidate_count", 0) or 0)
    repeated_ratio = float(result.debug_flags.get("repeated_template_ratio", 0.0) or 0.0)
    if max_template_count < 4 and repeated_ratio < 0.50:
        return False, []
    return True, ["compound_domain_education_template_inconsistency"] + education_reasons


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


UNIQUE_ARCHITECTURE_TERMS = [
    "bm25 + dense retrieval",
    "bm25 and dense retrieval",
    "dense retrieval",
    "llm-based re-ranker",
    "llm reranker",
    "learning-to-rank model",
    "faiss hnsw",
    "bge-large",
    "bge embeddings",
    "pinecone retrieval",
    "xgboost ltr",
    "hybrid retrieval architecture",
    "latency fallback",
]

UNIQUE_EVALUATION_TERMS = [
    "ndcg",
    "mrr",
    "recall@k",
    "precision@k",
    "a/b engagement metrics",
    "a/b testing",
    "human relevance judgments",
    "relevance labeling",
    "online/offline evaluation",
    "offline evaluation",
    "online evaluation",
]

UNIQUE_OPERATIONAL_TERMS = [
    "index versioning",
    "embedding versioning",
    "embedding drift monitoring",
    "rollback paths",
    "incremental index refresh",
    "latency budget",
    "p95",
    "monitoring",
    "dashboards",
    "recruiter-feedback loop",
    "feedback loop",
]

UNIQUE_DOMAIN_TERMS = [
    "candidate-jd matching",
    "candidate jd matching",
    "candidate-role matching",
    "candidate matching",
    "job matching",
    "recruiter-facing",
    "recruiter engagement",
    "time-to-shortlist",
]

UNIQUE_IMPACT_TERMS = [
    "improved ndcg",
    "reduced latency",
    "improved recruiter engagement",
    "recruiter engagement metrics",
    "reduced the average time-to-shortlist",
    "time-to-shortlist",
    "revenue-per-search",
    "search-relevance improvement",
]


def role_overlaps_recent(record: dict[str, object], months: int = 18) -> bool:
    if bool(record.get("is_current")):
        return True
    threshold = (REFERENCE_DATE.year * 12 + REFERENCE_DATE.month) - months
    end = parse_date(record.get("end_date")) or REFERENCE_DATE
    return (end.year * 12 + end.month) >= threshold


def unique_evidence_score(
    candidate: dict,
    records: list[dict[str, object]] | None = None,
    global_template_stats: EvidenceRealismIndex | None = None,
) -> tuple[float, list[str], bool]:
    if records is None:
        records = career_description_records(candidate)

    usable_records: list[dict[str, object]] = []
    for record in records:
        if global_template_stats is not None:
            exact = str(record.get("exact_template_fingerprint") or record.get("fingerprint") or "")
            semantic = str(record.get("semantic_template_signature") or "")
            candidate_count = max(
                global_template_stats.fingerprint_candidate_counts.get(exact, 0),
                global_template_stats.semantic_candidate_counts.get(semantic, 0),
            )
            if candidate_count >= 2 and has_any(exact + " " + semantic, ALL_CAREER_REASON_TERMS):
                continue
        usable_records.append(record)

    if not usable_records:
        return 0.0, [], False

    all_text = normalize(
        " ".join(str(record.get("description") or "") for record in usable_records)
    )
    recent_text = " ".join(
        str(record.get("description") or "")
        for record in usable_records
        if role_overlaps_recent(record)
    ).lower()
    text = recent_text or all_text
    reasons: list[str] = []

    scale_patterns = [
        r"\b(?:30|35|50)m\+?\b",
        r"\b\d+(?:\.\d+)?m\+?\s+(?:queries|documents|items|users)\b",
        r"\b8k\s*qps\b",
        r"\bsub-?200ms\b",
        r"\bp95\b",
    ]
    if any(re.search(pattern, text) for pattern in scale_patterns):
        reasons.append("unique_scale")
    if has_any(text, UNIQUE_ARCHITECTURE_TERMS):
        reasons.append("specific_architecture")
    if has_any(text, UNIQUE_EVALUATION_TERMS):
        reasons.append("evaluation_depth")
    if has_any(text, UNIQUE_OPERATIONAL_TERMS):
        reasons.append("operational_detail")
    if has_any(text, UNIQUE_DOMAIN_TERMS):
        reasons.append("candidate_or_recruiter_domain")
    if has_any(text, UNIQUE_IMPACT_TERMS) or re.search(r"\bimprov(?:ed|ement)\b.*\b\d+(?:\.\d+)?%", text):
        reasons.append("concrete_impact")

    if len(reasons) < 2 and recent_text:
        fallback_reasons = []
        if any(re.search(pattern, all_text) for pattern in scale_patterns):
            fallback_reasons.append("unique_scale")
        if has_any(all_text, UNIQUE_ARCHITECTURE_TERMS):
            fallback_reasons.append("specific_architecture")
        if has_any(all_text, UNIQUE_EVALUATION_TERMS):
            fallback_reasons.append("evaluation_depth")
        if has_any(all_text, UNIQUE_OPERATIONAL_TERMS):
            fallback_reasons.append("operational_detail")
        if has_any(all_text, UNIQUE_DOMAIN_TERMS):
            fallback_reasons.append("candidate_or_recruiter_domain")
        if has_any(all_text, UNIQUE_IMPACT_TERMS) or re.search(r"\bimprov(?:ed|ement)\b.*\b\d+(?:\.\d+)?%", all_text):
            fallback_reasons.append("concrete_impact")
        reasons = unique_ordered(reasons + fallback_reasons)

    recent_unique = bool(recent_text) and len(reasons) >= 2 and any(
        reason in reasons
        for reason in ["unique_scale", "specific_architecture", "evaluation_depth", "operational_detail"]
    )
    return min(1.0, len(reasons) / 4.0), unique_ordered(reasons, 6), recent_unique


CURRENT_SCALE_PATTERNS = [
    r"\b(?:30|35|50)m\+?\b",
    r"\b\d+(?:\.\d+)?m\+?\s+(?:queries|documents|items|users|candidates|candidate\s+corpus)\b",
    r"\b\d+(?:\.\d+)?k\+?\s+(?:documents|preference\s+pairs|pairs|queries|items|labels)\b",
    r"\b8k\s*qps\b",
    r"\bsub-?200ms\b",
    r"\bp95\b",
]

CURRENT_ARCHITECTURE_TERMS = [
    "bm25 + dense retrieval",
    "bm25+dense",
    "bm25",
    "elasticsearch",
    "opensearch",
    "dense retrieval",
    "hybrid retrieval",
    "hybrid search",
    "faiss hnsw",
    "bge-large",
    "bge-base",
    "bge embeddings",
    "pinecone retrieval",
    "pinecone",
    "qdrant",
    "milvus",
    "weaviate",
    "llm-based re-ranker",
    "llm based re-ranker",
    "llm reranker",
    "llm re-ranker",
    "xgboost re-scoring",
    "xgboost reranking",
    "xgboost",
    "bentoml",
    "llama-2",
    "mistral-7b",
    "lora",
    "qlora",
    "faiss",
    "sentence-transformers",
    "query expansion",
    "nearest-neighbor retrieval",
]

CURRENT_EVALUATION_TERMS = [
    "ndcg",
    "mrr",
    "recall@k",
    "precision@k",
    "a/b testing",
    "a/b test",
    "simulated a/b",
    "online a/b",
    "offline metrics",
    "offline-online correlation",
    "human relevance judgments",
    "explicit human judgments",
    "human judgments",
    "recruiter feedback loop",
    "recruiter-feedback",
]

CURRENT_OPS_TERMS = [
    "index versioning",
    "embedding versioning",
    "rollback paths",
    "rollback",
    "embedding drift monitoring",
    "embedding drift",
    "latency fallback",
    "incremental index refresh",
    "incremental refresh",
    "dashboards",
    "kubernetes",
    "quantizing",
    "batching",
]

CURRENT_IMPACT_TERMS = [
    "improved ndcg",
    "reduced p95",
    "reduced latency",
    "latency by",
    "recruiter engagement",
    "time-to-shortlist",
    "search-relevance improvement",
    "improved new-user retention",
    "cost per inference",
    "dropped from",
]

CURRENT_DOMAIN_TERMS = [
    "candidate-jd matching",
    "candidate jd matching",
    "recruiter-facing search",
    "recruiter facing search",
    "candidate corpus",
    "candidate sourcing",
    "retrieval/ranking pipeline",
    "ranking pipeline",
    "search relevance",
    "semantic search",
    "e-commerce search",
    "ecommerce search",
    "discovery ranking",
    "recommendation ranking",
    "recommendations-heavy consumer product",
    "behavioral-signal integration",
    "recruiter labels",
]


def date_month_value(value: object | None) -> int:
    parsed = parse_date(value)
    if parsed is None:
        return 0
    return parsed.year * 12 + parsed.month


def current_or_recent_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    if not records:
        return []
    current = [record for record in records if bool(record.get("is_current"))]
    if current:
        return current
    return [max(
        records,
        key=lambda record: max(
            date_month_value(record.get("end_date")),
            date_month_value(record.get("start_date")),
        ),
    )]


def record_template_candidate_count(
    record: dict[str, object],
    global_template_stats: EvidenceRealismIndex | None,
) -> int:
    if global_template_stats is None:
        return 0
    exact = str(record.get("exact_template_fingerprint") or record.get("fingerprint") or "")
    semantic = str(record.get("semantic_template_signature") or "")
    return max(
        global_template_stats.fingerprint_candidate_counts.get(exact, 0),
        global_template_stats.semantic_candidate_counts.get(semantic, 0),
    )


def record_is_repeated_template(
    record: dict[str, object],
    global_template_stats: EvidenceRealismIndex | None,
) -> bool:
    if global_template_stats is None:
        return False
    exact = str(record.get("exact_template_fingerprint") or record.get("fingerprint") or "")
    semantic = str(record.get("semantic_template_signature") or "")
    candidate_count = record_template_candidate_count(record, global_template_stats)
    return candidate_count >= 2 and has_any(exact + " " + semantic, ALL_CAREER_REASON_TERMS)


def unique_current_role_evidence(
    candidate: dict,
    records: list[dict[str, object]] | None = None,
) -> tuple[bool, int, list[str]]:
    if records is None:
        records = career_description_records(candidate)
    recent_records = current_or_recent_records(records)
    text = normalize_evidence_text(
        " ".join(str(record.get("description") or "") for record in recent_records)
    )
    reasons: list[str] = []
    if any(re.search(pattern, text) for pattern in CURRENT_SCALE_PATTERNS):
        reasons.append("specific_scale")
    if has_any(text, CURRENT_ARCHITECTURE_TERMS):
        reasons.append("specific_architecture")
    if has_any(text, CURRENT_EVALUATION_TERMS):
        reasons.append("evaluation_depth")
    if has_any(text, CURRENT_OPS_TERMS):
        reasons.append("production_ops")
    if has_any(text, CURRENT_IMPACT_TERMS) or re.search(r"\bimprov(?:ed|ement)\b.*\b\d+(?:\.\d+)?%", text):
        reasons.append("concrete_impact")
    if has_any(text, CURRENT_DOMAIN_TERMS):
        reasons.append("domain_match")
    reasons = unique_ordered(reasons, 8)
    return len(reasons) >= 3, len(reasons), reasons


def guardrail_recent_role_template_heavy(
    candidate: dict,
    global_template_stats: EvidenceRealismIndex | None,
    template_analysis: dict[str, object],
    records: list[dict[str, object]] | None = None,
) -> bool:
    if records is None:
        records = career_description_records(candidate)
    recent_records = current_or_recent_records(records)
    recent_repeated = any(record_is_repeated_template(record, global_template_stats) for record in recent_records)
    recent_high_frequency = any(
        record_template_candidate_count(record, global_template_stats) >= 4
        for record in recent_records
    )
    return bool(
        recent_high_frequency
        or template_analysis.get("same_candidate_repeated_template")
        or (
            float(template_analysis.get("repeated_template_ratio") or 0.0) >= 0.50
            and recent_repeated
        )
    )


def repeated_template_stats(
    candidate: dict,
    index: EvidenceRealismIndex | None,
    records: list[dict[str, object]] | None = None,
) -> tuple[float, int, bool, list[str]]:
    analysis = career_template_analysis(candidate, index, records=records)
    return (
        float(analysis["template_penalty_after_unique_offset"]),
        int(analysis["max_template_candidate_count"]),
        bool(analysis["same_candidate_repeated_template"]),
        list(analysis["career_template_reason_codes"]),
    )


def career_template_analysis(
    candidate: dict,
    global_template_stats: EvidenceRealismIndex | None,
    records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if global_template_stats is None:
        return {
            "career_template_penalty": 0.0,
            "career_template_reason_codes": [],
            "repeated_role_count": 0,
            "max_template_candidate_count": 0,
            "repeated_template_ratio": 0.0,
            "current_role_template_heavy": False,
            "unique_evidence_score": 0.0,
            "unique_evidence_reason_codes": [],
            "template_penalty_before_unique_offset": 0.0,
            "template_penalty_after_unique_offset": 0.0,
            "same_candidate_repeated_template": False,
        }

    if records is None:
        records = career_description_records(candidate)

    role_descriptions = [
        record for record in records
        if str(record.get("description") or "").strip()
    ]
    total_roles = len(role_descriptions)
    reason_codes: list[str] = []
    repeated_records = 0
    repeated_fingerprints: set[str] = set()
    repeated_signatures: set[str] = set()
    max_candidate_count = 0
    max_role_count = 0
    current_role_template_heavy = False

    exact_counts = Counter(
        str(record.get("exact_template_fingerprint") or record.get("fingerprint") or "")
        for record in role_descriptions
    )
    semantic_counts = Counter(
        str(record.get("semantic_template_signature") or "")
        for record in role_descriptions
    )

    same_candidate_repeated = any(
        count > 1 and len(value) >= 60 and has_any(value, ALL_CAREER_REASON_TERMS)
        for value, count in list(exact_counts.items()) + list(semantic_counts.items())
    )

    raw_penalty = 0.0
    if same_candidate_repeated:
        raw_penalty += 0.025
        reason_codes.append("same_candidate_repeated_role_template")

    for record in role_descriptions:
        exact = str(record.get("exact_template_fingerprint") or record.get("fingerprint") or "")
        semantic = str(record.get("semantic_template_signature") or "")
        exact_candidate_count = global_template_stats.fingerprint_candidate_counts.get(exact, 0)
        exact_role_count = global_template_stats.fingerprint_role_counts.get(exact, 0)
        semantic_candidate_count = global_template_stats.semantic_candidate_counts.get(semantic, 0)
        semantic_role_count = global_template_stats.semantic_role_counts.get(semantic, 0)
        candidate_count = max(exact_candidate_count, semantic_candidate_count)
        role_count = max(exact_role_count, semantic_role_count)
        max_candidate_count = max(max_candidate_count, candidate_count)
        max_role_count = max(max_role_count, role_count)

        is_repeated = (
            candidate_count >= 2
            and (
                (len(exact) >= 60 and has_any(exact, ALL_CAREER_REASON_TERMS))
                or (len(semantic) >= 50 and has_any(semantic, ALL_CAREER_REASON_TERMS))
            )
        )
        if not is_repeated:
            continue
        repeated_records += 1
        if exact:
            repeated_fingerprints.add(exact)
        if semantic:
            repeated_signatures.add(semantic)
        if role_overlaps_recent(record):
            current_role_template_heavy = True

    for value in repeated_fingerprints.union(repeated_signatures):
        candidate_count = max(
            global_template_stats.fingerprint_candidate_counts.get(value, 0),
            global_template_stats.semantic_candidate_counts.get(value, 0),
        )
        if 2 <= candidate_count <= 3:
            raw_penalty += 0.010
        elif 4 <= candidate_count <= 8:
            raw_penalty += 0.025
        elif candidate_count >= 9:
            raw_penalty += 0.040

    if repeated_records:
        reason_codes.append("repeated_career_template")
    if max_candidate_count >= 9:
        reason_codes.append("high_frequency_career_template")
    if len(repeated_fingerprints.union(repeated_signatures)) >= 2 or repeated_records >= 2:
        raw_penalty += min(0.025, 0.010 + 0.005 * (repeated_records - 1))
        reason_codes.append("multiple_repeated_templates")
    if current_role_template_heavy:
        raw_penalty *= 1.25
        reason_codes.append("template_heavy_recent_role")

    unique_score, unique_reasons, recent_unique = unique_evidence_score(
        candidate,
        records=records,
        global_template_stats=global_template_stats,
    )
    offset_multiplier = 1.0
    if unique_score >= 0.50:
        offset_multiplier = 0.50 if recent_unique else 0.70
        reason_codes.append("unique_evidence_offsets_template_risk")

    cap = 0.085 if current_role_template_heavy and unique_score < 0.50 else 0.065
    before_offset = min(raw_penalty, cap)
    after_offset = min(before_offset * offset_multiplier, cap)

    return {
        "career_template_penalty": after_offset,
        "career_template_reason_codes": unique_ordered(reason_codes, 10),
        "repeated_role_count": repeated_records,
        "max_template_candidate_count": max_candidate_count,
        "max_template_role_count": max_role_count,
        "repeated_template_ratio": (repeated_records / total_roles) if total_roles else 0.0,
        "current_role_template_heavy": current_role_template_heavy,
        "unique_evidence_score": unique_score,
        "unique_evidence_reason_codes": unique_reasons,
        "template_penalty_before_unique_offset": before_offset,
        "template_penalty_after_unique_offset": after_offset,
        "same_candidate_repeated_template": same_candidate_repeated,
    }


def career_template_penalty(
    candidate: dict,
    global_template_stats: EvidenceRealismIndex | None,
) -> tuple[float, list[str]]:
    analysis = career_template_analysis(candidate, global_template_stats)
    return (
        float(analysis["career_template_penalty"]),
        list(analysis["career_template_reason_codes"]),
    )


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

    if has_duration_contradiction(candidate):
        penalty += 0.04
        reason_codes.append("project_duration_exceeds_role_duration")

    if has_domain_mismatch(candidate):
        penalty += 0.018
        reason_codes.append("company_domain_description_mismatch")

    hard_realism_issue = any(
        code in reason_codes
        for code in [
            "project_duration_exceeds_role_duration",
            "salary_range_inverted",
            "weak_hireability_cluster",
        ]
    )
    if strong_technical and reason_codes and not hard_realism_issue:
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


def shorten_reasoning(reasoning: str, max_words: int = 52) -> str:
    if word_count(reasoning) <= max_words:
        return reasoning
    sentences = re.split(r"(?<=[.!?])\s+", reasoning)
    trimmed = " ".join(sentences[:2]).strip()
    if 20 <= word_count(trimmed) <= max_words:
        return trimmed
    words = reasoning.split()
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


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
            "Senior execution evidence drives the case",
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
        return f"{opening}: career history shows hands-on ML/retrieval delivery, keeping the profile closer to a coding IC than a pure manager."

    if differentiator == "adjacent_but_strong_ml_infra":
        infra = format_terms(match_terms(career_text(candidate), DATA_INFRA_REASON_TERMS + PRODUCTION_SYSTEM_TERMS), 3)
        return f"{opening}: strong {infra or 'ML/data infrastructure'} and feature-pipeline exposure, with a weaker link to direct search-ranking or retrieval ownership."

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
            f"The main practical tradeoff is {notice}-day notice.",
            f"Technical fit is convincing; {notice}-day notice is the main practical tradeoff.",
            f"{notice}-day notice reduces immediacy versus faster-moving profiles.",
        ])
    if {"low_recruiter_response", "very_slow_response_low_response_rate", "unverified_contacts_low_response"} & codes:
        return phrase_variant(candidate, "hire-note-reach", [
            "Strong search evidence, but reachability signals are weaker than stronger shortlist profiles.",
            "Technical fit is solid; recruiter response and contact signals make outreach riskier.",
            "Good technical proof, but weak response/contact signals reduce immediate hiring confidence.",
        ])
    if "weak_interview_completion" in codes:
        return phrase_variant(candidate, "hire-note-interview", [
            "Technical fit is useful, but interview follow-through is a practical concern.",
            "Search/ML evidence remains relevant; process reliability is the practical concern.",
            "Good systems evidence, but low interview completion keeps the profile lower.",
        ])
    return phrase_variant(candidate, "hire-note-general", [
        "Technical fit is useful, but hireability signals are less reassuring.",
        "Good technical evidence, with practical availability risk keeping the profile lower.",
        "Strong enough technically, though Redrob engagement signals are less clean.",
    ])


def realism_tradeoff_note(result: RerankResult) -> str:
    if (
        result.evidence_realism_penalty < 0.04
        and result.career_template_penalty < 0.04
        and result.top5_template_guardrail_penalty < 0.03
        and result.seniority_drift_penalty < 0.04
    ):
        return ""
    candidate = result.row["candidate"]
    codes = set(
        result.evidence_realism_reason_codes
        + result.career_template_reason_codes
        + result.seniority_alignment_reason_codes
    )
    if result.top5_guardrail_applied:
        return phrase_variant(candidate, "realism-top5-guardrail", [
            "More specific current production evidence would improve confidence.",
            "Comparable profiles show more distinctive hands-on retrieval depth.",
            "The recent role gives less specific proof than cleaner retrieval profiles.",
        ])
    if "recent_leadership_heavy" in codes or "recent_leadership_ownership_with_weak_hands_on" in codes:
        return phrase_variant(candidate, "realism-leadership", [
            "Recent role wording leans leadership-heavy.",
            "Recent ownership language is less hands-on.",
            "Recent role evidence leans less clearly hands-on.",
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
            "company_domain_description_mismatch",
        ]
    ):
        return phrase_variant(candidate, "realism-profile-consistency", [
            "The domain-mismatched older evidence is discounted.",
            "Older role wording is less domain-consistent.",
            "Older evidence is less convincing than the current role.",
        ])
    if any(
        code in codes
        for code in [
            "repeated_career_template",
            "same_candidate_repeated_role_template",
            "high_frequency_career_template",
            "multiple_repeated_templates",
            "template_heavy_recent_role",
        ]
    ):
        return phrase_variant(candidate, "realism-template-detail", [
            "More specific production detail would improve confidence.",
            "Comparable candidates show more distinctive implementation detail.",
            "Strong enough technically, though implementation detail is less unique.",
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


def should_use_template_cautious_reasoning(result: RerankResult) -> bool:
    if not bool(result.debug_flags.get("current_role_template_heavy", False)):
        return False
    if result.career_template_penalty >= 0.08:
        return True
    if float(result.debug_flags.get("unique_evidence_score", 0.0) or 0.0) < 0.25:
        return True
    return result.top5_template_guardrail_penalty >= 0.03


def template_cautious_evidence_sentence(result: RerankResult, diffs: list[str]) -> str:
    candidate = result.row["candidate"]
    tools = concise_tools(candidate, 2)
    evals = concise_eval(candidate, 2)
    impact = concise_impact(candidate)
    has_domain_mismatch = bool(result.debug_flags.get("company_domain_description_mismatch", False))

    if has_domain_mismatch:
        if "bm25_to_semantic_migration" in diffs or "hands_on_retrieval_builder" in diffs:
            return phrase_variant(candidate, "template-domain-retrieval", [
                f"Recent work shows BM25/FAISS retrieval{f' with {evals}' if evals else ''}; older domain-mismatched search wording is discounted.",
                f"BM25/FAISS work in the current role keeps the profile viable{f' at {impact} scale' if impact else ''}; the older mismatched paragraph lowers confidence.",
            f"The current role has FAISS/BM25 search-quality evidence{f' checked with {evals}' if evals else ''}, but repeated older wording prevents treating it as uniquely elite.",
            f"Recent search-quality work carries the profile, especially BM25/FAISS evidence{f' with {evals}' if evals else ''}; the older domain mismatch is not over-credited.",
            "Current retrieval evidence earns the shortlist spot, while the older e-commerce-style paragraph is treated as a realism caveat.",
            "FAISS/BM25 evidence from recent work matters most here; the domain-mismatched older role text lowers confidence versus cleaner profiles.",
        ])
        return phrase_variant(candidate, "template-domain-general", [
            "Current ML/search evidence keeps the profile viable, but older domain-mismatched role wording is discounted.",
            "Relevant technical evidence remains, though one older role reads less natural for its domain.",
            "Current production evidence keeps the profile viable; older mismatched wording is a realism caveat.",
            "Recent technical work matters more than the older mismatched paragraph, which is treated cautiously.",
            "The profile remains usable on current evidence, while older domain-fit wording lowers confidence.",
            "Production evidence is present, but the older role/domain mismatch keeps it out of the cleanest tier.",
        ])

    if "learning_to_rank_ownership" in diffs:
        return phrase_variant(candidate, "template-ltr-cautious", [
            f"LTR and relevance-evaluation wording maps to the JD{f' with {evals}' if evals else ''}, but current implementation detail is not as distinctive as the strongest profiles.",
            f"Ranking-layer evidence has value{f' with {evals}' if evals else ''}; cleaner search-system proof would make the case stronger.",
            "The profile has plausible LTR/search relevance signals, but the implementation story is less distinctive than elite retrieval builders.",
            f"Search-ranking language points in the right direction{f' with {evals}' if evals else ''}, though the career detail is not unusually specific.",
            "Learning-to-rank claims are directionally useful, but cleaner implementation detail would justify a stronger position.",
            "Relevance-scoring work supports inclusion, while delivery detail is not distinctive enough for the cleanest tier.",
            "Ranking-quality wording earns attention, but the implementation evidence is less individualized.",
            "LTR-style evidence is plausible enough for the shortlist, not as strong as cleaner current-role search builders.",
            "The ranking story has JD value, but it lacks unusually specific implementation detail.",
            "Evaluation-backed ranking language helps, though confidence stays moderate without more specific delivery proof.",
            "Hand-tuned-to-LTR wording fits the problem, but the evidence needs more specific delivery detail.",
            "Search relevance evidence is present, though stronger retrieval profiles show more individualized proof.",
        ])

    if "bm25_to_semantic_migration" in diffs or "hands_on_retrieval_builder" in diffs:
        return phrase_variant(candidate, "template-retrieval-cautious", [
            f"BM25/vector retrieval evidence maps to the role{f' with {tools}' if tools else ''}, but stronger profiles show more unique implementation detail.",
            "Search/retrieval terms match the JD, yet the evidence is less specific than cleaner current-role retrieval profiles.",
            f"Useful retrieval signal appears{f' with {tools}' if tools else ''}, though the implementation detail is less individualized.",
            f"Vector-search wording has value{f' with {tools}' if tools else ''}, but the delivery proof is not clean enough for the strongest tier.",
            "Retrieval claims support the shortlist, while cleaner current implementation proof would make the profile stronger.",
            "Semantic-search evidence is plausible, but more specific search-system history would raise confidence.",
            "Search-stack terms line up with the JD, yet the profile needs more individualized delivery detail.",
            "BM25-to-vector language helps, but the delivery proof is not distinctive enough to treat as elite.",
        ])

    return phrase_variant(candidate, "template-general-cautious", [
        "Technically plausible profile, but the delivery proof is treated cautiously rather than as unique elite evidence.",
        "Useful ML/search adjacency remains, though cleaner implementation detail would make the profile stronger.",
        "Relevant systems vocabulary is present, but delivery proof is less distinctive than stronger technical fits.",
        "Applied ML language supports inclusion, while stronger recent-role proof would lift confidence.",
        "The profile has enough systems signal to keep, but the claimed experience needs more specific delivery proof.",
        "ML/search vocabulary appears in the right places, though the evidence is not highly distinctive.",
        "The role points toward applied AI work, but cleaner implementation evidence would make it more compelling.",
        "Systems evidence is usable but not uniquely strong compared with cleaner retrieval profiles.",
        "Technical fit remains plausible, making this a cautious shortlist pick rather than a top evidence profile.",
        "The profile belongs in consideration, but more specific delivery proof would strengthen confidence.",
    ])


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
        if support:
            support_text = join_phrases(support, 3)
            connector = "; evidence includes " if " with " in sentence.lower() else " with "
            result = f"{sentence}{connector}{support_text}."
        else:
            result = f"{sentence}."
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


def lower_first(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if len(text) > 1 and text[:2].isupper():
        return text
    return text[0].lower() + text[1:]


def contextualized_lead(candidate: dict, lead: str, rank: int) -> str:
    company = str(profile(candidate).get("current_company") or "").strip()
    if rank > 30 or not company:
        return lead
    if company.lower() in lead.lower():
        return lead
    if word_count(lead) > 34:
        return lead
    lowered = lower_first(lead)
    if lowered.startswith("recent work shows "):
        rest = lowered.removeprefix("recent work shows ").strip()
        return phrase_variant(candidate, "company-recent-work-context", [
            f"{company} recent role shows {rest}",
            f"Recent {company} work shows {rest}",
            f"In the {company} role, {rest}",
            lead,
        ])
    variants = [
        f"{company} work shows {lowered}",
        f"In the {company} role, {lowered}",
        f"{lead} in the {company} context",
        f"{company} adds useful context: {lowered}",
        lead,
        f"Recent {company} work: {lowered}",
    ]
    if rank <= 4:
        variants = [
            f"{company} work shows {lowered}",
            f"In the {company} role, {lowered}",
            f"{lead} in the {company} context",
            f"Recent {company} work: {lowered}",
        ]
    return phrase_variant(candidate, "company-lead-context", variants)


def advantage_summary(result: RerankResult, diffs: list[str]) -> str:
    candidate = result.row["candidate"]
    career = career_text(candidate)
    tools = concise_tools(candidate, 2)
    evals = concise_eval(candidate, 2)
    impact = concise_impact(candidate)
    has_recruiter_match = bool(result.recruiter_facing_matching_evidence)

    if has_recruiter_match and ("bm25_to_semantic_migration" in diffs or "hands_on_retrieval_builder" in diffs):
        return phrase_variant(candidate, "adv-recruiter-retrieval", [
            "recruiter-facing retrieval plus matching relevance and evaluation",
            "candidate-matching context with retrieval and measurement work",
            "matching-product context plus search/retrieval execution",
            "retrieval implementation tied to recruiter/candidate matching",
        ])
    if "bm25_to_semantic_migration" in diffs:
        bits = ["BM25-to-semantic retrieval"]
        non_bm25_tools = format_terms(
            [
                term for term in match_terms(career, TOOL_REASON_TERMS)
                if term.lower() not in {"bm25", "elasticsearch", "opensearch"}
            ],
            2,
        )
        if non_bm25_tools:
            bits.append(f"backed by {non_bm25_tools}")
        if evals:
            bits.append(f"validation via {evals}")
        return join_phrases(bits, 3)
    if "learning_to_rank_ownership" in diffs:
        bits = ["learning-to-rank/relevance scoring"]
        if evals:
            bits.append(f"evaluation via {evals}")
        if impact:
            bits.append(f"{impact} scale")
        return join_phrases(bits, 4)
    if "hands_on_retrieval_builder" in diffs or "large_scale_search_system" in diffs:
        bits = ["retrieval implementation"]
        if tools:
            bits.append(f"{tools} tooling")
        if evals:
            bits.append(f"validation via {evals}")
        if impact:
            bits.append(f"{impact} scale")
        return join_phrases(bits, 4)
    if "recommendation_matching_ownership" in diffs or "candidate_job_matching_relevance" in diffs:
        bits = ["recommendation/matching ownership"]
        if evals:
            bits.append(f"measurement via {evals}")
        if impact:
            bits.append(f"{impact} scale")
        return join_phrases(bits, 3)
    if "production_ml_infrastructure" in diffs:
        return "production ML delivery and system reliability"
    if "adjacent_but_strong_ml_infra" in diffs:
        return "feature-pipeline and ML/data infrastructure depth"
    if "nlp_embedding_experience" in diffs:
        return "NLP/embedding exposure for semantic matching"
    return "applied ML systems work"


def rank_tradeoff(result: RerankResult, rank: int, diffs: list[str]) -> str:
    candidate = result.row["candidate"]
    s = signals(candidate)
    notice = int(float(s.get("notice_period_days") or 0))
    open_to_work = bool(s.get("open_to_work_flag"))
    repeated = bool(result.debug_flags.get("repeated_current_role_evidence", False))
    unique_count = result.unique_current_role_signal_count
    strong_unique = result.strong_unique_current_role_evidence

    if rank <= 4:
        if rank == 4:
            return phrase_variant(candidate, "trade-top4", [
                "It trails the very top profiles only because recruiter/candidate matching is less explicit.",
                "The current search proof is excellent; the only gap versus the top profiles is less direct candidate-matching ownership.",
                "This remains elite, with slightly less end-to-end recruiter-product context than the top three.",
            ])
        return phrase_variant(candidate, "trade-top3", [
            "This combination is why it belongs ahead of broader recommendation or infrastructure profiles.",
            "That mix outranks profiles showing only one of retrieval, ranking, or evaluation.",
            "It has the rare full stack the JD asks for, not just AI keywords or isolated model work.",
        ])

    if not open_to_work:
        return "Not-open status is the main availability tradeoff."
    if notice >= 120:
        return f"{notice}-day notice is the main reason it falls behind faster-moving technical matches."
    if notice >= 90:
        return f"{notice}-day notice raises the bar, so it needs cleaner evidence than shorter-notice peers."
    if repeated and not strong_unique:
        return phrase_variant(candidate, "trade-repeat-weak", [
            "The main gap is less individualized implementation detail.",
            "The delivery story is less specific than cleaner search specialists.",
            "It has relevant signals, but fewer unique implementation details.",
            "The evidence is useful, though less specific than stronger retrieval profiles.",
        ])
    if repeated and unique_count < 4:
        return phrase_variant(candidate, "trade-repeat-some", [
            "The delivery story is less distinctive than cleaner current-role proof.",
            "It needs more unique current-role detail to move higher.",
            "The fit is real, but the implementation proof is less individualized.",
        ])
    if result.seniority_drift_penalty >= 0.035:
        return "Recent leadership/tech-lead direction makes it slightly less ideal for a code-heavy IC role."
    if result.hireability_penalty >= 0.05:
        return "Availability or engagement signals are the practical reason it does not sit higher."
    if "adjacent_but_strong_ml_infra" in diffs:
        return "Useful for ML systems, but less direct on ranking/retrieval ownership than higher profiles."
    if rank > 60:
        return "Final-cut value comes from adjacent fit; direct ranking/retrieval proof is thinner."
    if rank > 20:
        return "Good shortlist value, but not as complete as profiles combining retrieval, evaluation, and product matching."
    return "It stays below the cleanest profiles because the JD-specific evidence is narrower."


def advantage_intro(result: RerankResult, rank: int) -> str:
    candidate = result.row["candidate"]
    if rank <= 4:
        return phrase_variant(candidate, "adv-intro-elite", [
            "the clearest separator is",
            "its top-tier case rests on",
            "few nearby profiles show the same mix of",
            "the strongest proof is",
            "the rank is driven by",
        ])
    if rank <= 20:
        return phrase_variant(candidate, "adv-intro-high", [
            "it earns a higher shortlist spot through",
            "the shortlist case comes from",
            "its best evidence is",
            "the strongest support is",
            "the profile is carried by",
            "the useful differentiator is",
        ])
    return phrase_variant(candidate, "adv-intro-mid", [
        "kept in the cut for",
        "its value is mostly",
        "the useful part is",
        "the profile helps through",
        "shortlist value comes from",
        "it remains useful for",
    ])


def rank_bridge_sentence(result: RerankResult, rank: int, lead: str, advantage: str, tradeoff: str) -> str:
    candidate = result.row["candidate"]
    if rank <= 4:
        return phrase_variant(candidate, "rank-bridge-elite", [
            f"{lead}; {advantage} is the clearest separator. {tradeoff}",
            f"{lead}; its top-tier case rests on {advantage}. {tradeoff}",
            f"{lead}; few nearby profiles show the same mix of {advantage}. {tradeoff}",
            f"{lead}; the strongest proof is {advantage}. {tradeoff}",
        ])
    if rank <= 20:
        return phrase_variant(candidate, "rank-bridge-high", [
            f"{lead}; it earns a higher shortlist spot through {advantage}. {tradeoff}",
            f"{lead}; the shortlist case comes from {advantage}. {tradeoff}",
            f"{lead}; its best evidence is {advantage}. {tradeoff}",
            f"{lead}; {advantage} keeps it ahead of broader ML profiles. {tradeoff}",
            f"{lead}; the useful differentiator is {advantage}. {tradeoff}",
            f"{lead}; that evidence beats generic AI exposure because of {advantage}. {tradeoff}",
            f"{lead}; it remains above adjacent infrastructure profiles because of {advantage}. {tradeoff}",
        ])
    if rank <= 60:
        return phrase_variant(candidate, "rank-bridge-mid", [
            f"{lead}; kept in the cut for {advantage}. {tradeoff}",
            f"{lead}; {advantage} makes it shortlistable. {tradeoff}",
            f"{lead}; its value is mostly {advantage}. {tradeoff}",
            f"{lead}; the profile helps through {advantage}. {tradeoff}",
            f"{lead}; {advantage} is useful, but not enough to outrank cleaner specialists. {tradeoff}",
            f"{lead}; the case is narrower, but {advantage} still matters. {tradeoff}",
            f"{lead}; it stays in the final pool because of {advantage}. {tradeoff}",
        ])
    return phrase_variant(candidate, "rank-bridge-late", [
        f"{lead}; kept for {advantage}. {tradeoff}",
        f"{lead}; final-cut value comes from {advantage}. {tradeoff}",
        f"{lead}; this is more of a depth/adjacency pick through {advantage}. {tradeoff}",
        f"{lead}; {advantage} earns the slot, while the JD-specific evidence is thinner. {tradeoff}",
        f"{lead}; useful support comes from {advantage}. {tradeoff}",
    ])


def rank_context_reasoning(result: RerankResult, rank: int, diffs: list[str]) -> str:
    candidate = result.row["candidate"]
    lead = template_cautious_evidence_sentence(result, diffs) if should_use_template_cautious_reasoning(result) else top_evidence_sentence(candidate, diffs)
    lead = contextualized_lead(candidate, lead.rstrip("."), rank)
    advantage = advantage_summary(result, diffs)
    tradeoff = rank_tradeoff(result, rank, diffs)
    intro = advantage_intro(result, rank)
    sentence = rank_bridge_sentence(result, rank, lead, advantage, tradeoff)
    if word_count(sentence) > 52 and word_count(lead) >= 28:
        sentence = f"{lead}. {tradeoff}"
    if word_count(sentence) > 52:
        sentence = f"{lead}; {intro} {advantage}."
    return re.sub(r"\s+", " ", sentence).replace(" ,", ",").replace(" ;", ";").strip()


def build_ranked_reasoning(result: RerankResult, rank: int) -> str:
    candidate = result.row["candidate"]
    diffs = candidate_differentiators(candidate, result.components)
    primary = result.primary_differentiator
    if primary not in diffs:
        diffs.insert(0, primary)
    hireability = exceptional_hireability_note(candidate)

    reasoning = rank_context_reasoning(result, rank, diffs)
    if hireability and rank <= 40 and word_count(reasoning) <= 42 and result.hireability_penalty < 0.04:
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
    template_analysis: dict[str, object] | None = None,
    hands_score: float | None = None,
    leadership_score: float | None = None,
) -> dict[str, object]:
    if template_analysis is None:
        template_analysis = career_template_analysis(candidate, realism_index)
    if repeated_template_details is None:
        repeated_template_details = (
            float(template_analysis["career_template_penalty"]),
            int(template_analysis["max_template_candidate_count"]),
            bool(template_analysis["same_candidate_repeated_template"]),
            list(template_analysis["career_template_reason_codes"]),
        )
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
    education_bad, education_reasons = education_timeline_inconsistency(candidate)
    strong_current_unique, current_unique_count, _ = unique_current_role_evidence(candidate)
    compound_bad = (
        education_bad
        and not strong_current_unique
        and current_unique_count < 3
        and has_domain_mismatch(candidate)
        and bool(template_analysis["current_role_template_heavy"])
        and (
            int(template_analysis["max_template_candidate_count"]) >= 4
            or float(template_analysis["repeated_template_ratio"]) >= 0.50
        )
    )
    return {
        "salary_range_inverted": salary_range_inverted(candidate),
        "weak_hireability_signal_count": weak_count,
        "repeated_career_template_count": repeated_count,
        "same_candidate_repeated_template": same_repeated,
        "repeated_current_role_evidence": bool(template_analysis["current_role_template_heavy"]),
        "repeated_evidence_count": int(template_analysis["max_template_candidate_count"]),
        "same_candidate_repeated_evidence": same_repeated,
        "current_role_domain_mismatch": has_domain_mismatch(candidate),
        "company_domain_description_mismatch": has_domain_mismatch(candidate),
        "education_timeline_inconsistency": education_bad,
        "education_timeline_reason_codes": education_reasons,
        "compound_realism_top100_risk": compound_bad,
        "recent_hands_on_score": hands_score,
        "recent_leadership_score": leadership_score,
        "recent_leadership_heavy": leadership_score >= 0.67 and hands_score < 0.67,
        "tech_lead_aspiration_soft_risk": tech_lead_risk,
        "repeated_career_template_penalty": repeated_penalty,
        "career_template_penalty": float(template_analysis["career_template_penalty"]),
        "career_template_reason_codes": list(template_analysis["career_template_reason_codes"]),
        "repeated_role_count": int(template_analysis["repeated_role_count"]),
        "max_template_candidate_count": int(template_analysis["max_template_candidate_count"]),
        "repeated_template_ratio": float(template_analysis["repeated_template_ratio"]),
        "current_role_template_heavy": bool(template_analysis["current_role_template_heavy"]),
        "unique_evidence_score": float(template_analysis["unique_evidence_score"]),
        "unique_evidence_reason_codes": list(template_analysis["unique_evidence_reason_codes"]),
        "template_penalty_before_unique_offset": float(template_analysis["template_penalty_before_unique_offset"]),
        "template_penalty_after_unique_offset": float(template_analysis["template_penalty_after_unique_offset"]),
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
    template_analysis = career_template_analysis(candidate, realism_index, records=description_records)
    template_penalty = float(template_analysis["career_template_penalty"])
    template_reason_codes = list(template_analysis["career_template_reason_codes"])
    guardrail_current_heavy = guardrail_recent_role_template_heavy(
        candidate,
        realism_index,
        template_analysis,
        records=description_records,
    )
    strong_current_unique, current_unique_count, current_unique_reasons = unique_current_role_evidence(
        candidate,
        records=description_records,
    )
    if guardrail_current_heavy and strong_current_unique:
        current_reason_set = set(current_unique_reasons)
        has_distinctive_system_detail = bool(
            current_reason_set
            & {"specific_scale", "specific_architecture", "production_ops"}
        )
        if not has_distinctive_system_detail:
            strong_current_unique = False
    realism_penalty, realism_reason_codes = evidence_realism_penalty(
        candidate,
        realism_index,
        strong_technical,
        weak_hireability_count=weak_count,
        records=description_records,
    )
    seniority_drift_penalty, senior_ic_bonus, seniority_alignment_reason_codes = seniority_alignment_adjustment(
        candidate,
        strong_technical,
        hireability_penalty,
        realism_penalty,
    )
    nontechnical_penalty_raw = hireability_penalty + realism_penalty + template_penalty + seniority_drift_penalty
    nontechnical_penalty_cap = 0.18 if strong_technical else 0.30
    nontechnical_penalty_total = min(nontechnical_penalty_raw, nontechnical_penalty_cap)
    final = clamp(
        (base * penalty_factor)
        - nontechnical_penalty_total
        + senior_ic_bonus
    )
    components["hireability_penalty"] = hireability_penalty
    components["evidence_realism_penalty"] = realism_penalty
    components["career_template_penalty"] = template_penalty
    components["seniority_drift_penalty"] = seniority_drift_penalty
    components["senior_ic_alignment_bonus"] = senior_ic_bonus
    components["nontechnical_penalty_total"] = nontechnical_penalty_total
    debug_flags = candidate_debug_flags(
        candidate,
        realism_index,
        nontechnical_penalty_total=nontechnical_penalty_total,
        weak_count=weak_count,
        template_analysis=template_analysis,
    )
    debug_flags["current_role_template_heavy"] = guardrail_current_heavy
    debug_flags["repeated_current_role_evidence"] = guardrail_current_heavy
    debug_flags["strong_unique_current_role_evidence"] = strong_current_unique
    debug_flags["unique_current_role_signal_count"] = current_unique_count
    debug_flags["unique_current_role_reason_codes"] = current_unique_reasons
    debug_flags["top5_template_guardrail_penalty"] = 0.0
    debug_flags["top5_template_guardrail_reason"] = ""
    debug_flags["top5_guardrail_applied"] = False
    debug_flags["top10_repeated_evidence_guardrail_penalty"] = 0.0
    debug_flags["top10_repeated_evidence_guardrail_reason"] = ""
    debug_flags["top10_repeated_evidence_guardrail_applied"] = False
    debug_flags["top10_not_open_guardrail_penalty"] = 0.0
    debug_flags["top10_not_open_guardrail_reason"] = ""
    debug_flags["top10_not_open_guardrail_applied"] = False
    debug_flags["jd_alignment_boost"] = 0.0
    debug_flags["jd_alignment_penalty"] = 0.0
    debug_flags["jd_alignment_reason_codes"] = []
    debug_flags["notice_availability_penalty"] = 0.0
    debug_flags["behavioral_availability_penalty"] = 0.0
    debug_flags["final_jd_adjusted_score"] = final
    debug_flags["recruiter_facing_matching_evidence"] = False
    debug_flags["production_eval_ownership"] = False
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
        career_template_penalty=template_penalty,
        career_template_reason_codes=template_reason_codes,
        seniority_drift_penalty=seniority_drift_penalty,
        senior_ic_alignment_bonus=senior_ic_bonus,
        seniority_alignment_reason_codes=seniority_alignment_reason_codes,
        top5_template_guardrail_penalty=0.0,
        top5_template_guardrail_reason="",
        top5_guardrail_applied=False,
        strong_unique_current_role_evidence=strong_current_unique,
        unique_current_role_signal_count=current_unique_count,
        final_calibration_penalty=0.0,
        final_calibration_reason_codes=[],
        jd_alignment_boost=0.0,
        jd_alignment_penalty=0.0,
        jd_alignment_reason_codes=[],
        notice_availability_penalty=0.0,
        behavioral_availability_penalty=0.0,
        final_jd_adjusted_score=final,
        recruiter_facing_matching_evidence=False,
        production_eval_ownership=False,
        top10_repeated_evidence_guardrail_penalty=0.0,
        top10_repeated_evidence_guardrail_reason="",
        top10_repeated_evidence_guardrail_applied=False,
        top10_not_open_guardrail_penalty=0.0,
        top10_not_open_guardrail_reason="",
        top10_not_open_guardrail_applied=False,
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


TOP5_TEMPLATE_GUARDRAIL_CODES = {
    "high_frequency_career_template",
    "multiple_repeated_templates",
    "same_candidate_repeated_role_template",
    "template_heavy_recent_role",
}


def technical_composite_score(result: RerankResult) -> float:
    return (
        0.45 * float(result.components.get("career_evidence_score", 0.0))
        + 0.35 * float(result.components.get("jd_pillar_score", 0.0))
        + 0.20 * float(result.components.get("hands_on_depth_score", 0.0))
    )


def result_has_high_frequency_template(result: RerankResult) -> bool:
    codes = set(result.career_template_reason_codes)
    return bool(
        int(result.debug_flags.get("max_template_candidate_count", 0) or 0) >= 4
        or bool(codes & TOP5_TEMPLATE_GUARDRAIL_CODES)
    )


def result_is_template_top5_risk(result: RerankResult) -> bool:
    return bool(
        result.debug_flags.get("current_role_template_heavy", False)
        and result_has_high_frequency_template(result)
        and not result.strong_unique_current_role_evidence
    )


def result_has_severe_hireability_risk(result: RerankResult) -> bool:
    return bool(
        result.hireability_penalty >= 0.07
        or int(result.debug_flags.get("weak_hireability_signal_count", 0) or 0) >= 4
    )


def comparable_guardrail_replacement(candidate: RerankResult, target: RerankResult) -> bool:
    if bool(candidate.debug_flags.get("current_role_template_heavy", False)):
        return False
    if result_is_template_top5_risk(candidate):
        return False
    if result_has_severe_hireability_risk(candidate):
        return False
    candidate_unique = float(candidate.debug_flags.get("unique_evidence_score", 0.0) or 0.0)
    target_unique = float(target.debug_flags.get("unique_evidence_score", 0.0) or 0.0)
    if candidate_unique <= target_unique and candidate.unique_current_role_signal_count <= target.unique_current_role_signal_count:
        return False
    candidate_technical = technical_composite_score(candidate)
    target_technical = technical_composite_score(target)
    return bool(
        abs(candidate_technical - target_technical) <= 0.05
        or (
            float(candidate.components.get("career_evidence_score", 0.0)) >= 0.75
            and float(candidate.components.get("jd_pillar_score", 0.0)) >= 0.75
        )
        or candidate_unique >= target_unique + 0.15
        or candidate.unique_current_role_signal_count >= target.unique_current_role_signal_count + 2
    )


def top5_guardrail_penalty_size(result: RerankResult) -> float:
    codes = set(result.career_template_reason_codes)
    has_some_unique = result.unique_current_role_signal_count > 0 or float(result.debug_flags.get("unique_evidence_score", 0.0) or 0.0) > 0
    if (
        "same_candidate_repeated_role_template" in codes
        and bool(result.debug_flags.get("current_role_template_heavy", False))
    ):
        return 0.07
    if bool(result.debug_flags.get("current_role_template_heavy", False)) and not result.strong_unique_current_role_evidence:
        return 0.05
    if has_some_unique:
        return 0.03
    return 0.05


def top5_guardrail_reason(result: RerankResult, replacement: RerankResult) -> str:
    if "same_candidate_repeated_role_template" in result.career_template_reason_codes:
        return f"same_candidate_repeated_recent_template;nearby_unique_current_evidence:{replacement.candidate_id}"
    if "high_frequency_career_template" in result.career_template_reason_codes:
        return f"high_frequency_recent_template;nearby_unique_current_evidence:{replacement.candidate_id}"
    return f"template_heavy_recent_role_without_strong_unique_current_evidence;nearby_unique_current_evidence:{replacement.candidate_id}"


def apply_top5_template_guardrail(results: list[RerankResult]) -> None:
    if len(results) < 6:
        return
    iterations = 0
    while iterations < 5:
        iterations += 1
        results.sort(key=lambda item: (-item.final_score, item.candidate_id))
        changed = False
        floor_index = min(99, len(results) - 1)
        floor_score = results[floor_index].final_score
        for result in list(results[:5]):
            if result.top5_guardrail_applied or not result_is_template_top5_risk(result):
                continue
            nearby = [
                candidate for candidate in results[5:20]
                if comparable_guardrail_replacement(candidate, result)
            ]
            if not nearby:
                continue
            replacement = max(
                nearby,
                key=lambda item: (
                    item.unique_current_role_signal_count,
                    float(item.debug_flags.get("unique_evidence_score", 0.0) or 0.0),
                    technical_composite_score(item),
                    item.final_score,
                ),
            )
            penalty = top5_guardrail_penalty_size(result)
            penalty = max(0.0, min(penalty, result.final_score - floor_score + 0.000001))
            if penalty <= 0:
                continue
            reason = top5_guardrail_reason(result, replacement)
            result.final_score = clamp(result.final_score - penalty)
            result.top5_template_guardrail_penalty = penalty
            result.top5_template_guardrail_reason = reason
            result.top5_guardrail_applied = True
            result.components["top5_template_guardrail_penalty"] = penalty
            result.debug_flags["top5_template_guardrail_penalty"] = penalty
            result.debug_flags["top5_template_guardrail_reason"] = reason
            result.debug_flags["top5_guardrail_applied"] = True
            changed = True
        if not changed:
            break
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))


def final_calibration_penalty(result: RerankResult) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []
    max_template_count = int(result.debug_flags.get("max_template_candidate_count", 0) or 0)
    repeated_ratio = float(result.debug_flags.get("repeated_template_ratio", 0.0) or 0.0)
    current_template_heavy = bool(result.debug_flags.get("current_role_template_heavy", False))
    high_frequency_current = current_template_heavy and max_template_count >= 4

    if high_frequency_current:
        if result.strong_unique_current_role_evidence:
            if max_template_count >= 9 and repeated_ratio >= 0.75:
                penalty += 0.030
                reasons.append("high_frequency_current_template_with_repeated_ratio")
            else:
                penalty += 0.015
                reasons.append("high_frequency_current_template_with_unique_current_evidence")
        else:
            penalty += 0.025
            reasons.append("high_frequency_current_template_weak_unique_current_evidence")
            if repeated_ratio >= 0.75:
                penalty += 0.010
                reasons.append("dominant_repeated_current_template")
            if "same_candidate_repeated_role_template" in result.career_template_reason_codes:
                penalty += 0.010
                reasons.append("same_candidate_repeated_current_template")

    s = signals(result.row["candidate"])
    notice = float(s.get("notice_period_days") if s.get("notice_period_days") is not None else 0.0)
    open_to_work = bool(s.get("open_to_work_flag"))
    location_score = float(result.components.get("location_availability_score", 0.0))
    if not open_to_work and current_template_heavy:
        if result.strong_unique_current_role_evidence:
            penalty += 0.025
            reasons.append("not_open_template_heavy_with_unique_current_evidence")
        else:
            penalty += 0.040
            reasons.append("not_open_template_heavy_weak_unique_current_evidence")
        if repeated_ratio >= 0.75 or max_template_count >= 9:
            penalty += 0.010
            reasons.append("not_open_high_frequency_current_template")

    if current_template_heavy and notice >= 90 and location_score <= 0.60:
        penalty += 0.020
        reasons.append("template_heavy_long_notice_location_risk")

    if (
        result.final_score >= 0.80
        and not open_to_work
        and not result.strong_unique_current_role_evidence
    ):
        penalty += 0.012
        reasons.append("not_open_weak_unique_current_evidence")
        if notice > 30:
            penalty += 0.005
            reasons.append("not_open_notice_above_30")

    compound_bad, compound_reasons = compound_realism_top100_risk(result)
    if compound_bad:
        penalty += 0.060
        reasons.extend(compound_reasons)

    return min(penalty, 0.090), unique_ordered(reasons, 10)


def apply_final_score_calibration(results: list[RerankResult]) -> None:
    if not results:
        return
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))
    floor_score = results[min(99, len(results) - 1)].final_score
    for result in results:
        penalty, reasons = final_calibration_penalty(result)
        if penalty <= 0:
            result.debug_flags["final_calibration_penalty"] = 0.0
            result.debug_flags["final_calibration_reason_codes"] = []
            continue
        penalty = min(penalty, max(0.0, result.final_score - floor_score + 0.000001))
        if penalty <= 0:
            result.debug_flags["final_calibration_penalty"] = 0.0
            result.debug_flags["final_calibration_reason_codes"] = []
            continue
        result.final_score = clamp(result.final_score - penalty)
        result.final_calibration_penalty = penalty
        result.final_calibration_reason_codes = reasons
        result.components["final_calibration_penalty"] = penalty
        result.debug_flags["final_calibration_penalty"] = penalty
        result.debug_flags["final_calibration_reason_codes"] = reasons
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))


PRODUCT_COMPANY_TERMS = [
    "product",
    "marketplace",
    "e-commerce",
    "ecommerce",
    "consumer internet",
    "saas",
    "fintech",
    "gaming",
    "platform",
    "software",
    "internet",
]

SERVICES_CONSULTING_TERMS = [
    "it services",
    "consulting",
    "service company",
    "services",
    "consultant",
    "client delivery",
]

RECRUITER_MATCHING_TERMS = [
    "candidate-jd matching",
    "candidate jd matching",
    "candidate-role matching",
    "candidate matching",
    "job matching",
    "recruiter-facing",
    "recruiter facing",
    "recruiter engagement",
    "recruiter feedback",
    "recruiter labels",
    "candidate corpus",
    "time-to-shortlist",
]


def signal_float(candidate: dict, key: str, default: float = 0.0) -> float:
    value = signals(candidate).get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def days_since_last_active(candidate: dict) -> int | None:
    last_active = parse_date(signals(candidate).get("last_active_date"))
    if last_active is None:
        return None
    return (REFERENCE_DATE - last_active).days


def recruiter_facing_matching_evidence(result: RerankResult) -> bool:
    career = career_text(result.row["candidate"])
    return has_any(career, RECRUITER_MATCHING_TERMS)


def production_eval_ownership(result: RerankResult) -> bool:
    candidate = result.row["candidate"]
    career = career_text(candidate)
    has_eval = has_any(career, EVALUATION_TERMS + CURRENT_EVALUATION_TERMS)
    has_production = has_any(career, PRODUCTION_TERMS + CURRENT_OPS_TERMS + ["deployed", "shipped", "production"])
    has_hands_on = has_any(career, HANDS_ON_VERBS)
    return has_eval and has_production and has_hands_on


def product_company_hands_on_ranking_retrieval(result: RerankResult) -> bool:
    candidate = result.row["candidate"]
    career = career_text(candidate)
    if not has_any(career, RANKING_RETRIEVAL_TERMS + RECOMMENDATION_MATCHING_TERMS):
        return False
    if not has_any(career, HANDS_ON_VERBS):
        return False
    product_like = False
    services_like = False
    for job in candidate.get("career_history", []) or []:
        company_context = normalize(
            " ".join(str(job.get(key) or "") for key in ["company", "industry", "description"])
        )
        if has_any(company_context, PRODUCT_COMPANY_TERMS):
            product_like = True
        if has_any(company_context, SERVICES_CONSULTING_TERMS):
            services_like = True
    return product_like and not services_like


def skills_outweigh_career_evidence(candidate: dict) -> bool:
    skill_matches = match_terms(skills_text(candidate), AI_SKILL_TERMS)
    if len(skill_matches) < 5:
        return False
    career = career_text(candidate)
    return not has_any(career, RANKING_RETRIEVAL_TERMS + RECOMMENDATION_MATCHING_TERMS + EVALUATION_TERMS)


def jd_alignment_adjustment(result: RerankResult) -> tuple[float, float, list[str], float, float, bool, bool]:
    candidate = result.row["candidate"]
    boost = 0.0
    penalty = 0.0
    notice_penalty = 0.0
    behavioral_penalty = 0.0
    reasons: list[str] = []

    recruiter_match = recruiter_facing_matching_evidence(result)
    production_eval = production_eval_ownership(result)
    product_hands_on = product_company_hands_on_ranking_retrieval(result)

    if result.strong_unique_current_role_evidence:
        unique_boost = 0.010 + min(max(result.unique_current_role_signal_count - 3, 0), 3) * 0.005
        boost += min(unique_boost, 0.025)
        reasons.append("strong_unique_current_role_evidence")
    if recruiter_match:
        boost += 0.020 if result.strong_unique_current_role_evidence else 0.010
        reasons.append("recruiter_facing_candidate_matching_evidence")
    if production_eval:
        boost += 0.015
        reasons.append("production_evaluation_ownership")
    if product_hands_on:
        boost += 0.010
        reasons.append("product_company_hands_on_ranking_retrieval")

    repeated_current = bool(result.debug_flags.get("repeated_current_role_evidence", False))
    repeated_count = int(result.debug_flags.get("repeated_evidence_count", 0) or 0)
    repeated_ratio = float(result.debug_flags.get("repeated_template_ratio", 0.0) or 0.0)
    same_repeated = bool(result.debug_flags.get("same_candidate_repeated_evidence", False))
    if repeated_current:
        if result.strong_unique_current_role_evidence:
            if repeated_count >= 9 or repeated_ratio >= 0.75:
                penalty += 0.015
                reasons.append("repeated_current_role_evidence_with_unique_details")
        else:
            penalty += 0.040
            reasons.append("repeated_current_role_evidence_weak_uniqueness")
    if same_repeated:
        penalty += 0.020 if repeated_current else 0.015
        reasons.append("same_repeated_description_across_roles")

    notice = signal_float(candidate, "notice_period_days", 0.0)
    response_rate = signal_float(candidate, "recruiter_response_rate", 1.0)
    response_hours = signal_float(candidate, "avg_response_time_hours", 0.0)
    interview_completion = signal_float(candidate, "interview_completion_rate", 1.0)
    open_to_work = bool(signals(candidate).get("open_to_work_flag"))
    verified_email = bool(signals(candidate).get("verified_email"))

    if notice >= 120:
        notice_penalty += 0.050 if not result.strong_unique_current_role_evidence else 0.030
        reasons.append("notice_120_plus_raises_bar")
    elif notice >= 90:
        if repeated_current or not result.strong_unique_current_role_evidence:
            notice_penalty += 0.030
            reasons.append("notice_90_with_repeated_or_less_distinctive_evidence")
        else:
            notice_penalty += 0.015
            reasons.append("notice_90")
    elif notice >= 45 and (response_rate < 0.60 or repeated_current):
        notice_penalty += 0.008
        reasons.append("notice_45_to_60_with_availability_tradeoff")

    if notice >= 90 and repeated_current:
        notice_penalty += 0.020
        reasons.append("notice_90_plus_repeated_evidence")
    if notice >= 90 and response_rate < 0.60:
        notice_penalty += 0.015
        reasons.append("notice_90_plus_weak_response_rate")

    if response_rate < 0.60:
        behavioral_penalty += 0.015 if response_rate < 0.40 else 0.010
        reasons.append("recruiter_response_rate_below_060")
    if response_hours > 48:
        behavioral_penalty += 0.015 if response_hours > 96 else 0.010
        reasons.append("response_time_above_48h")
    if 0 <= interview_completion < 0.75:
        behavioral_penalty += 0.015 if interview_completion < 0.50 else 0.010
        reasons.append("interview_completion_below_075")
    if not verified_email:
        behavioral_penalty += 0.005
        reasons.append("email_not_verified")
    inactive_days = days_since_last_active(candidate)
    if inactive_days is not None and inactive_days > 120:
        behavioral_penalty += 0.010
        reasons.append("not_recently_active")

    if not open_to_work:
        if result.strong_unique_current_role_evidence and result.unique_current_role_signal_count >= 4:
            behavioral_penalty += 0.020
            reasons.append("not_open_to_work_but_distinctive_evidence")
        else:
            behavioral_penalty += 0.040
            reasons.append("not_open_to_work")

    if salary_range_inverted(candidate):
        penalty += 0.015
        reasons.append("salary_range_inverted")

    recent_hands = float(result.debug_flags.get("recent_hands_on_score", 0.0) or 0.0)
    recent_leadership = float(result.debug_flags.get("recent_leadership_score", 0.0) or 0.0)
    if recent_leadership >= 0.67 and recent_hands < 0.34:
        penalty += 0.040
        reasons.append("leadership_heavy_without_recent_coding")
    elif recent_leadership >= 0.67 and recent_hands < 0.67:
        penalty += 0.020
        reasons.append("leadership_heavy_with_limited_recent_coding")

    if skills_outweigh_career_evidence(candidate):
        penalty += 0.020
        reasons.append("skills_keywords_stronger_than_career_history")

    if has_any(summary_text(candidate) + " " + career_text(candidate), GENAI_EXPLORER_TERMS) and not has_any(
        career_text(candidate),
        DIRECT_SEARCH_EVALUATION_EVIDENCE_TERMS,
    ):
        penalty += 0.015
        reasons.append("genai_or_wrapper_evidence_without_pre_llm_depth")

    penalty += notice_penalty + behavioral_penalty
    boost = min(boost, 0.050)
    penalty = min(penalty, 0.100)
    return (
        boost,
        penalty,
        unique_ordered(reasons, 16),
        min(notice_penalty, 0.070),
        min(behavioral_penalty, 0.080),
        recruiter_match,
        production_eval,
    )


def apply_jd_alignment_adjustment(results: list[RerankResult]) -> None:
    if not results:
        return
    for result in results:
        (
            boost,
            penalty,
            reasons,
            notice_penalty,
            behavioral_penalty,
            recruiter_match,
            production_eval,
        ) = jd_alignment_adjustment(result)
        result.jd_alignment_boost = boost
        result.jd_alignment_penalty = penalty
        result.jd_alignment_reason_codes = reasons
        result.notice_availability_penalty = notice_penalty
        result.behavioral_availability_penalty = behavioral_penalty
        result.recruiter_facing_matching_evidence = recruiter_match
        result.production_eval_ownership = production_eval
        result.final_score = clamp(result.final_score + boost - penalty)
        result.final_jd_adjusted_score = result.final_score
        result.components["jd_alignment_boost"] = boost
        result.components["jd_alignment_penalty"] = penalty
        result.components["notice_availability_penalty"] = notice_penalty
        result.components["behavioral_availability_penalty"] = behavioral_penalty
        result.components["final_jd_adjusted_score"] = result.final_jd_adjusted_score
        result.debug_flags["jd_alignment_boost"] = boost
        result.debug_flags["jd_alignment_penalty"] = penalty
        result.debug_flags["jd_alignment_reason_codes"] = reasons
        result.debug_flags["notice_availability_penalty"] = notice_penalty
        result.debug_flags["behavioral_availability_penalty"] = behavioral_penalty
        result.debug_flags["final_jd_adjusted_score"] = result.final_jd_adjusted_score
        result.debug_flags["recruiter_facing_matching_evidence"] = recruiter_match
        result.debug_flags["production_eval_ownership"] = production_eval
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))


def result_has_repeated_current_role_evidence(result: RerankResult) -> bool:
    return bool(result.debug_flags.get("repeated_current_role_evidence", False))


def result_has_distinctive_current_production_evidence(result: RerankResult) -> bool:
    if result_has_severe_hireability_risk(result):
        return False
    return bool(
        result.strong_unique_current_role_evidence
        or result.unique_current_role_signal_count >= 4
        or (
            float(result.debug_flags.get("unique_evidence_score", 0.0) or 0.0) >= 0.50
            and result.unique_current_role_signal_count >= 2
        )
    )


def result_is_weak_repeated_top10_risk(result: RerankResult) -> bool:
    if not result_has_repeated_current_role_evidence(result):
        return False
    if not result.strong_unique_current_role_evidence:
        return True

    notice = float(signals(result.row["candidate"]).get("notice_period_days") or 0.0)
    location_score = float(result.components.get("location_availability_score", 0.0))
    weak_count = int(result.debug_flags.get("weak_hireability_signal_count", 0) or 0)
    same_repeated = bool(result.debug_flags.get("same_candidate_repeated_evidence", False))
    return bool(
        (same_repeated and (notice >= 90 or location_score <= 0.60))
        or (notice >= 90 and weak_count >= 4)
        or (notice >= 90 and location_score <= 0.60)
    )


def repeated_evidence_guardrail_penalty_size(result: RerankResult) -> float:
    if not result_is_weak_repeated_top10_risk(result):
        return 0.0
    penalty = 0.035
    if result.unique_current_role_signal_count <= 1:
        penalty += 0.015
    if bool(result.debug_flags.get("same_candidate_repeated_evidence", False)):
        penalty += 0.025
    if bool(result.debug_flags.get("current_role_domain_mismatch", False)):
        penalty += 0.020
    notice = float(signals(result.row["candidate"]).get("notice_period_days") or 0.0)
    location_score = float(result.components.get("location_availability_score", 0.0))
    weak_count = int(result.debug_flags.get("weak_hireability_signal_count", 0) or 0)
    if notice >= 90 and location_score <= 0.60:
        penalty += 0.020
    if notice >= 90 and weak_count >= 4:
        penalty += 0.020
    max_template_count = int(result.debug_flags.get("repeated_evidence_count", 0) or 0)
    repeated_ratio = float(result.debug_flags.get("repeated_template_ratio", 0.0) or 0.0)
    if max_template_count >= 9 or repeated_ratio >= 0.75:
        penalty += 0.010
    return min(max(penalty, 0.035), 0.090)


def top10_repeated_evidence_guardrail_reason(result: RerankResult, replacement: RerankResult) -> str:
    reasons = ["repeated_current_role_evidence"]
    if result.unique_current_role_signal_count < 3:
        reasons.append("weak_unique_current_role_evidence")
    if bool(result.debug_flags.get("same_candidate_repeated_evidence", False)):
        reasons.append("same_candidate_repeated_evidence")
    if bool(result.debug_flags.get("current_role_domain_mismatch", False)):
        reasons.append("current_role_domain_mismatch")
    notice = float(signals(result.row["candidate"]).get("notice_period_days") or 0.0)
    location_score = float(result.components.get("location_availability_score", 0.0))
    weak_count = int(result.debug_flags.get("weak_hireability_signal_count", 0) or 0)
    if notice >= 90 and location_score <= 0.60:
        reasons.append("long_notice_location_tradeoff")
    if notice >= 90 and weak_count >= 4:
        reasons.append("long_notice_weak_hireability_cluster")
    reasons.append(f"nearby_distinctive_current_evidence:{replacement.candidate_id}")
    return ";".join(reasons)


def apply_top10_repeated_evidence_guardrail(results: list[RerankResult]) -> None:
    if len(results) < 11:
        return
    iterations = 0
    while iterations < 7:
        iterations += 1
        results.sort(key=lambda item: (-item.final_score, item.candidate_id))
        nearby_distinctive = [
            item for item in results[10:25]
            if result_has_distinctive_current_production_evidence(item)
        ]
        if not nearby_distinctive:
            return
        changed = False
        floor_score = results[min(99, len(results) - 1)].final_score
        for result in list(results[3:10]):
            if not result_is_weak_repeated_top10_risk(result):
                continue
            replacement = max(
                nearby_distinctive,
                key=lambda item: (
                    item.unique_current_role_signal_count,
                    float(item.debug_flags.get("unique_evidence_score", 0.0) or 0.0),
                    technical_composite_score(item),
                    item.final_score,
                ),
            )
            needed = max(0.0, result.final_score - replacement.final_score + 0.000001)
            remaining_guardrail = max(0.0, 0.090 - result.top10_repeated_evidence_guardrail_penalty)
            target_penalty = repeated_evidence_guardrail_penalty_size(result)
            penalty = min(max(0.035, needed), target_penalty, remaining_guardrail)
            penalty = min(penalty, max(0.0, result.final_score - floor_score + 0.000001))
            if penalty <= 0:
                continue
            reason = top10_repeated_evidence_guardrail_reason(result, replacement)
            result.final_score = clamp(result.final_score - penalty)
            result.top10_repeated_evidence_guardrail_penalty += penalty
            if result.top10_repeated_evidence_guardrail_reason:
                result.top10_repeated_evidence_guardrail_reason = (
                    f"{result.top10_repeated_evidence_guardrail_reason};{reason}"
                )
            else:
                result.top10_repeated_evidence_guardrail_reason = reason
            result.top10_repeated_evidence_guardrail_applied = True
            result.components["top10_repeated_evidence_guardrail_penalty"] = (
                result.top10_repeated_evidence_guardrail_penalty
            )
            result.debug_flags["top10_repeated_evidence_guardrail_penalty"] = (
                result.top10_repeated_evidence_guardrail_penalty
            )
            result.debug_flags["top10_repeated_evidence_guardrail_reason"] = (
                result.top10_repeated_evidence_guardrail_reason
            )
            result.debug_flags["top10_repeated_evidence_guardrail_applied"] = True
            changed = True
        if not changed:
            break
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))


def result_is_open_to_work(result: RerankResult) -> bool:
    return bool(signals(result.row["candidate"]).get("open_to_work_flag"))


def not_open_elite_exception(result: RerankResult, nearby_open: list[RerankResult]) -> bool:
    if result.debug_flags.get("current_role_template_heavy", False):
        return False
    if not result.strong_unique_current_role_evidence or result.unique_current_role_signal_count < 4:
        return False
    unique_score = float(result.debug_flags.get("unique_evidence_score", 0.0) or 0.0)
    if unique_score < 0.75:
        return False
    if not nearby_open:
        return True
    best_open_unique_count = max(item.unique_current_role_signal_count for item in nearby_open)
    best_open_technical = max(technical_composite_score(item) for item in nearby_open)
    return bool(
        result.unique_current_role_signal_count >= best_open_unique_count + 1
        or technical_composite_score(result) >= best_open_technical + 0.05
    )


def top10_not_open_guardrail_reason(result: RerankResult, replacement: RerankResult) -> str:
    if result.debug_flags.get("current_role_template_heavy", False):
        return f"not_open_template_heavy_top10;nearby_open_candidate:{replacement.candidate_id}"
    if not result.strong_unique_current_role_evidence:
        return f"not_open_weak_unique_current_evidence_top10;nearby_open_candidate:{replacement.candidate_id}"
    return f"not_open_not_elite_vs_nearby_open_candidate:{replacement.candidate_id}"


def apply_top10_not_open_guardrail(results: list[RerankResult]) -> None:
    if len(results) < 11:
        return
    iterations = 0
    while iterations < 5:
        iterations += 1
        results.sort(key=lambda item: (-item.final_score, item.candidate_id))
        nearby_open = [
            item for item in results[10:20]
            if result_is_open_to_work(item) and not result_has_severe_hireability_risk(item)
        ]
        if not nearby_open:
            nearby_open = [
                item for item in results[10:20]
                if result_is_open_to_work(item)
            ]
        changed = False
        floor_score = results[min(99, len(results) - 1)].final_score
        for result in list(results[:10]):
            if result_is_open_to_work(result):
                continue
            if not_open_elite_exception(result, nearby_open):
                continue
            if nearby_open:
                replacement = max(
                    nearby_open,
                    key=lambda item: (
                        item.final_score,
                        item.unique_current_role_signal_count,
                        technical_composite_score(item),
                    ),
                )
                needed = max(0.0, result.final_score - replacement.final_score + 0.000001)
            else:
                replacement = results[min(10, len(results) - 1)]
                needed = 0.035
            remaining_guardrail = max(0.0, 0.120 - result.top10_not_open_guardrail_penalty)
            penalty = min(max(0.035, needed), 0.080, remaining_guardrail)
            penalty = min(penalty, max(0.0, result.final_score - floor_score + 0.000001))
            if penalty <= 0:
                continue
            reason = top10_not_open_guardrail_reason(result, replacement)
            result.final_score = clamp(result.final_score - penalty)
            result.top10_not_open_guardrail_penalty += penalty
            if result.top10_not_open_guardrail_reason:
                result.top10_not_open_guardrail_reason = f"{result.top10_not_open_guardrail_reason};{reason}"
            else:
                result.top10_not_open_guardrail_reason = reason
            result.top10_not_open_guardrail_applied = True
            result.components["top10_not_open_guardrail_penalty"] = result.top10_not_open_guardrail_penalty
            result.debug_flags["top10_not_open_guardrail_penalty"] = result.top10_not_open_guardrail_penalty
            result.debug_flags["top10_not_open_guardrail_reason"] = result.top10_not_open_guardrail_reason
            result.debug_flags["top10_not_open_guardrail_applied"] = True
            changed = True
        if not changed:
            break
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))


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
            "rank",
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
            "career_template_penalty",
            "top5_template_guardrail_penalty",
            "final_calibration_penalty",
            "jd_alignment_boost",
            "jd_alignment_penalty",
            "final_jd_adjusted_score",
            "top10_repeated_evidence_guardrail_penalty",
            "top10_not_open_guardrail_penalty",
            "notice_availability_penalty",
            "behavioral_availability_penalty",
            "seniority_drift_penalty",
            "senior_ic_alignment_bonus",
            "nontechnical_penalty_total",
            "penalties",
            "reason_codes",
            "hireability_reason_codes",
            "evidence_realism_reason_codes",
            "career_template_reason_codes",
            "top5_template_guardrail_reason",
            "final_calibration_reason_codes",
            "jd_alignment_reason_codes",
            "top10_repeated_evidence_guardrail_reason",
            "top10_not_open_guardrail_reason",
            "seniority_alignment_reason_codes",
            "salary_range_inverted",
            "weak_hireability_signal_count",
            "repeated_career_template_count",
            "same_candidate_repeated_template",
            "repeated_role_count",
            "max_template_candidate_count",
            "repeated_template_ratio",
            "current_role_template_heavy",
            "strong_unique_current_role_evidence",
            "unique_current_role_signal_count",
            "recruiter_facing_matching_evidence",
            "production_eval_ownership",
            "unique_current_role_reason_codes",
            "top5_guardrail_applied",
            "top10_repeated_evidence_guardrail_applied",
            "top10_not_open_guardrail_applied",
            "unique_evidence_score",
            "unique_evidence_reason_codes",
            "template_penalty_before_unique_offset",
            "template_penalty_after_unique_offset",
            "company_domain_description_mismatch",
            "repeated_current_role_evidence",
            "repeated_evidence_count",
            "same_candidate_repeated_evidence",
            "current_role_domain_mismatch",
            "education_timeline_inconsistency",
            "education_timeline_reason_codes",
            "compound_realism_top100_risk",
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
        for rank, result in enumerate(results, start=1):
            candidate = result.row["candidate"]
            s = signals(candidate)
            row = {
                "candidate_id": result.candidate_id,
                "rank": rank,
                "original_rank": result.original_rank,
                "original_hybrid_score": f"{result.original_hybrid_score:.6f}",
                "final_score": f"{result.final_score:.6f}",
                "primary_differentiator": result.primary_differentiator,
                "hireability_penalty": f"{result.hireability_penalty:.6f}",
                "evidence_realism_penalty": f"{result.evidence_realism_penalty:.6f}",
                "career_template_penalty": f"{result.career_template_penalty:.6f}",
                "top5_template_guardrail_penalty": f"{result.top5_template_guardrail_penalty:.6f}",
                "final_calibration_penalty": f"{result.final_calibration_penalty:.6f}",
                "jd_alignment_boost": f"{result.jd_alignment_boost:.6f}",
                "jd_alignment_penalty": f"{result.jd_alignment_penalty:.6f}",
                "final_jd_adjusted_score": f"{result.final_jd_adjusted_score:.6f}",
                "top10_repeated_evidence_guardrail_penalty": f"{result.top10_repeated_evidence_guardrail_penalty:.6f}",
                "top10_not_open_guardrail_penalty": f"{result.top10_not_open_guardrail_penalty:.6f}",
                "notice_availability_penalty": f"{result.notice_availability_penalty:.6f}",
                "behavioral_availability_penalty": f"{result.behavioral_availability_penalty:.6f}",
                "seniority_drift_penalty": f"{result.seniority_drift_penalty:.6f}",
                "senior_ic_alignment_bonus": f"{result.senior_ic_alignment_bonus:.6f}",
                "nontechnical_penalty_total": f"{float(result.debug_flags.get('nontechnical_penalty_total', 0.0)):.6f}",
                "penalties": ";".join(result.penalties),
                "reason_codes": ";".join(result.reason_codes),
                "hireability_reason_codes": ";".join(result.hireability_reason_codes),
                "evidence_realism_reason_codes": ";".join(result.evidence_realism_reason_codes),
                "career_template_reason_codes": ";".join(result.career_template_reason_codes),
                "top5_template_guardrail_reason": result.top5_template_guardrail_reason,
                "final_calibration_reason_codes": ";".join(result.final_calibration_reason_codes),
                "jd_alignment_reason_codes": ";".join(result.jd_alignment_reason_codes),
                "top10_repeated_evidence_guardrail_reason": result.top10_repeated_evidence_guardrail_reason,
                "top10_not_open_guardrail_reason": result.top10_not_open_guardrail_reason,
                "seniority_alignment_reason_codes": ";".join(result.seniority_alignment_reason_codes),
                "salary_range_inverted": result.debug_flags.get("salary_range_inverted", False),
                "weak_hireability_signal_count": result.debug_flags.get("weak_hireability_signal_count", 0),
                "repeated_career_template_count": result.debug_flags.get("repeated_career_template_count", 0),
                "same_candidate_repeated_template": result.debug_flags.get("same_candidate_repeated_template", False),
                "repeated_role_count": result.debug_flags.get("repeated_role_count", 0),
                "max_template_candidate_count": result.debug_flags.get("max_template_candidate_count", 0),
                "repeated_template_ratio": f"{float(result.debug_flags.get('repeated_template_ratio', 0.0)):.6f}",
                "current_role_template_heavy": result.debug_flags.get("current_role_template_heavy", False),
                "strong_unique_current_role_evidence": result.strong_unique_current_role_evidence,
                "unique_current_role_signal_count": result.unique_current_role_signal_count,
                "recruiter_facing_matching_evidence": result.recruiter_facing_matching_evidence,
                "production_eval_ownership": result.production_eval_ownership,
                "unique_current_role_reason_codes": ";".join(result.debug_flags.get("unique_current_role_reason_codes", [])),
                "top5_guardrail_applied": result.top5_guardrail_applied,
                "top10_repeated_evidence_guardrail_applied": result.top10_repeated_evidence_guardrail_applied,
                "top10_not_open_guardrail_applied": result.top10_not_open_guardrail_applied,
                "unique_evidence_score": f"{float(result.debug_flags.get('unique_evidence_score', 0.0)):.6f}",
                "unique_evidence_reason_codes": ";".join(result.debug_flags.get("unique_evidence_reason_codes", [])),
                "template_penalty_before_unique_offset": f"{float(result.debug_flags.get('template_penalty_before_unique_offset', 0.0)):.6f}",
                "template_penalty_after_unique_offset": f"{float(result.debug_flags.get('template_penalty_after_unique_offset', 0.0)):.6f}",
                "company_domain_description_mismatch": result.debug_flags.get("company_domain_description_mismatch", False),
                "repeated_current_role_evidence": result.debug_flags.get("repeated_current_role_evidence", False),
                "repeated_evidence_count": result.debug_flags.get("repeated_evidence_count", 0),
                "same_candidate_repeated_evidence": result.debug_flags.get("same_candidate_repeated_evidence", False),
                "current_role_domain_mismatch": result.debug_flags.get("current_role_domain_mismatch", False),
                "education_timeline_inconsistency": result.debug_flags.get("education_timeline_inconsistency", False),
                "education_timeline_reason_codes": ";".join(result.debug_flags.get("education_timeline_reason_codes", [])),
                "compound_realism_top100_risk": result.debug_flags.get("compound_realism_top100_risk", False),
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
    if candidate_ids[:4] != EXPECTED_TOP4_ORDER:
        raise ValueError(f"Top 4 changed unexpectedly: {candidate_ids[:4]}")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate_id in final top100")
    if any(cid in rejected_ids for cid in candidate_ids):
        raise ValueError("Hard rejected candidate found in final top100")
    if any(not result.reasoning.strip() for result in results):
        raise ValueError("Missing reasoning in final top100")
    for result in results:
        words = word_count(result.reasoning)
        if words < 20 or words > 52:
            raise ValueError(f"Reasoning length outside 20-52 words for {result.candidate_id}: {words}")
    if any(math.isnan(result.final_score) for result in results):
        raise ValueError("NaN score in final top100")
    for result in results:
        penalty_values = [
            result.hireability_penalty,
            result.evidence_realism_penalty,
            result.career_template_penalty,
            result.top5_template_guardrail_penalty,
            result.final_calibration_penalty,
            result.jd_alignment_boost,
            result.jd_alignment_penalty,
            result.notice_availability_penalty,
            result.behavioral_availability_penalty,
            result.final_jd_adjusted_score,
            result.top10_repeated_evidence_guardrail_penalty,
            result.top10_not_open_guardrail_penalty,
            result.seniority_drift_penalty,
            float(result.debug_flags.get("nontechnical_penalty_total", 0.0)),
        ]
        if any(math.isnan(value) for value in penalty_values):
            raise ValueError(f"NaN penalty in final top100 for {result.candidate_id}")
    for result in results[:5]:
        if result_is_template_top5_risk(result):
            raise ValueError(
                "Template-heavy high-frequency candidate without strong current evidence remained in top5: "
                f"{result.candidate_id}"
            )
    nearby_open_for_top10 = [
        result for result in results[10:20]
        if result_is_open_to_work(result)
    ]
    for result in results[:10]:
        if not result_is_open_to_work(result) and not not_open_elite_exception(result, nearby_open_for_top10):
            raise ValueError(f"Non-elite not-open candidate remained in top10: {result.candidate_id}")
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
        "rank",
        "evidence_realism_penalty",
        "evidence_realism_reason_codes",
        "career_template_penalty",
        "career_template_reason_codes",
        "top5_template_guardrail_penalty",
        "top5_template_guardrail_reason",
        "final_calibration_penalty",
        "final_calibration_reason_codes",
        "jd_alignment_boost",
        "jd_alignment_penalty",
        "jd_alignment_reason_codes",
        "notice_availability_penalty",
        "behavioral_availability_penalty",
        "final_jd_adjusted_score",
        "top10_repeated_evidence_guardrail_penalty",
        "top10_repeated_evidence_guardrail_reason",
        "top10_not_open_guardrail_penalty",
        "top10_not_open_guardrail_reason",
        "salary_range_inverted",
        "weak_hireability_signal_count",
        "repeated_career_template_count",
        "same_candidate_repeated_template",
        "repeated_role_count",
        "max_template_candidate_count",
        "repeated_template_ratio",
        "current_role_template_heavy",
        "strong_unique_current_role_evidence",
        "unique_current_role_signal_count",
        "recruiter_facing_matching_evidence",
        "production_eval_ownership",
        "top5_guardrail_applied",
        "top10_repeated_evidence_guardrail_applied",
        "top10_not_open_guardrail_applied",
        "unique_evidence_score",
        "unique_evidence_reason_codes",
        "template_penalty_before_unique_offset",
        "template_penalty_after_unique_offset",
        "company_domain_description_mismatch",
        "repeated_current_role_evidence",
        "repeated_evidence_count",
        "same_candidate_repeated_evidence",
        "current_role_domain_mismatch",
        "education_timeline_inconsistency",
        "education_timeline_reason_codes",
        "compound_realism_top100_risk",
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
    apply_top5_template_guardrail(results)
    apply_final_score_calibration(results)
    apply_jd_alignment_adjustment(results)
    apply_top10_repeated_evidence_guardrail(results)
    apply_top10_not_open_guardrail(results)
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

#!/usr/bin/env python3
"""Evidence-based final reranker from top-2000 retrieval output to submission.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


DEFAULT_TOP2000_PATH = Path("top_2000_candidates.jsonl")
DEFAULT_SUBMISSION_PATH = Path("submission.csv")
DEFAULT_DEBUG_PATH = Path("rerank_debug_top2000.csv")
DEFAULT_REJECTED_PATH = Path("rejected_candidates.jsonl")
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


@dataclass
class RerankResult:
    candidate_id: str
    original_rank: int
    original_hybrid_score: float
    final_score: float
    components: dict[str, float]
    penalties: list[str]
    reason_codes: list[str]
    reasoning: str
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

    notice = float(signals(candidate).get("notice_period_days") or 0)
    open_to_work = bool(signals(candidate).get("open_to_work_flag"))
    if notice > 120 and not open_to_work:
        factor *= 0.75
        penalties.append("long notice and not open to work")
        reason_codes.append("very_long_notice_penalty")
    elif notice > 90:
        factor *= 0.90
        penalties.append("long notice")
        reason_codes.append("long_notice_penalty")

    return factor, penalties, reason_codes


def rerank_row(row: dict) -> RerankResult:
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
    final = clamp(base * penalty_factor)
    components = {
        "career_evidence_score": career_score,
        "jd_pillar_score": pillar_score,
        "hands_on_depth_score": hands_score,
        "seniority_fit_score": seniority_score,
        "redrob_score": behavior_score,
        "location_availability_score": location_score,
        "penalty_factor": penalty_factor,
    }
    return RerankResult(
        candidate_id=str(candidate["candidate_id"]),
        original_rank=int(row.get("rank") or 0),
        original_hybrid_score=original_hybrid,
        final_score=final,
        components=components,
        penalties=penalties,
        reason_codes=reason_codes,
        reasoning=build_reasoning(candidate, components, penalties),
        row=row,
    )


def build_reasoning(candidate: dict, components: dict[str, float], penalties: list[str]) -> str:
    career = career_text(candidate)
    p = profile(candidate)
    s = signals(candidate)
    evidence = []
    core_hits = match_terms(career, RANKING_RETRIEVAL_TERMS + RECOMMENDATION_MATCHING_TERMS + EMBEDDING_VECTOR_TERMS)
    prod_hits = match_terms(career, PRODUCTION_TERMS)
    eval_hits = match_terms(career, EVALUATION_TERMS)

    if core_hits:
        evidence.append("career-backed " + ", ".join(core_hits[:3]))
    else:
        evidence.append("limited direct retrieval career evidence")
    if prod_hits:
        evidence.append("production signal: " + ", ".join(prod_hits[:2]))
    if eval_hits:
        evidence.append("evaluation signal: " + ", ".join(eval_hits[:2]))

    availability = []
    if s.get("open_to_work_flag"):
        availability.append("open to work")
    notice = s.get("notice_period_days")
    if notice is not None:
        availability.append(f"{int(float(notice))}-day notice")
    response_rate = s.get("recruiter_response_rate")
    if response_rate is not None and float(response_rate) >= 0.5:
        availability.append(f"{float(response_rate):.2f} response rate")

    penalty_note = f" Penalty: {penalties[0]}." if penalties else ""
    return (
        f"{p.get('current_title', 'Candidate')} with {float(p.get('years_of_experience') or 0.0):.1f} yrs; "
        f"{'; '.join(evidence[:3])}. "
        f"Availability: {', '.join(availability[:3]) if availability else 'limited Redrob signal'}."
        f"{penalty_note}"
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


def write_submission(results: list[RerankResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        previous_score: float | None = None
        for rank, result in enumerate(results, start=1):
            score = round(result.final_score, 6)
            if previous_score is not None and score >= previous_score:
                score = max(0.0, previous_score - 0.000001)
            previous_score = score
            writer.writerow([result.candidate_id, rank, f"{score:.6f}", result.reasoning])


def write_debug(results: list[RerankResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "candidate_id",
            "original_rank",
            "original_hybrid_score",
            "final_score",
            "career_evidence_score",
            "jd_pillar_score",
            "hands_on_depth_score",
            "seniority_fit_score",
            "redrob_score",
            "location_availability_score",
            "penalty_factor",
            "penalties",
            "reason_codes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "candidate_id": result.candidate_id,
                "original_rank": result.original_rank,
                "original_hybrid_score": f"{result.original_hybrid_score:.6f}",
                "final_score": f"{result.final_score:.6f}",
                "penalties": ";".join(result.penalties),
                "reason_codes": ";".join(result.reason_codes),
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
    if any(math.isnan(result.final_score) for result in results):
        raise ValueError("NaN score in final top100")
    for previous, current in zip(results, results[1:]):
        if previous.final_score < current.final_score:
            raise ValueError("Scores are not sorted descending")


def rerank(top2000_path: Path, rejected_path: Path, submission_path: Path, debug_path: Path) -> None:
    rejected_ids = load_rejected_ids(rejected_path)
    rows = load_top2000(top2000_path)
    results = [
        rerank_row(row)
        for row in rows
        if str(row.get("candidate_id") or row.get("candidate", {}).get("candidate_id")) not in rejected_ids
    ]
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))
    selected = results[:100]
    validate_submission(selected, rejected_ids)
    write_submission(selected, submission_path)
    write_debug(results, debug_path)
    print(f"Loaded top2000 rows: {len(rows)}")
    print(f"Reranked candidates: {len(results)}")
    print(f"Wrote submission: {submission_path}")
    print(f"Wrote debug CSV: {debug_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-based rerank from top2000 to final top100 submission")
    parser.add_argument("--input", type=Path, default=DEFAULT_TOP2000_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--debug-out", type=Path, default=DEFAULT_DEBUG_PATH)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED_PATH)
    args = parser.parse_args()
    rerank(args.input, args.rejected, args.out, args.debug_out)


if __name__ == "__main__":
    main()

"""Rule-based scoring logic."""

from __future__ import annotations

from src.config import FEATURE_WEIGHTS
from src.features import (
    behavior_multiplier,
    culture_fit,
    current_title,
    evaluation_fit,
    experience_score,
    ideal_recruiter_fit,
    product_company_fit,
    production_fit,
    python_skill_score,
    ranking_fit,
    retrieval_fit,
    title_fit,
    years_of_experience,
)


def score_candidate(candidate: dict) -> tuple[float, dict]:
    retrieval_score, retrieval_terms = retrieval_fit(candidate)
    ranking_score, ranking_terms = ranking_fit(candidate)
    production_score, production_terms = production_fit(candidate)
    evaluation_score, evaluation_terms = evaluation_fit(candidate)
    culture_score, culture_terms = culture_fit(candidate)
    ideal_score, ideal_details = ideal_recruiter_fit(candidate)
    product_score, product_seen, service_only = product_company_fit(candidate)
    multiplier, behavior = behavior_multiplier(candidate)

    scores = {
        "title_fit": title_fit(candidate),
        "experience_fit": experience_score(candidate),
        "retrieval_fit": retrieval_score,
        "ranking_fit": ranking_score,
        "production_fit": production_score,
        "evaluation_fit": evaluation_score,
        "python_fit": python_skill_score(candidate),
        "product_company_fit": product_score,
        "culture_fit": culture_score,
        "ideal_recruiter_fit": ideal_score,
    }
    weighted_score = sum(scores[name] * FEATURE_WEIGHTS[name] for name in FEATURE_WEIGHTS)
    score = weighted_score * multiplier

    details = {
        "current_title": current_title(candidate),
        "years_of_experience": years_of_experience(candidate),
        "feature_scores": scores,
        "retrieval_terms": retrieval_terms,
        "ranking_terms": ranking_terms,
        "production_terms": production_terms,
        "evaluation_terms": evaluation_terms,
        "culture_terms": culture_terms,
        "ideal_recruiter": ideal_details,
        "product_seen": product_seen,
        "service_only": service_only,
        "behavior_multiplier": multiplier,
        "behavior": behavior,
    }
    return score, details

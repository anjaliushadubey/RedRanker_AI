"""JD-specific feature functions."""

from __future__ import annotations

from functools import lru_cache
import re

from src.config import (
    EVALUATION_TERMS,
    IDEAL_EXPERIENCE_YEARS,
    MAX_TARGET_EXPERIENCE_YEARS,
    MIN_TARGET_EXPERIENCE_YEARS,
    NEGATIVE_TITLES,
    PRODUCT_COMPANIES,
    PRODUCTION_TERMS,
    PYTHON_RELATED_TERMS,
    PYTHON_SKILL_NAMES,
    RANKING_TERMS,
    RETRIEVAL_TERMS,
    SERVICE_COMPANIES,
    STRONG_TITLES,
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize(value: object) -> str:
    return str(value or "").strip().lower()


def years_of_experience(candidate: dict) -> float:
    return float(candidate.get("profile", {}).get("years_of_experience") or 0.0)


def current_title(candidate: dict) -> str:
    return str(candidate.get("profile", {}).get("current_title") or "")


def all_titles(candidate: dict) -> list[str]:
    cached = candidate.get("_all_titles")
    if cached is not None:
        return cached
    titles = [current_title(candidate)]
    titles.extend(str(job.get("title") or "") for job in candidate.get("career_history", []))
    candidate["_all_titles"] = titles
    return titles


def candidate_text(candidate: dict) -> str:
    cached = candidate.get("_candidate_text")
    if cached is not None:
        return cached
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
        profile.get("current_company", ""),
        profile.get("location", ""),
        profile.get("country", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.extend(
            [
                job.get("company", ""),
                job.get("title", ""),
                job.get("industry", ""),
                job.get("description", ""),
            ]
        )
    for education in candidate.get("education", []):
        parts.extend(
            [
                education.get("institution", ""),
                education.get("degree", ""),
                education.get("field_of_study", ""),
            ]
        )
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    text = " ".join(str(part) for part in parts).lower()
    candidate["_candidate_text"] = text
    return text


def match_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term_in_text(text, term)]


def has_any_term(text: str, terms: list[str]) -> bool:
    return any(term_in_text(text, term) for term in terms)


def term_in_text(text: str, term: str) -> bool:
    term = term.lower()
    if len(term) <= 3 and term.replace(".", "").isalnum():
        return short_term_pattern(term).search(text) is not None
    return term in text


@lru_cache(maxsize=256)
def short_term_pattern(term: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")


def term_score(text: str, terms: list[str], saturation: int) -> tuple[float, list[str]]:
    matches = match_terms(text, terms)
    return clamp(len(matches) / saturation), matches


def title_fit(candidate: dict) -> float:
    title_text = " ".join(all_titles(candidate)).lower()
    current = normalize(current_title(candidate))
    strong_match = any(title in title_text for title in STRONG_TITLES)
    negative_current = any(title == current for title in NEGATIVE_TITLES)

    if strong_match and not negative_current:
        return 1.0
    if strong_match:
        return 0.55
    if negative_current:
        return 0.05
    if any(word in title_text for word in ["engineer", "developer", "scientist"]):
        return 0.45
    return 0.20


def experience_score(candidate: dict) -> float:
    """Prefer 5-9 years, with peak around 7 years."""
    years = years_of_experience(candidate)
    if MIN_TARGET_EXPERIENCE_YEARS <= years <= MAX_TARGET_EXPERIENCE_YEARS:
        return 1.0 - clamp(abs(years - IDEAL_EXPERIENCE_YEARS) / 8.0)
    if years < MIN_TARGET_EXPERIENCE_YEARS:
        return clamp(years / MIN_TARGET_EXPERIENCE_YEARS) * 0.75
    return clamp(1.0 - ((years - MAX_TARGET_EXPERIENCE_YEARS) / 12.0)) * 0.85


def python_skill_score(candidate: dict) -> float:
    for skill in candidate.get("skills", []):
        name = str(skill.get("name", "")).strip().lower()
        if name in PYTHON_SKILL_NAMES:
            proficiency = str(skill.get("proficiency", "")).lower()
            proficiency_score = {
                "beginner": 0.35,
                "intermediate": 0.60,
                "advanced": 0.85,
                "expert": 1.00,
            }.get(proficiency, 0.50)
            duration_score = min(float(skill.get("duration_months") or 0.0) / 36.0, 1.0)
            endorsement_score = min(float(skill.get("endorsements") or 0.0) / 50.0, 1.0)
            return 0.60 * proficiency_score + 0.25 * duration_score + 0.15 * endorsement_score
    text = candidate_text(candidate)
    if any(term in text for term in PYTHON_RELATED_TERMS):
        return 0.45
    return 0.0


def retrieval_fit(candidate: dict) -> tuple[float, list[str]]:
    return term_score(candidate_text(candidate), RETRIEVAL_TERMS, saturation=4)


def ranking_fit(candidate: dict) -> tuple[float, list[str]]:
    return term_score(candidate_text(candidate), RANKING_TERMS, saturation=3)


def production_fit(candidate: dict) -> tuple[float, list[str]]:
    return term_score(candidate_text(candidate), PRODUCTION_TERMS, saturation=5)


def evaluation_fit(candidate: dict) -> tuple[float, list[str]]:
    return term_score(candidate_text(candidate), EVALUATION_TERMS, saturation=2)


def product_company_fit(candidate: dict) -> tuple[float, bool, bool]:
    companies = [normalize(job.get("company")) for job in candidate.get("career_history", [])]
    companies.append(normalize(candidate.get("profile", {}).get("current_company")))
    product_seen = any(company in PRODUCT_COMPANIES for company in companies)
    service_only = bool(companies) and all(company in SERVICE_COMPANIES for company in companies if company)
    score = 1.0 if product_seen else 0.25
    if service_only:
        score = 0.0
    return score, product_seen, service_only


def behavior_multiplier(candidate: dict) -> tuple[float, dict]:
    signals = candidate.get("redrob_signals", {})
    response_rate = clamp(float(signals.get("recruiter_response_rate") or 0.0))
    interview_rate = clamp(float(signals.get("interview_completion_rate") or 0.0))
    profile_complete = clamp(float(signals.get("profile_completeness_score") or 0.0) / 100.0)
    notice_days = float(signals.get("notice_period_days") or 180.0)
    notice_score = 1.0 - clamp(notice_days / 180.0)
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    github = float(signals.get("github_activity_score") or -1.0)
    github_score = 0.0 if github < 0 else clamp(github / 100.0)

    behavior_score = (
        0.28 * response_rate
        + 0.20 * interview_rate
        + 0.18 * profile_complete
        + 0.16 * notice_score
        + 0.10 * open_to_work
        + 0.08 * github_score
    )
    multiplier = 0.75 + (0.40 * behavior_score)
    details = {
        "response_rate": response_rate,
        "interview_rate": interview_rate,
        "notice_days": notice_days,
        "open_to_work": bool(signals.get("open_to_work_flag")),
        "behavior_score": behavior_score,
    }
    return multiplier, details

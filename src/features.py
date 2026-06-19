"""JD-specific feature functions."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import re

from src.config import (
    APPLIED_ML_ROLE_TERMS,
    CULTURE_ASYNC_WRITING_TERMS,
    CULTURE_FAST_AMBIGUITY_TERMS,
    CULTURE_OPEN_DECISION_TERMS,
    CULTURE_OWNERSHIP_TERMS,
    EVALUATION_TERMS,
    IDEAL_EXPERIENCE_YEARS,
    LLM_INTEGRATION_JUDGMENT_TERMS,
    MAX_TARGET_EXPERIENCE_YEARS,
    MEANINGFUL_SCALE_TERMS,
    MIN_TARGET_EXPERIENCE_YEARS,
    NEGATIVE_TITLES,
    PREFERRED_OFFICE_LOCATIONS,
    PRODUCT_COMPANIES,
    PRODUCTION_TERMS,
    PYTHON_RELATED_TERMS,
    PYTHON_SKILL_NAMES,
    RANKING_TERMS,
    REFERENCE_DATE,
    RETRIEVAL_JUDGMENT_TERMS,
    RETRIEVAL_TERMS,
    SERVICE_COMPANIES,
    SHIPPED_RANKING_SYSTEM_TERMS,
    STRONG_TITLES,
    WELCOME_INDIA_LOCATIONS,
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
    if term.replace(".", "").isalnum():
        if term not in text:
            return False
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


def culture_fit(candidate: dict) -> tuple[float, list[str]]:
    text = candidate_text(candidate)
    category_matches = [
        match_terms(text, CULTURE_ASYNC_WRITING_TERMS),
        match_terms(text, CULTURE_OWNERSHIP_TERMS),
        match_terms(text, CULTURE_OPEN_DECISION_TERMS),
        match_terms(text, CULTURE_FAST_AMBIGUITY_TERMS),
    ]
    matched_categories = sum(1 for matches in category_matches if matches)
    unique_terms = []
    seen = set()
    for matches in category_matches:
        for term in matches:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)

    breadth_score = matched_categories / len(category_matches)
    depth_score = clamp(len(unique_terms) / 8.0)
    return 0.70 * breadth_score + 0.30 * depth_score, unique_terms


def product_company_fit(candidate: dict) -> tuple[float, bool, bool]:
    companies = [normalize(job.get("company")) for job in candidate.get("career_history", [])]
    companies.append(normalize(candidate.get("profile", {}).get("current_company")))
    product_seen = any(company in PRODUCT_COMPANIES for company in companies)
    service_only = bool(companies) and all(company in SERVICE_COMPANIES for company in companies if company)
    score = 1.0 if product_seen else 0.25
    if service_only:
        score = 0.0
    return score, product_seen, service_only


def ideal_recruiter_fit(candidate: dict) -> tuple[float, dict]:
    applied_months = applied_ml_product_months(candidate)
    shipped_score, shipped_terms = shipped_ranking_system_fit(candidate)
    judgment_score, judgment_terms = systems_judgment_fit(candidate)
    location_score = noida_pune_relocation_score(candidate)
    market_score = job_market_signal_score(candidate)
    experience_focus = ideal_6_to_8_experience_score(candidate)
    applied_score = applied_ml_product_years_score(applied_months / 12.0)

    score = (
        0.15 * experience_focus
        + 0.20 * applied_score
        + 0.25 * shipped_score
        + 0.20 * judgment_score
        + 0.10 * location_score
        + 0.10 * market_score
    )
    details = {
        "experience_focus": experience_focus,
        "applied_ml_product_years": round(applied_months / 12.0, 2),
        "applied_ml_product_score": applied_score,
        "shipped_system_score": shipped_score,
        "shipped_system_terms": shipped_terms,
        "systems_judgment_score": judgment_score,
        "systems_judgment_terms": judgment_terms,
        "location_score": location_score,
        "job_market_score": market_score,
    }
    return score, details


def ideal_6_to_8_experience_score(candidate: dict) -> float:
    years = years_of_experience(candidate)
    if 6.0 <= years <= 8.0:
        return 1.0
    if 5.0 <= years < 6.0:
        return 0.75
    if 8.0 < years <= 9.0:
        return 0.75
    if 4.0 <= years < 5.0:
        return 0.45
    if 9.0 < years <= 11.0:
        return 0.45
    return 0.20


def applied_ml_product_months(candidate: dict) -> int:
    months = 0
    for job in candidate.get("career_history", []):
        title = normalize(job.get("title"))
        company = normalize(job.get("company"))
        industry = normalize(job.get("industry"))
        job_text = " ".join(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        ).lower()
        has_applied_ml_role = has_any_term(title + " " + job_text, APPLIED_ML_ROLE_TERMS)
        product_like = (
            company in PRODUCT_COMPANIES
            or "product" in job_text
            or "customer-facing" in job_text
            or "users" in job_text
            or "startup" in job_text
            or "saas" in industry
        )
        service_like = company in SERVICE_COMPANIES or industry == "it services"
        if has_applied_ml_role and product_like and not service_like:
            months += int(job.get("duration_months") or 0)
    return months


def applied_ml_product_years_score(years: float) -> float:
    if 4.0 <= years <= 5.5:
        return 1.0
    if 3.0 <= years < 4.0:
        return 0.72
    if 5.5 < years <= 7.0:
        return 0.82
    if 2.0 <= years < 3.0:
        return 0.42
    if years > 7.0:
        return 0.62
    return 0.15


def shipped_ranking_system_fit(candidate: dict) -> tuple[float, list[str]]:
    terms: list[str] = []
    for job in candidate.get("career_history", []):
        job_text = " ".join(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        ).lower()
        system_terms = match_terms(job_text, SHIPPED_RANKING_SYSTEM_TERMS + RANKING_TERMS + RETRIEVAL_TERMS)
        scale_terms = match_terms(job_text, MEANINGFUL_SCALE_TERMS + PRODUCTION_TERMS)
        build_terms = match_terms(job_text, ["built", "implemented", "deployed", "shipped", "launched", "owned"])
        if system_terms and scale_terms and build_terms:
            terms = unique_ordered(system_terms[:3] + scale_terms[:2] + build_terms[:1])
            return 1.0, terms
        if system_terms and (scale_terms or build_terms):
            terms = unique_ordered(system_terms[:3] + scale_terms[:1] + build_terms[:1])
    if terms:
        return 0.55, terms
    return 0.0, []


def systems_judgment_fit(candidate: dict) -> tuple[float, list[str]]:
    text = candidate_text(candidate)
    retrieval_terms = match_terms(text, RETRIEVAL_JUDGMENT_TERMS)
    evaluation_terms = match_terms(text, EVALUATION_TERMS)
    llm_terms = match_terms(text, LLM_INTEGRATION_JUDGMENT_TERMS)
    built_terms = match_terms(text, ["built", "implemented", "deployed", "shipped", "launched", "owned"])

    categories = sum(1 for group in [retrieval_terms, evaluation_terms, llm_terms] if group)
    if categories == 0:
        return 0.0, []
    score = (0.25 * categories) + (0.25 if built_terms else 0.0)
    score += min((len(retrieval_terms) + len(evaluation_terms) + len(llm_terms)) / 10.0, 0.25)
    return clamp(score), unique_ordered(retrieval_terms[:2] + evaluation_terms[:2] + llm_terms[:2])


def noida_pune_relocation_score(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    location = normalize(profile.get("location"))
    country = normalize(profile.get("country"))
    willing_to_relocate = bool(signals.get("willing_to_relocate"))
    if any(city in location for city in PREFERRED_OFFICE_LOCATIONS):
        return 1.0
    if willing_to_relocate and country == "india":
        return 0.88
    if country == "india" and any(city in location for city in WELCOME_INDIA_LOCATIONS):
        return 0.76
    if country == "india":
        return 0.55
    return 0.30 if willing_to_relocate else 0.0


def job_market_signal_score(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {})
    score = 0.0
    if signals.get("open_to_work_flag"):
        score += 0.35
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (REFERENCE_DATE - last_active).days
        if days_inactive <= 30:
            score += 0.35
        elif days_inactive <= 90:
            score += 0.22
        elif days_inactive <= 180:
            score += 0.10
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    if response_rate >= 0.50:
        score += 0.30
    elif response_rate >= 0.25:
        score += 0.18
    elif response_rate >= 0.10:
        score += 0.08
    return clamp(score)


def behavior_multiplier(candidate: dict) -> tuple[float, dict]:
    signals = candidate.get("redrob_signals", {})
    response_rate = clamp(float(signals.get("recruiter_response_rate") or 0.0))
    interview_rate = clamp(float(signals.get("interview_completion_rate") or 0.0))
    profile_complete = clamp(float(signals.get("profile_completeness_score") or 0.0) / 100.0)
    notice_days = float(signals.get("notice_period_days") or 180.0)
    notice_score = notice_period_score(notice_days)
    location_score = location_logistics_score(candidate)
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    github = float(signals.get("github_activity_score") or -1.0)
    github_score = 0.0 if github < 0 else clamp(github / 100.0)

    behavior_score = (
        0.25 * response_rate
        + 0.18 * interview_rate
        + 0.16 * profile_complete
        + 0.16 * notice_score
        + 0.10 * location_score
        + 0.08 * open_to_work
        + 0.07 * github_score
    )
    multiplier = 0.75 + (0.40 * behavior_score)
    details = {
        "response_rate": response_rate,
        "interview_rate": interview_rate,
        "notice_days": notice_days,
        "notice_score": notice_score,
        "location_score": location_score,
        "open_to_work": bool(signals.get("open_to_work_flag")),
        "behavior_score": behavior_score,
    }
    return multiplier, details


def notice_period_score(notice_days: float) -> float:
    if notice_days <= 30:
        return 1.0
    if notice_days <= 60:
        return 0.78
    if notice_days <= 90:
        return 0.58
    if notice_days <= 120:
        return 0.38
    if notice_days <= 150:
        return 0.22
    return 0.10


def location_logistics_score(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    location = normalize(profile.get("location"))
    country = normalize(profile.get("country"))
    willing_to_relocate = bool(signals.get("willing_to_relocate"))

    if any(city in location for city in PREFERRED_OFFICE_LOCATIONS):
        return 1.0
    if country == "india" and any(city in location for city in WELCOME_INDIA_LOCATIONS):
        return 0.92
    if country == "india":
        return 0.82 if willing_to_relocate else 0.70
    return 0.55 if willing_to_relocate else 0.35


def unique_ordered(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def parse_date(value: object):
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None

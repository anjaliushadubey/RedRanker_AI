from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re

from src.config import (
    COMPANY_FOUNDING_YEAR,
    CODING_TERMS,
    HARD_RELEVANCE_TERMS,
    IMPOSSIBLE_TECH_DURATION_LIMITS,
    NEGATIVE_TITLES,
    PRODUCT_OR_PRODUCTION_TERMS,
    RANKING_TERMS,
    REAL_ML_RETRIEVAL_TERMS,
    REFERENCE_DATE,
    RETRIEVAL_TERMS,
    SENIORITY_TRAP_TITLES,
)
from src.features import candidate_text, current_title, has_any_term, match_terms, normalize, years_of_experience


@dataclass(frozen=True)
class HoneypotFinding:
    reason: str
    points: int


def honeypot_score(candidate: dict) -> tuple[int, list[str]]:
    findings: list[HoneypotFinding] = []
    findings.extend(date_consistency_findings(candidate))
    findings.extend(project_duration_consistency_findings(candidate))
    findings.extend(skill_duration_findings(candidate))
    findings.extend(role_overlap_findings(candidate))
    findings.extend(profile_consistency_findings(candidate))
    findings.extend(keyword_stuffing_findings(candidate))
    findings.extend(seniority_findings(candidate))
    findings.extend(company_timeline_findings(candidate))

    total = sum(finding.points for finding in findings)
    reasons = [finding.reason for finding in findings]
    return total, reasons


def date_consistency_findings(candidate: dict) -> list[HoneypotFinding]:
    findings: list[HoneypotFinding] = []
    signals = candidate.get("redrob_signals", {})
    for field in ["last_active_date", "signup_date"]:
        value = parse_date(signals.get(field))
        if value and value > REFERENCE_DATE:
            findings.append(HoneypotFinding(f"future_{field}", 5))
    signup = parse_date(signals.get("signup_date"))
    last_active = parse_date(signals.get("last_active_date"))
    if signup and last_active and last_active < signup:
        findings.append(HoneypotFinding("last_active_before_signup", 1))

    for job in candidate.get("career_history", []):
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        if start and start > REFERENCE_DATE:
            findings.append(HoneypotFinding("future_dated_role_start", 5))
        if end and end > REFERENCE_DATE:
            findings.append(HoneypotFinding("future_dated_role_end", 5))
        if start and end and end < start:
            findings.append(HoneypotFinding("role_end_before_start", 5))
    return findings


def project_duration_consistency_findings(candidate: dict) -> list[HoneypotFinding]:
    findings: list[HoneypotFinding] = []
    for job in candidate.get("career_history", []):
        role_duration = int(float(job.get("duration_months") or 0))
        if role_duration <= 0:
            continue
        description = normalize(job.get("description"))
        claimed_months = [
            int(match.group(1))
            for match in re.finditer(
                r"\b(?:over|for|during|across|in)\s+(\d{1,2})\s+months?\b",
                description,
            )
        ]
        if any(months > role_duration + 1 for months in claimed_months):
            findings.append(HoneypotFinding("project_duration_exceeds_role_duration", 5))
            break
    return findings


def skill_duration_findings(candidate: dict) -> list[HoneypotFinding]:
    findings: list[HoneypotFinding] = []
    total_months = years_of_experience(candidate) * 12
    expert_zero = 0
    expert_under_six = 0

    for skill in candidate.get("skills", []):
        duration = int(skill.get("duration_months") or 0)
        proficiency = normalize(skill.get("proficiency"))
        name = normalize(skill.get("name"))

        if duration > total_months + 36:
            findings.append(HoneypotFinding("skill_duration_exceeds_total_experience", 3))
            break

        if proficiency == "expert" and duration == 0:
            expert_zero += 1
        if proficiency == "expert" and duration < 6:
            expert_under_six += 1

        for term, max_months in IMPOSSIBLE_TECH_DURATION_LIMITS.items():
            if term in name and duration > max_months:
                findings.append(HoneypotFinding(f"unrealistic_{term}_duration", 3))
                break

    if expert_zero >= 5:
        findings.append(HoneypotFinding("expert_skills_with_zero_months", 5))
    if expert_under_six >= 5:
        findings.append(HoneypotFinding("too_many_expert_skills_with_tiny_duration", 3))

    return findings


def role_overlap_findings(candidate: dict) -> list[HoneypotFinding]:
    intervals = []
    total_role_months = 0
    for job in candidate.get("career_history", []):
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) or REFERENCE_DATE
        if not start or end < start:
            continue
        start_month = start.year * 12 + start.month
        end_month = end.year * 12 + end.month
        intervals.append((start_month, end_month))
        total_role_months += max(0, end_month - start_month + 1)

    findings: list[HoneypotFinding] = []
    if intervals:
        months = {}
        for start, end in intervals:
            for month in range(start, end + 1):
                months[month] = months.get(month, 0) + 1
        heavy_overlap_months = sum(1 for count in months.values() if count >= 3)
        if heavy_overlap_months >= 12:
            findings.append(HoneypotFinding("three_or_more_heavily_overlapping_roles", 3))

    claimed_months = years_of_experience(candidate) * 12
    if claimed_months > 0 and total_role_months > claimed_months + 24:
        findings.append(HoneypotFinding("career_months_far_exceed_claimed_experience", 3))

    return findings


def profile_consistency_findings(candidate: dict) -> list[HoneypotFinding]:
    findings: list[HoneypotFinding] = []
    current_jobs = [job for job in candidate.get("career_history", []) if job.get("is_current")]
    if len(current_jobs) > 1:
        findings.append(HoneypotFinding("multiple_current_roles", 5))

    profile = candidate.get("profile", {})
    if current_jobs:
        current = current_jobs[0]
        if current.get("title") and profile.get("current_title") and current.get("title") != profile.get("current_title"):
            findings.append(HoneypotFinding("profile_current_title_mismatch", 3))
        if current.get("company") and profile.get("current_company") and current.get("company") != profile.get("current_company"):
            findings.append(HoneypotFinding("profile_current_company_mismatch", 3))

    for job in candidate.get("career_history", []):
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) or REFERENCE_DATE
        if not start or end < start:
            continue
        actual_months = ((end.year - start.year) * 12) + (end.month - start.month) + 1
        declared_months = int(job.get("duration_months") or 0)
        if abs(actual_months - declared_months) > 12:
            findings.append(HoneypotFinding("role_duration_does_not_match_dates", 5))
            break

    career_starts = [
        parse_date(job.get("start_date"))
        for job in candidate.get("career_history", [])
    ]
    career_starts = [start for start in career_starts if start is not None]
    claimed_months = years_of_experience(candidate) * 12
    if career_starts and claimed_months > 0:
        first_start = min(career_starts)
        observed_months = ((REFERENCE_DATE.year - first_start.year) * 12) + (REFERENCE_DATE.month - first_start.month) + 1
        if claimed_months > observed_months + 24:
            findings.append(HoneypotFinding("claimed_experience_exceeds_observed_career_span", 3))

    stated_years = explicit_profile_experience_years(candidate)
    claimed_years = years_of_experience(candidate)
    if stated_years and min(abs(value - claimed_years) for value in stated_years) > 5:
        findings.append(HoneypotFinding("profile_text_experience_contradicts_structured_experience", 5))

    for education in candidate.get("education", []):
        start_year = education.get("start_year")
        end_year = education.get("end_year")
        if isinstance(start_year, int) and isinstance(end_year, int) and end_year < start_year:
            findings.append(HoneypotFinding("education_end_before_start", 5))
            break
        if isinstance(start_year, int) and start_year > REFERENCE_DATE.year + 1:
            findings.append(HoneypotFinding("education_future_start_year", 5))
            break
        if isinstance(end_year, int) and end_year > REFERENCE_DATE.year + 1:
            findings.append(HoneypotFinding("education_future_end_year", 5))
            break

    for certification in candidate.get("certifications", []):
        year = certification.get("year")
        if isinstance(year, int) and year > REFERENCE_DATE.year + 1:
            findings.append(HoneypotFinding("certification_future_year", 5))
            break

    salary = candidate.get("redrob_signals", {}).get("expected_salary_range_inr_lpa") or {}
    min_salary = salary.get("min")
    max_salary = salary.get("max")
    if min_salary is not None and max_salary is not None and min_salary > max_salary:
        findings.append(HoneypotFinding("salary_min_greater_than_max", 1))

    return findings


def keyword_stuffing_findings(candidate: dict) -> list[HoneypotFinding]:
    title = normalize(current_title(candidate))
    non_tech_title = title in NEGATIVE_TITLES
    skill_text = " ".join(normalize(skill.get("name")) for skill in candidate.get("skills", []))
    ai_skill_hits = match_terms(skill_text, HARD_RELEVANCE_TERMS + RETRIEVAL_TERMS + RANKING_TERMS + REAL_ML_RETRIEVAL_TERMS)
    career_text = career_history_text(candidate)
    has_technical_career_evidence = has_any_term(career_text, CODING_TERMS + PRODUCT_OR_PRODUCTION_TERMS + REAL_ML_RETRIEVAL_TERMS)

    if non_tech_title and len(set(ai_skill_hits)) >= 8 and not has_technical_career_evidence:
        return [HoneypotFinding("non_tech_title_with_ai_keyword_stuffing", 5)]
    return []


def seniority_findings(candidate: dict) -> list[HoneypotFinding]:
    title = normalize(current_title(candidate))
    text = candidate_text(candidate)
    senior_title = has_any_term(title, SENIORITY_TRAP_TITLES)
    findings: list[HoneypotFinding] = []

    if senior_title and years_of_experience(candidate) < 3:
        findings.append(HoneypotFinding("senior_title_with_impossibly_low_experience", 3))

    if senior_title and not has_any_term(text, CODING_TERMS + PRODUCT_OR_PRODUCTION_TERMS + REAL_ML_RETRIEVAL_TERMS):
        findings.append(HoneypotFinding("senior_ai_profile_without_coding_tool_evidence", 3))

    return findings


def company_timeline_findings(candidate: dict) -> list[HoneypotFinding]:
    findings: list[HoneypotFinding] = []
    if not COMPANY_FOUNDING_YEAR:
        return findings

    for job in candidate.get("career_history", []):
        company = normalize(job.get("company"))
        founding_year = COMPANY_FOUNDING_YEAR.get(company)
        if not founding_year:
            continue
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) or REFERENCE_DATE
        if not start:
            continue
        company_age_months = max(0, (end.year - founding_year) * 12 + end.month)
        duration = int(job.get("duration_months") or 0)
        if duration > company_age_months + 12:
            findings.append(HoneypotFinding("role_duration_exceeds_possible_company_age", 5))
    return findings


def career_history_text(candidate: dict) -> str:
    cached = candidate.get("_career_history_text")
    if cached is not None:
        return cached
    parts = []
    for job in candidate.get("career_history", []):
        parts.extend(str(job.get(key) or "") for key in ["title", "industry", "description", "company"])
    text = " ".join(parts).lower()
    candidate["_career_history_text"] = text
    return text


def explicit_profile_experience_years(candidate: dict) -> list[float]:
    """Parse explicit "X years of experience" claims from headline/summary only."""
    profile = candidate.get("profile", {})
    text = " ".join(str(profile.get(key) or "") for key in ["headline", "summary"])
    matches = re.finditer(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?experience", text, re.IGNORECASE)
    return [float(match.group(1)) for match in matches]


def parse_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None

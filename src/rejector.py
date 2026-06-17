"""Hard reject layer before scoring.

These rules are intentionally narrow. They come from the JD section describing
ownership of ranking, retrieval, matching, and evaluation systems. Everything
else should be handled as a soft score/penalty rather than elimination.
"""

from __future__ import annotations

from datetime import datetime

from src.config import (
    CODING_TERMS,
    CORE_ASSESSMENT_SKILLS,
    GENAI_APP_ONLY_TERMS,
    HARD_RELEVANCE_TERMS,
    MANAGER_ARCHITECT_TERMS,
    PRODUCT_OR_PRODUCTION_TERMS,
    RANKING_TERMS,
    REAL_ML_RETRIEVAL_TERMS,
    REFERENCE_DATE,
    RESEARCH_ONLY_TERMS,
    RETRIEVAL_TERMS,
    ROLE_REQUIRES_HYBRID_OR_ONSITE,
)
from src.features import candidate_text, current_title, has_any_term, match_terms, normalize
from src.honeypot import honeypot_score


def hard_reject(candidate: dict) -> tuple[bool, str | None]:
    """Return whether candidate should be eliminated before scoring."""
    text = candidate_text(candidate)

    hp_score, hp_reasons = honeypot_score(candidate)
    if hp_score >= 5:
        return True, f"honeypot_{hp_reasons[0]}"

    jd_rejected, jd_reasons = jd_explicit_hard_reject(candidate, text)
    if jd_rejected:
        return True, jd_reasons[0]

    if not has_any_term(text, HARD_RELEVANCE_TERMS):
        return True, "no_core_ranking_retrieval_matching_ml_relevance"

    rejected_by_behavior, behavior_reason = redrob_signal_reject(candidate)
    if rejected_by_behavior:
        return True, behavior_reason

    return False, None


def jd_explicit_hard_reject(candidate: dict, text: str) -> tuple[bool, list[str]]:
    """Direct disqualifiers from the JD's explicit 5-9 years section."""
    reasons: list[str] = []

    if is_research_only_without_production(candidate, text):
        reasons.append("pure_research_without_production_deployment")

    if is_recent_langchain_openai_only(candidate, text):
        reasons.append("recent_langchain_openai_only_without_real_ml_production")

    if is_senior_architect_without_recent_code(candidate):
        reasons.append("senior_architect_or_tech_lead_without_recent_production_code")

    return bool(reasons), reasons


def is_research_only_without_production(candidate: dict, text: str) -> bool:
    has_research_background = has_any_term(text, RESEARCH_ONLY_TERMS)
    career_text = career_history_text(candidate)
    has_production_deployment = has_any_term(career_text, PRODUCT_OR_PRODUCTION_TERMS)
    has_real_ml_production = has_any_term(career_text, REAL_ML_RETRIEVAL_TERMS) and has_production_deployment
    return has_research_background and not has_production_deployment and not has_real_ml_production


def is_recent_langchain_openai_only(candidate: dict, text: str) -> bool:
    genai_signal_text = career_history_text(candidate) + " " + skills_text(candidate)
    has_genai_wrapper_signal = has_any_term(genai_signal_text, GENAI_APP_ONLY_TERMS)
    if not has_genai_wrapper_signal:
        return False

    career_text = career_history_text(candidate)
    has_real_ml_retrieval = has_any_term(career_text, REAL_ML_RETRIEVAL_TERMS)
    has_production_deployment = has_any_term(career_text, PRODUCT_OR_PRODUCTION_TERMS)
    if has_real_ml_retrieval and has_production_deployment:
        return False

    return max_genai_skill_duration(candidate) <= 12


def is_senior_architect_without_recent_code(candidate: dict) -> bool:
    recent_titles = [current_title(candidate)]
    recent_titles.extend(
        str(job.get("title") or "")
        for job in candidate.get("career_history", [])
        if is_recent_job(job)
    )
    senior_architect_or_manager = has_any_term(" ".join(recent_titles).lower(), MANAGER_ARCHITECT_TERMS)
    if not senior_architect_or_manager:
        return False

    return not has_recent_production_code_signal(candidate)


def has_recent_production_code_signal(candidate: dict) -> bool:
    for job in candidate.get("career_history", []):
        if not is_recent_job(job):
            continue
        job_text = " ".join(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        ).lower()
        has_coding = has_any_term(job_text, CODING_TERMS)
        has_production = has_any_term(job_text, PRODUCT_OR_PRODUCTION_TERMS)
        has_relevant_system = has_any_term(job_text, REAL_ML_RETRIEVAL_TERMS + RETRIEVAL_TERMS + RANKING_TERMS)
        if has_coding and (has_production or has_relevant_system):
            return True
    return False


def is_recent_job(job: dict) -> bool:
    if job.get("is_current"):
        return True
    end_date = parse_date(job.get("end_date"))
    if not end_date:
        return False
    return (REFERENCE_DATE - end_date).days <= 548


def max_genai_skill_duration(candidate: dict) -> int:
    durations = []
    for skill in candidate.get("skills", []):
        name = normalize(skill.get("name"))
        if match_terms(name, GENAI_APP_ONLY_TERMS):
            durations.append(int(skill.get("duration_months") or 0))
    return max(durations, default=0)


def career_history_text(candidate: dict) -> str:
    cached = candidate.get("_career_history_text")
    if cached is not None:
        return cached
    parts = []
    for job in candidate.get("career_history", []):
        parts.extend(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        )
    text = " ".join(parts).lower()
    candidate["_career_history_text"] = text
    return text


def skills_text(candidate: dict) -> str:
    cached = candidate.get("_skills_text")
    if cached is not None:
        return cached
    text = " ".join(str(skill.get("name") or "") for skill in candidate.get("skills", [])).lower()
    candidate["_skills_text"] = text
    return text


def redrob_signal_reject(candidate: dict) -> tuple[bool, str | None]:
    signals = candidate.get("redrob_signals", {})
    last_active = parse_date(signals.get("last_active_date"))
    days_inactive = (REFERENCE_DATE - last_active).days if last_active else None

    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    response_hours = float(signals.get("avg_response_time_hours") or 0.0)
    interview_rate = signals.get("interview_completion_rate")
    offer_rate = signals.get("offer_acceptance_rate")
    completeness = float(signals.get("profile_completeness_score") or 0.0)
    notice_days = int(signals.get("notice_period_days") or 0)
    open_to_work = bool(signals.get("open_to_work_flag"))
    preferred_work_mode = normalize(signals.get("preferred_work_mode"))
    willing_to_relocate = bool(signals.get("willing_to_relocate"))

    if response_rate <= 0.05 and response_hours > 168:
        return True, "non_responsive_recruiter_signal"

    if days_inactive is not None and days_inactive > 180 and response_rate <= 0.10:
        return True, "inactive_and_low_response"

    if interview_rate is not None and float(interview_rate) < 0.25:
        return True, "very_poor_interview_completion"

    if offer_rate is not None and float(offer_rate) == 0.0:
        return True, "zero_offer_acceptance_with_history"

    if (
        not bool(signals.get("verified_email"))
        and not bool(signals.get("verified_phone"))
        and not bool(signals.get("linkedin_connected"))
    ):
        return True, "no_verified_contact_channel"

    if completeness < 30:
        return True, "extremely_low_profile_completeness"

    rejected_by_assessment, assessment_reason = core_assessment_reject(signals)
    if rejected_by_assessment:
        return True, assessment_reason

    if ROLE_REQUIRES_HYBRID_OR_ONSITE and preferred_work_mode == "remote" and not willing_to_relocate:
        return True, "remote_only_not_relocatable_for_hybrid_role"

    if notice_days >= 150 and not open_to_work:
        return True, "long_notice_and_not_open_to_work"

    if completeness < 30 and not open_to_work and days_inactive is not None and days_inactive > 90:
        return True, "incomplete_inactive_not_open"

    return False, None


def core_assessment_reject(signals: dict) -> tuple[bool, str | None]:
    assessments = signals.get("skill_assessment_scores") or {}
    low_core_scores: list[float] = []

    for skill_name, score in assessments.items():
        normalized_skill = normalize(skill_name)
        if any(core in normalized_skill for core in CORE_ASSESSMENT_SKILLS):
            numeric_score = float(score)
            if numeric_score < 40:
                return True, "core_skill_assessment_below_40"
            if numeric_score < 50:
                low_core_scores.append(numeric_score)

    if len(low_core_scores) >= 2:
        return True, "multiple_core_skill_assessments_below_50"

    return False, None


def parse_date(value: object):
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None

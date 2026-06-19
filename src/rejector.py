"""Hard reject layer before scoring.

These rules are intentionally narrow. They come from the JD section describing
ownership of ranking, retrieval, matching, and evaluation systems. Everything
else should be handled as a soft score/penalty rather than elimination.
"""

from __future__ import annotations

from datetime import datetime

from src.config import (
    ADJACENT_TECHNICAL_ROLE_TERMS,
    AI_EXPLORER_ONLY_TERMS,
    AI_KEYWORD_TERMS,
    AI_PRODUCTIVITY_USAGE_TERMS,
    BUSINESS_NON_TARGET_TERMS,
    CLOSED_SOURCE_SYSTEM_TERMS,
    CODING_TERMS,
    CORE_ASSESSMENT_SKILLS,
    CV_SPEECH_ROBOTICS_TERMS,
    EXTERNAL_VALIDATION_TERMS,
    GENAI_APP_ONLY_TERMS,
    GENAI_EXPLORER_TERMS,
    HARD_RELEVANCE_TERMS,
    MANAGER_ARCHITECT_TERMS,
    NLP_IR_EXPOSURE_TERMS,
    NEGATIVE_TITLES,
    NON_TARGET_ROLE_TERMS,
    PRE_LLM_ML_PRODUCTION_TERMS,
    PRODUCT_OR_PRODUCTION_TERMS,
    RANKING_TERMS,
    REAL_AI_ENGINEERING_EVIDENCE_TERMS,
    REAL_AI_CAREER_ANCHOR_TERMS,
    REAL_AI_EVALUATION_TERMS,
    REAL_ML_RETRIEVAL_TERMS,
    REAL_CAREER_EVIDENCE_TERMS,
    REFERENCE_DATE,
    RESEARCH_ENVIRONMENT_TERMS,
    RESEARCH_ONLY_TERMS,
    RESEARCH_ROLE_TITLE_TERMS,
    RETRIEVAL_TERMS,
    ROLE_REQUIRES_HYBRID_OR_ONSITE,
    SENIOR_ENGINEER_ROLE_TERMS,
    STRONG_REAL_AI_CAREER_EVIDENCE_CATEGORIES,
)
from src.features import candidate_text, current_title, has_any_term, match_terms, normalize
from src.honeypot import honeypot_score


def hard_reject(candidate: dict) -> tuple[bool, str | None]:
    """Return whether candidate should be eliminated before scoring."""
    reasons = hard_reject_reasons(candidate)
    return bool(reasons), reasons[0] if reasons else None


def hard_reject_reasons(candidate: dict) -> list[str]:
    """Return ordered hard-reject reason codes for audit output."""
    hp_score, hp_reasons = honeypot_score(candidate)
    if hp_score >= 5:
        return [f"honeypot_{hp_reasons[0]}"]

    text = candidate_text(candidate)
    jd_rejected, jd_reasons = jd_explicit_hard_reject(candidate, text)
    if jd_rejected:
        return jd_reasons

    if not has_any_term(text, HARD_RELEVANCE_TERMS):
        return ["no_core_ranking_retrieval_matching_ml_relevance"]

    if is_closed_source_5y_without_external_validation(candidate):
        return ["closed_source_5y_without_external_validation"]

    rejected_by_behavior, behavior_reason = redrob_signal_reject(candidate)
    if rejected_by_behavior:
        return [behavior_reason or "unknown_behavior_reject"]

    return []


def jd_explicit_hard_reject(candidate: dict, text: str) -> tuple[bool, list[str]]:
    """Direct disqualifiers from the JD's explicit 5-9 years section."""
    reasons: list[str] = []

    if is_research_only_without_production(candidate, text):
        reasons.append("pure_research_without_production_deployment")

    if is_recent_langchain_openai_only(candidate, text):
        reasons.append("recent_langchain_openai_only_without_real_ml_production")

    if is_senior_architect_without_recent_code(candidate):
        reasons.append("senior_engineer_architect_or_tech_lead_without_recent_production_code")

    if is_cv_speech_robotics_without_nlp_ir(candidate):
        reasons.append("primary_cv_speech_robotics_without_significant_nlp_ir_exposure")

    fake_ai_rejected, fake_ai_reasons = detect_fake_ai_fit(candidate)
    if fake_ai_rejected:
        reasons.extend(fake_ai_reasons)

    return bool(reasons), reasons


def is_research_only_without_production(candidate: dict, text: str) -> bool:
    evidence_text = profile_and_career_text(candidate)
    if has_any_term(evidence_text, PRODUCT_OR_PRODUCTION_TERMS):
        return False

    career = candidate.get("career_history", [])
    if not career:
        return has_any_term(text, RESEARCH_ONLY_TERMS)

    research_roles = 0
    for job in career:
        title = normalize(job.get("title"))
        job_text = " ".join(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        ).lower()
        if has_any_term(title, RESEARCH_ROLE_TITLE_TERMS) or has_any_term(job_text, RESEARCH_ENVIRONMENT_TERMS):
            research_roles += 1

    if research_roles == 0:
        return False

    career_is_research_only = research_roles == len(career)
    career_is_mostly_research = research_roles / len(career) >= 0.67
    current_research_role = has_any_term(normalize(current_title(candidate)), RESEARCH_ROLE_TITLE_TERMS)
    return career_is_research_only or (career_is_mostly_research and current_research_role)


def is_recent_langchain_openai_only(candidate: dict, text: str) -> bool:
    genai_signal_text = career_history_text(candidate) + " " + skills_text(candidate)
    has_genai_wrapper_signal = has_any_term(genai_signal_text, GENAI_APP_ONLY_TERMS)
    if not has_genai_wrapper_signal:
        return False

    if max_genai_skill_duration(candidate) > 12:
        return False

    return not has_pre_llm_ml_production_evidence(candidate)


def has_pre_llm_ml_production_evidence(candidate: dict) -> bool:
    for job in candidate.get("career_history", []):
        job_text = " ".join(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        ).lower()
        has_ml_system = has_any_term(job_text, PRE_LLM_ML_PRODUCTION_TERMS + REAL_ML_RETRIEVAL_TERMS)
        has_production_deployment = has_any_term(job_text, PRODUCT_OR_PRODUCTION_TERMS)
        if has_ml_system and has_production_deployment:
            return True
    return False


def is_cv_speech_robotics_without_nlp_ir(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    profile_focus_text = " ".join(
        [
            str(profile.get("current_title") or ""),
            str(profile.get("headline") or ""),
            str(profile.get("current_industry") or ""),
        ]
    ).lower()
    recent_career_text = " ".join(
        " ".join(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        )
        for job in candidate.get("career_history", [])[:2]
    ).lower()
    career_text = career_history_text(candidate)
    skill_text = skills_text(candidate)

    profile_domain_hits = set(match_terms(profile_focus_text, CV_SPEECH_ROBOTICS_TERMS))
    recent_career_domain_hits = set(match_terms(recent_career_text, CV_SPEECH_ROBOTICS_TERMS))
    skill_domain_hits = set(match_terms(skill_text, CV_SPEECH_ROBOTICS_TERMS))

    nlp_ir_hits = set(
        match_terms(
            profile_focus_text + " " + career_text + " " + skill_text,
            NLP_IR_EXPOSURE_TERMS,
        )
    )
    has_significant_nlp_ir = len(nlp_ir_hits) >= 2

    primary_by_profile = bool(profile_domain_hits)
    primary_by_recent_career = (
        len(recent_career_domain_hits) >= 2
        and len(recent_career_domain_hits) > len(nlp_ir_hits)
    )
    primary_by_skills = (
        len(skill_domain_hits) >= 5
        and len(skill_domain_hits) >= len(nlp_ir_hits) + 3
    )

    primary_cv_speech_robotics = primary_by_profile or primary_by_recent_career or primary_by_skills
    return primary_cv_speech_robotics and not has_significant_nlp_ir


def detect_fake_ai_fit(candidate: dict) -> tuple[bool, list[str]]:
    """Detect fake AI-fit candidates before embeddings are prepared."""
    reasons: list[str] = []
    profile = candidate.get("profile", {})
    current_title_text = normalize(profile.get("current_title"))
    headline_text = normalize(profile.get("headline"))
    summary_text = normalize(profile.get("summary"))
    headline_summary_text = " ".join([headline_text, summary_text])
    career_title_text = career_titles_text(candidate)
    career_text = career_history_text(candidate)
    career_role_text = career_title_text + " " + career_text
    skill_text = skills_text(candidate)

    if has_strong_real_ai_career_evidence(candidate):
        add_soft_flag(candidate, "non_cs_background_but_strong_ai_career")
        return False, []

    if has_adjacent_technical_role(candidate):
        if not ai_keywords_supported_by_career(candidate):
            add_soft_flag(candidate, "adjacent_technical_no_retrieval_evidence")
        return False, []

    title_headline_career_titles = " ".join(
        [
            current_title_text,
            headline_text,
            career_title_text,
        ]
    )
    current_or_career_non_target = has_any_term(current_title_text + " " + career_title_text, NON_TARGET_ROLE_TERMS)
    has_non_target_role = has_any_term(title_headline_career_titles, NON_TARGET_ROLE_TERMS)
    has_genai_explorer_profile = has_any_term(headline_summary_text, GENAI_EXPLORER_TERMS)
    skill_ai_terms = set(match_terms(skill_text, AI_KEYWORD_TERMS))
    profile_ai_terms = set(match_terms(headline_summary_text, AI_KEYWORD_TERMS))
    career_ai_terms = set(match_terms(career_text, AI_KEYWORD_TERMS + REAL_AI_ENGINEERING_EVIDENCE_TERMS))
    has_career_evidence = ai_keywords_supported_by_career(candidate)
    ai_terms_only_summary_or_skills = bool(profile_ai_terms or skill_ai_terms) and not career_ai_terms
    business_career = career_is_mainly_business_or_operations(candidate)
    productivity_ai_usage = has_any_term(headline_summary_text + " " + career_text, AI_PRODUCTIVITY_USAGE_TERMS)
    wrapper_ai_context = has_any_term(headline_summary_text + " " + skill_text, AI_KEYWORD_TERMS + GENAI_EXPLORER_TERMS)
    years = float(profile.get("years_of_experience") or 0.0)

    if (
        has_non_target_role
        and has_genai_explorer_profile
        and len(skill_ai_terms) >= 3
        and not has_career_evidence
    ):
        reasons.append("fake_ai_fit_non_target_genai_explorer")

    if productivity_ai_usage and wrapper_ai_context and not has_career_evidence:
        reasons.append("ai_productivity_usage_not_ai_engineering")

    if len(skill_ai_terms) >= 5 and current_or_career_non_target and not has_career_evidence:
        reasons.append("ai_keywords_not_supported_by_career")

    if (
        years < 3
        and has_any_term(current_title_text, NON_TARGET_ROLE_TERMS)
        and ai_terms_only_summary_or_skills
        and not has_career_evidence
    ):
        reasons.append("junior_non_target_ai_keyword_profile")

    if (
        business_career
        and (has_genai_explorer_profile or productivity_ai_usage)
        and ai_terms_only_summary_or_skills
        and not has_career_evidence
    ):
        reasons.append("business_profile_ai_wrapper")

    return bool(reasons), reasons


def detect_fake_ai_fit_candidate(candidate: dict) -> tuple[bool, list[str]]:
    """Backward-compatible alias for older debug helpers."""
    return detect_fake_ai_fit(candidate)


def ai_keywords_supported_by_career(candidate: dict) -> bool:
    """Return true only when AI/retrieval/ranking evidence appears in career history."""
    career_text = career_history_text(candidate)
    return has_any_term(career_text, REAL_AI_CAREER_ANCHOR_TERMS)


def has_strong_real_ai_career_evidence(candidate: dict) -> bool:
    career_text = career_history_text(candidate)
    matched_categories = {
        category
        for category, terms in STRONG_REAL_AI_CAREER_EVIDENCE_CATEGORIES.items()
        if has_any_term(career_text, terms)
    }
    has_ai_anchor = has_any_term(career_text, REAL_AI_CAREER_ANCHOR_TERMS)
    has_eval_signal = has_any_term(career_text, REAL_AI_EVALUATION_TERMS)
    return has_ai_anchor and len(matched_categories) >= 2 and (
        has_eval_signal
        or "search_retrieval_ranking" in matched_categories
        or "recommendation_matching" in matched_categories
        or "production_ml_system_ownership" in matched_categories
    )


def career_is_mainly_business_or_operations(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    if not career:
        return False

    business_roles = 0
    for job in career:
        job_text = " ".join(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        ).lower()
        if has_any_term(job_text, BUSINESS_NON_TARGET_TERMS):
            business_roles += 1
    return business_roles / len(career) >= 0.60


def has_adjacent_technical_role(candidate: dict) -> bool:
    role_text = " ".join(
        [
            str(candidate.get("profile", {}).get("current_title") or ""),
            career_titles_text(candidate),
        ]
    ).lower()
    return has_any_term(role_text, ADJACENT_TECHNICAL_ROLE_TERMS)


def add_soft_flag(candidate: dict, flag: str) -> None:
    flags = candidate.setdefault("_soft_flags", [])
    if flag not in flags:
        flags.append(flag)


def is_non_tech_ai_keyword_profile_without_career_evidence(candidate: dict) -> bool:
    current = normalize(current_title(candidate))
    title_is_non_tech = has_any_term(current, NEGATIVE_TITLES)
    if not title_is_non_tech:
        return False

    career_text = career_history_text(candidate)
    career_ml_hits = match_terms(
        career_text,
        PRE_LLM_ML_PRODUCTION_TERMS + REAL_ML_RETRIEVAL_TERMS + RETRIEVAL_TERMS + RANKING_TERMS,
    )
    if career_ml_hits:
        return False

    profile = candidate.get("profile", {})
    profile_text = " ".join(
        [
            str(profile.get("headline") or ""),
            str(profile.get("summary") or ""),
            str(profile.get("current_industry") or ""),
        ]
    ).lower()
    skill_text = skills_text(candidate)
    profile_ai_hits = match_terms(profile_text, GENAI_APP_ONLY_TERMS + AI_EXPLORER_ONLY_TERMS + RETRIEVAL_TERMS)
    skill_ai_hits = match_terms(
        skill_text,
        GENAI_APP_ONLY_TERMS + REAL_ML_RETRIEVAL_TERMS + RETRIEVAL_TERMS + RANKING_TERMS,
    )

    has_keyword_density = len(set(profile_ai_hits + skill_ai_hits)) >= 4
    has_ai_explorer_language = has_any_term(profile_text, AI_EXPLORER_ONLY_TERMS)
    return has_keyword_density or has_ai_explorer_language


def is_closed_source_5y_without_external_validation(candidate: dict) -> bool:
    if float(candidate.get("profile", {}).get("years_of_experience") or 0.0) < 5.0:
        return False
    if has_external_validation(candidate):
        return False

    career = candidate.get("career_history", [])
    if not career:
        return False

    closed_source_months = 0
    total_role_months = 0
    for job in career:
        duration = int(job.get("duration_months") or 0)
        total_role_months += duration
        job_text = " ".join(
            str(job.get(key) or "")
            for key in ["title", "industry", "description", "company"]
        ).lower()
        if has_any_term(job_text, CLOSED_SOURCE_SYSTEM_TERMS):
            closed_source_months += duration

    if closed_source_months < 60:
        return False
    return total_role_months == 0 or closed_source_months >= total_role_months * 0.80


def has_external_validation(candidate: dict) -> bool:
    github_score = float(candidate.get("redrob_signals", {}).get("github_activity_score") or -1.0)
    if github_score > 0:
        return True

    profile = candidate.get("profile", {})
    text = " ".join(
        [
            str(profile.get("headline") or ""),
            str(profile.get("summary") or ""),
            career_history_text(candidate),
        ]
    ).lower()
    return has_any_term(text, EXTERNAL_VALIDATION_TERMS)


def is_senior_architect_without_recent_code(candidate: dict) -> bool:
    recent_titles = [current_title(candidate)]
    recent_titles.extend(
        str(job.get("title") or "")
        for job in candidate.get("career_history", [])
        if is_recent_job(job)
    )
    recent_title_text = " ".join(recent_titles).lower()
    senior_hands_on_role = has_any_term(
        recent_title_text,
        MANAGER_ARCHITECT_TERMS + SENIOR_ENGINEER_ROLE_TERMS,
    )
    if not senior_hands_on_role:
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


def career_titles_text(candidate: dict) -> str:
    cached = candidate.get("_career_titles_text")
    if cached is not None:
        return cached
    text = " ".join(str(job.get("title") or "") for job in candidate.get("career_history", [])).lower()
    candidate["_career_titles_text"] = text
    return text


def profile_and_career_text(candidate: dict) -> str:
    cached = candidate.get("_profile_and_career_text")
    if cached is not None:
        return cached
    profile = candidate.get("profile", {})
    parts = [
        str(profile.get("headline") or ""),
        str(profile.get("summary") or ""),
        str(profile.get("current_title") or ""),
        str(profile.get("current_industry") or ""),
        career_history_text(candidate),
    ]
    text = " ".join(parts).lower()
    candidate["_profile_and_career_text"] = text
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
    country = normalize(candidate.get("profile", {}).get("country"))

    rejected_by_notice, notice_reason = notice_based_hard_reject(
        notice_days=notice_days,
        open_to_work=open_to_work,
        response_rate=response_rate,
        days_inactive=days_inactive,
        interview_rate=interview_rate,
        preferred_work_mode=preferred_work_mode,
        willing_to_relocate=willing_to_relocate,
    )
    if rejected_by_notice:
        return True, notice_reason

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

    if (
        country
        and country != "india"
        and preferred_work_mode == "remote"
        and not willing_to_relocate
        and not open_to_work
        and notice_days > 60
    ):
        return True, "outside_india_remote_only_not_open_long_notice"

    if preferred_work_mode == "remote" and not willing_to_relocate:
        return True, "remote_only_not_relocatable_for_quarterly_travel"

    if country and country != "india" and not willing_to_relocate:
        return True, "outside_india_no_relocation_path_no_visa_sponsorship"

    if completeness < 30 and not open_to_work and days_inactive is not None and days_inactive > 90:
        return True, "incomplete_inactive_not_open"

    return False, None


def notice_based_hard_reject(
    notice_days: int,
    open_to_work: bool,
    response_rate: float,
    days_inactive: int | None,
    interview_rate: object,
    preferred_work_mode: str,
    willing_to_relocate: bool,
) -> tuple[bool, str | None]:
    """Reject only when long notice combines with low hireability signals."""
    if notice_days > 120 and not open_to_work:
        return True, "very_long_notice_and_not_open_to_work"

    if notice_days > 90 and response_rate < 0.20 and not open_to_work:
        return True, "long_notice_low_response_not_open"

    if notice_days > 90 and days_inactive is not None and days_inactive > 120 and response_rate < 0.25:
        return True, "long_notice_inactive_low_response"

    if interview_rate is not None:
        numeric_interview_rate = float(interview_rate)
        if notice_days > 90 and 0 <= numeric_interview_rate < 0.30:
            return True, "long_notice_poor_interview_completion"

    if (
        ROLE_REQUIRES_HYBRID_OR_ONSITE
        and notice_days > 60
        and preferred_work_mode == "remote"
        and not willing_to_relocate
    ):
        return True, "long_notice_remote_only_not_relocating"

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

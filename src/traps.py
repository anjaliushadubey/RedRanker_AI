"""Trap and keyword-stuffing checks."""

from __future__ import annotations

from src.config import NEGATIVE_TITLES, RETRIEVAL_TERMS
from src.features import candidate_text, normalize, product_company_fit, title_fit
from src.honeypot import honeypot_score


def trap_penalty(candidate: dict) -> tuple[float, list[str]]:
    notes: list[str] = []
    penalty = 0.0

    hp_score, hp_reasons = honeypot_score(candidate)
    if 3 <= hp_score < 5:
        penalty += 0.25
        notes.append(f"honeypot risk: {hp_reasons[0]}")
    elif 1 <= hp_score < 3:
        penalty += 0.08
        notes.append(f"minor profile consistency risk: {hp_reasons[0]}")

    expert_zero_duration = [
        skill
        for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert" and (skill.get("duration_months") or 0) == 0
    ]
    if len(expert_zero_duration) >= 3:
        penalty += 0.10
        notes.append("possible skill-duration inconsistency")

    text = candidate_text(candidate)
    retrieval_hits = sum(1 for term in RETRIEVAL_TERMS if term in text)
    current_title = normalize(candidate.get("profile", {}).get("current_title"))
    negative_title = any(current_title == title for title in NEGATIVE_TITLES)
    if retrieval_hits >= 5 and negative_title and title_fit(candidate) < 0.25:
        penalty += 0.20
        notes.append("AI keyword stuffing risk")

    _, _, service_only = product_company_fit(candidate)
    if service_only:
        penalty += 0.06
        notes.append("consulting-only background")

    years = float(candidate.get("profile", {}).get("years_of_experience") or 0.0)
    if years < 3 and retrieval_hits >= 6:
        penalty += 0.08
        notes.append("seniority and keyword density mismatch")

    return penalty, notes

"""Small baseline feature functions."""

from __future__ import annotations

from src.config import IDEAL_EXPERIENCE_YEARS, MAX_EXPERIENCE_SCORE_YEARS, PYTHON_SKILL_NAMES


def years_of_experience(candidate: dict) -> float:
    return float(candidate.get("profile", {}).get("years_of_experience") or 0.0)


def experience_score(candidate: dict) -> float:
    """Prefer candidates near senior IC range while keeping the signal simple."""
    years = years_of_experience(candidate)
    if years <= 0:
        return 0.0
    distance = abs(years - IDEAL_EXPERIENCE_YEARS)
    return max(0.0, 1.0 - (distance / MAX_EXPERIENCE_SCORE_YEARS))


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
    return 0.0

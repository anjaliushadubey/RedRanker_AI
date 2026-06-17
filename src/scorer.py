"""Baseline scoring logic."""

from __future__ import annotations

from src.features import experience_score, python_skill_score, years_of_experience


def score_candidate(candidate: dict) -> tuple[float, dict]:
    exp_score = experience_score(candidate)
    py_score = python_skill_score(candidate)
    score = 0.65 * exp_score + 0.35 * py_score

    details = {
        "years_of_experience": years_of_experience(candidate),
        "experience_score": exp_score,
        "python_skill_score": py_score,
    }
    return score, details

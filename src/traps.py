"""Basic trap checks.

This file starts intentionally conservative. Later iterations can add stronger
honeypot and keyword-stuffing detection without changing the pipeline shape.
"""

from __future__ import annotations


def trap_penalty(candidate: dict) -> tuple[float, list[str]]:
    notes: list[str] = []
    penalty = 0.0

    expert_zero_duration = [
        skill
        for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert" and (skill.get("duration_months") or 0) == 0
    ]
    if len(expert_zero_duration) >= 3:
        penalty += 0.10
        notes.append("possible skill-duration inconsistency")

    return penalty, notes

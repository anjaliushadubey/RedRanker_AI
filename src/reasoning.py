"""Reasoning text for submission rows."""

from __future__ import annotations


def build_reasoning(candidate: dict, score_details: dict, trap_notes: list[str]) -> str:
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Candidate")
    years = score_details["years_of_experience"]
    py_score = score_details["python_skill_score"]
    exp_score = score_details["experience_score"]

    strengths = [f"{years:.1f} years of experience"]
    if py_score > 0:
        strengths.append("has Python skill signal")
    else:
        strengths.append("no explicit Python skill found")

    concern = ""
    if trap_notes:
        concern = f" Concern: {trap_notes[0]}."

    return (
        f"{title} with {', '.join(strengths)}; baseline score uses experience "
        f"fit ({exp_score:.2f}) and Python fit ({py_score:.2f}).{concern}"
    )

"""Reasoning text for submission rows."""

from __future__ import annotations


def build_reasoning(candidate: dict, score_details: dict, trap_notes: list[str]) -> str:
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Candidate")
    years = score_details["years_of_experience"]
    scores = score_details["feature_scores"]
    behavior = score_details["behavior"]

    evidence = []
    if score_details["retrieval_terms"]:
        evidence.append("retrieval/search evidence: " + ", ".join(score_details["retrieval_terms"][:2]))
    if score_details["ranking_terms"]:
        evidence.append("ranking/matching evidence: " + ", ".join(score_details["ranking_terms"][:2]))
    if score_details["production_terms"]:
        evidence.append("production evidence: " + ", ".join(score_details["production_terms"][:2]))
    ideal = score_details.get("ideal_recruiter", {})
    if ideal.get("shipped_system_terms"):
        evidence.append("shipped system: " + ", ".join(ideal["shipped_system_terms"][:2]))
    if score_details.get("culture_terms"):
        evidence.append("culture signal: " + ", ".join(score_details["culture_terms"][:2]))
    if scores["python_fit"] > 0:
        evidence.append("Python signal present")
    if not evidence:
        evidence.append("limited direct retrieval evidence")

    concern = ""
    if trap_notes:
        concern = f" Concern: {trap_notes[0]}."
    elif behavior["notice_days"] >= 90:
        concern = f" Concern: {int(behavior['notice_days'])} day notice period."

    culture = ""
    if score_details.get("culture_terms"):
        culture = " Culture: " + ", ".join(score_details["culture_terms"][:2]) + "."
    ideal_note = ""
    if ideal.get("systems_judgment_terms"):
        ideal_note = " Judgment: " + ", ".join(ideal["systems_judgment_terms"][:2]) + "."

    return (
        f"{title} with {years:.1f} years; {', '.join(evidence[:2])}. "
        f"Behavior: response rate {behavior['response_rate']:.2f}, "
        f"interview completion {behavior['interview_rate']:.2f}.{culture}{ideal_note}{concern}"
    )

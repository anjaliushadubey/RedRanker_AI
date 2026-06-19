#!/usr/bin/env python3
"""Debug checks for fake-AI-fit hard rejection rules."""

from __future__ import annotations

from src.rejector import detect_fake_ai_fit


def candidate(
    candidate_id: str,
    title: str,
    headline: str,
    summary: str,
    career_title: str,
    career_description: str,
    skills: list[str],
    years: float = 5.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "current_title": title,
            "headline": headline,
            "summary": summary,
            "years_of_experience": years,
            "current_industry": "",
        },
        "career_history": [
            {
                "title": career_title,
                "industry": "",
                "description": career_description,
                "company": "ExampleCo",
            }
        ],
        "skills": [{"name": skill, "proficiency": "intermediate", "duration_months": 12} for skill in skills],
    }


def assert_reject(example: dict, expected_reason: str) -> None:
    rejected, reasons = detect_fake_ai_fit(example)
    assert rejected, f"Expected {example['candidate_id']} to be rejected"
    assert expected_reason in reasons, f"Expected {expected_reason}, got {reasons}"


def assert_not_reject_with_soft_flag(example: dict, expected_flag: str) -> None:
    rejected, reasons = detect_fake_ai_fit(example)
    assert not rejected, f"Expected {example['candidate_id']} to stay eligible, got {reasons}"
    assert expected_flag in example.get("_soft_flags", []), f"Expected soft flag {expected_flag}"


def main() -> None:
    strong_non_cs = candidate(
        "debug_mech_strong_ai_career",
        "Mechanical Engineer",
        "Mechanical background, search relevance engineer",
        "Started in mechanical engineering but now owns ranking and retrieval systems.",
        "Search Engineer",
        "Built search relevance and vector search systems with BM25 and Elasticsearch. "
        "Owned production ML model serving with latency monitoring and scale.",
        ["Python", "BM25", "Elasticsearch", "Vector Search", "NDCG"],
        years=6.0,
    )
    rejected, reasons = detect_fake_ai_fit(strong_non_cs)
    assert not rejected, f"Strong AI career evidence should override background, got {reasons}"
    assert "non_cs_background_but_strong_ai_career" in strong_non_cs.get("_soft_flags", [])

    assert_reject(
        candidate(
            "debug_civil_ai_enthusiast",
            "Civil Engineer",
            "Civil Engineer | AI enthusiast | Building with LLMs",
            "Exploring GenAI side projects with OpenAI API and curious about AI tools.",
            "Content Support Specialist",
            "Worked on customer support, content updates, brand design, and SEO workflows.",
            ["LangChain", "RAG", "Vector Search", "FAISS", "Pinecone"],
            years=2.0,
        ),
        "fake_ai_fit_non_target_genai_explorer",
    )
    assert_reject(
        candidate(
            "debug_mech_genai",
            "Mechanical Engineer",
            "Mechanical Engineer | GenAI explorer",
            "Recently excited about AI, taking online courses and experimenting with LangChain side projects.",
            "Mechanical Engineer",
            "Managed warehouse fulfillment operations and logistics KPIs.",
            ["LangChain", "RAG", "Vector Search", "Pinecone"],
            years=1.5,
        ),
        "fake_ai_fit_non_target_genai_explorer",
    )
    assert_reject(
        candidate(
            "debug_ba_courses",
            "Business Analyst",
            "Business analyst learning modern ML",
            "Online courses in GenAI and interested in transitioning into AI roles.",
            "Business Analyst",
            "Prepared stakeholder dashboards and business requirement documents.",
            ["OpenAI", "LLMs", "Embeddings", "Qdrant"],
        ),
        "fake_ai_fit_non_target_genai_explorer",
    )
    assert_not_reject_with_soft_flag(
        candidate(
            "debug_analytics_engineer",
            "Analytics Engineer",
            "Analytics engineer with data platform exposure",
            "Built analytics datasets and has small ML exposure.",
            "Analytics Engineer",
            "Built Spark, Kafka, Airflow, and dbt data pipelines for reporting.",
            ["Spark", "Kafka", "Airflow", "Python", "Machine Learning"],
        ),
        "adjacent_technical_no_retrieval_evidence",
    )
    assert_reject(
        candidate(
            "debug_marketing_chatgpt",
            "Marketing Manager",
            "Marketing manager using ChatGPT",
            "Uses ChatGPT and LLM tools for content creation, drafting, editing, SEO, and productivity.",
            "Content Marketing Manager",
            "Owned editorial calendar, content writing, SEO, and campaign reporting.",
            ["ChatGPT", "Prompt Engineering", "Generative AI"],
        ),
        "ai_productivity_usage_not_ai_engineering",
    )
    assert_reject(
        candidate(
            "debug_business_wrapper",
            "Project Manager",
            "Operations leader exploring AI",
            "AI appears in my work through productivity, content creation, and side projects.",
            "Operations Manager",
            "Managed logistics, warehouse fulfillment, sales coordination, and project management.",
            ["OpenAI", "ChatGPT", "LLMs", "Prompt Engineering", "Generative AI"],
        ),
        "business_profile_ai_wrapper",
    )
    print("fake AI-fit debug checks passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Streamlit demo for ranking a small RedRanker candidate sample."""

from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rerank_top100 import (
    apply_final_score_calibration,
    apply_jd_alignment_adjustment,
    apply_top10_not_open_guardrail,
    apply_top10_repeated_evidence_guardrail,
    apply_top5_template_guardrail,
    build_evidence_realism_index,
    build_ranked_reasoning,
    ranked_output_rows,
    rerank_row,
)
from src.rejector import hard_reject
from src.scorer import score_candidate
from src.traps import trap_penalty


APP_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = APP_DIR / "sample_candidates.jsonl"
OUTPUT_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "open"}


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def split_skills(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        skills = value
    else:
        skills = [
            item.strip()
            for item in str(value).replace(";", ",").split(",")
            if item.strip()
        ]
    normalized = []
    for skill in skills:
        if isinstance(skill, dict):
            normalized.append(skill)
        else:
            normalized.append(
                {
                    "name": str(skill),
                    "proficiency": "intermediate",
                    "duration_months": 0,
                }
            )
    return normalized


def generated_candidate_id(index: int) -> str:
    return f"CAND_{9000000 + index:07d}"


def csv_row_to_candidate(row: dict[str, Any], index: int) -> dict[str, Any]:
    years = as_float(row.get("years_of_experience") or row.get("experience_years"), 0.0)
    candidate_id = str(row.get("candidate_id") or generated_candidate_id(index))
    current_title = str(row.get("current_title") or row.get("title") or "")
    summary = str(row.get("summary") or row.get("profile_summary") or "")
    headline = str(row.get("headline") or current_title)
    career_description = str(
        row.get("career_history")
        or row.get("career_description")
        or row.get("description")
        or summary
    )

    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": str(row.get("anonymized_name") or row.get("name") or ""),
            "headline": headline,
            "summary": summary,
            "location": str(row.get("location") or ""),
            "country": str(row.get("country") or "India"),
            "years_of_experience": years,
            "current_title": current_title,
            "current_company": str(row.get("current_company") or row.get("company") or ""),
            "current_company_size": str(row.get("current_company_size") or ""),
            "current_industry": str(row.get("current_industry") or row.get("industry") or ""),
        },
        "career_history": [
            {
                "company": str(row.get("current_company") or row.get("company") or ""),
                "title": current_title,
                "start_date": str(row.get("start_date") or ""),
                "end_date": None,
                "duration_months": int(max(years * 12, 0)),
                "is_current": True,
                "industry": str(row.get("current_industry") or row.get("industry") or ""),
                "company_size": str(row.get("current_company_size") or ""),
                "description": career_description,
            }
        ],
        "education": [],
        "skills": split_skills(row.get("skills") or row.get("skills_text")),
        "redrob_signals": {
            "open_to_work_flag": as_bool(row.get("open_to_work_flag"), True),
            "notice_period_days": as_float(row.get("notice_period_days"), 30),
            "recruiter_response_rate": as_float(row.get("recruiter_response_rate"), 0.70),
            "avg_response_time_hours": as_float(row.get("avg_response_time_hours"), 24),
            "interview_completion_rate": as_float(row.get("interview_completion_rate"), 0.80),
            "offer_acceptance_rate": as_float(row.get("offer_acceptance_rate"), -1),
            "verified_email": as_bool(row.get("verified_email"), True),
            "verified_phone": as_bool(row.get("verified_phone"), True),
            "linkedin_connected": as_bool(row.get("linkedin_connected"), True),
            "willing_to_relocate": as_bool(row.get("willing_to_relocate"), True),
            "preferred_work_mode": str(row.get("preferred_work_mode") or "hybrid"),
            "github_activity_score": as_float(row.get("github_activity_score"), -1),
            "profile_completeness_score": as_float(row.get("profile_completeness_score"), 80),
        },
    }


def parse_candidates_text(text: str, filename: str = "uploaded.jsonl") -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    text = text.strip()
    if not text:
        raise ValueError("Uploaded file is empty.")

    if suffix == ".csv":
        frame = pd.read_csv(io.StringIO(text))
        return [
            csv_row_to_candidate(row, index + 1)
            for index, row in enumerate(frame.fillna("").to_dict(orient="records"))
        ]

    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON upload must contain a list of candidate records.")
        return payload

    candidates = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
    return candidates


def load_sample_candidates() -> list[dict[str, Any]]:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            "Bundled sample_candidates.jsonl was not found. "
            "Upload a small JSONL/JSON/CSV file or commit sample_candidates.jsonl to the repo."
        )
    return parse_candidates_text(SAMPLE_PATH.read_text(encoding="utf-8"), SAMPLE_PATH.name)


def run_demo_ranking(candidates: list[dict[str, Any]], top_n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rejected_rows = []
    for index, candidate in enumerate(candidates, start=1):
        candidate.setdefault("candidate_id", generated_candidate_id(index))
        rejected, reason = hard_reject(candidate)
        if rejected:
            rejected_rows.append(
                {
                    "candidate_id": candidate.get("candidate_id", ""),
                    "reason": reason or "hard_rejected",
                }
            )
            continue

        base_score, _ = score_candidate(candidate)
        penalty, _ = trap_penalty(candidate)
        hybrid_score = max(0.0, min(1.0, base_score - penalty))
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "rank": index,
                "hybrid_score": hybrid_score,
                "candidate": candidate,
            }
        )

    if not rows:
        raise ValueError("All uploaded candidates were hard rejected. Try a sample with at least one technical AI/search profile.")

    realism_index = build_evidence_realism_index(rows)
    results = [rerank_row(row, realism_index) for row in rows]
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))
    apply_top5_template_guardrail(results)
    apply_final_score_calibration(results)
    apply_jd_alignment_adjustment(results)
    apply_top10_repeated_evidence_guardrail(results)
    apply_top10_not_open_guardrail(results)
    results.sort(key=lambda item: (-item.final_score, item.candidate_id))

    selected = results[: min(top_n, len(results))]
    for rank, result in enumerate(selected, start=1):
        result.reasoning = build_ranked_reasoning(result, rank)

    output_rows = []
    for rank, result, score in ranked_output_rows(selected):
        output_rows.append(
            {
                "candidate_id": result.candidate_id,
                "rank": rank,
                "score": f"{score:.6f}",
                "reasoning": result.reasoning,
            }
        )

    return pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS), pd.DataFrame(rejected_rows)


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, quoting=csv.QUOTE_MINIMAL)
    return buffer.getvalue().encode("utf-8")


def render_scrollable_results(frame: pd.DataFrame) -> None:
    """Render results as a wide table with horizontal scrolling."""
    import streamlit.components.v1 as components

    rows_html = []
    for row in frame.to_dict(orient="records"):
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['candidate_id']))}</td>"
            f"<td>{html.escape(str(row['rank']))}</td>"
            f"<td>{html.escape(str(row['score']))}</td>"
            f"<td class=\"reasoning-cell\">{html.escape(str(row['reasoning']))}</td>"
            "</tr>"
        )

    table_html = "\n".join(rows_html)
    height = min(720, max(180, 72 + 46 * len(frame)))
    components.html(
        f"""<!doctype html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
          body {{
            margin: 0;
            font-family: "Source Sans Pro", sans-serif;
            color: #111827;
          }}
          .scroll-table-wrap {{
            width: 100%;
            overflow-x: auto;
            overflow-y: auto;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background: #ffffff;
          }}
          .scroll-table {{
            border-collapse: collapse;
            min-width: 1900px;
            width: 1900px;
            font-size: 0.94rem;
          }}
          .scroll-table th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: #f9fafb;
            color: #374151;
            text-align: left;
            font-weight: 600;
            border-bottom: 1px solid #e5e7eb;
            padding: 10px 12px;
            white-space: nowrap;
          }}
          .scroll-table td {{
            border-top: 1px solid #eef0f3;
            padding: 10px 12px;
            white-space: nowrap;
            vertical-align: top;
            color: #111827;
          }}
          .scroll-table th:nth-child(1),
          .scroll-table td:nth-child(1) {{
            width: 170px;
          }}
          .scroll-table th:nth-child(2),
          .scroll-table td:nth-child(2) {{
            width: 80px;
            text-align: right;
          }}
          .scroll-table th:nth-child(3),
          .scroll-table td:nth-child(3) {{
            width: 120px;
          }}
          .scroll-table .reasoning-cell {{
            min-width: 1450px;
            max-width: none;
          }}
        </style>
        </head>
        <body>
        <div class="scroll-table-wrap">
          <table class="scroll-table">
            <thead>
              <tr>
                <th>candidate_id</th>
                <th>rank</th>
                <th>score</th>
                <th>reasoning</th>
              </tr>
            </thead>
            <tbody>
              {table_html}
            </tbody>
          </table>
        </div>
        </body>
        </html>
        """,
        height=height,
        scrolling=True,
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="RedRanker AI Demo", layout="wide")
    st.title("RedRanker AI Candidate Ranking Demo")
    st.caption("Small-sample sandbox demo. The full 100K candidate pipeline is unchanged.")

    uploaded = st.file_uploader("Upload a small candidate file", type=["jsonl", "json", "csv"])
    sample_available = SAMPLE_PATH.exists()
    use_sample = st.checkbox(
        "Use included sample_candidates.jsonl",
        value=uploaded is None and sample_available,
        disabled=not sample_available,
    )
    top_n = st.slider("Rows to output", min_value=1, max_value=100, value=20)

    if uploaded is not None:
        st.info(f"Uploaded `{uploaded.name}` ({uploaded.size:,} bytes).")
    elif use_sample:
        st.info(f"Using bundled `{SAMPLE_PATH.name}`.")
    elif not sample_available:
        st.warning("Bundled sample file is missing. Upload a candidate file or push `sample_candidates.jsonl` to GitHub.")
    else:
        st.warning("Upload a file or enable the bundled sample.")

    if st.button("Run demo ranking", type="primary"):
        try:
            with st.spinner("Running hard rejection, scoring, and final reranking..."):
                if uploaded is not None:
                    text = uploaded.getvalue().decode("utf-8-sig")
                    candidates = parse_candidates_text(text, uploaded.name)
                elif use_sample:
                    candidates = load_sample_candidates()
                else:
                    raise ValueError("No candidate file selected.")

                ranked, rejected = run_demo_ranking(candidates, top_n=top_n)

            st.success(f"Ranked {len(ranked)} candidates from {len(candidates)} uploaded records.")
            if not rejected.empty:
                with st.expander(f"Hard-rejected candidates ({len(rejected)})"):
                    st.dataframe(rejected, use_container_width=True)

            st.subheader("Ranked Results")
            render_scrollable_results(ranked)
            st.download_button(
                label="Download submission.csv",
                data=dataframe_to_csv_bytes(ranked),
                file_name="submission.csv",
                mime="text/csv",
            )
        except Exception as exc:  # Streamlit should show friendly pipeline errors.
            st.error(str(exc))


if __name__ == "__main__":
    main()

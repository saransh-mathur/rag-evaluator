"""Streamlit dashboard for RAG evaluation results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def load_run_dir(run_dir: Path) -> None:
    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.csv"
    type_path = run_dir / "question_type_summary.csv"
    halluc_path = run_dir / "hallucination_examples.json"

    if not summary_path.exists():
        st.error(f"No summary found at {summary_path}. Run ./run_eval.sh first.")
        return

    summary = read_json(summary_path)
    results = pd.read_csv(results_path)
    type_summary = pd.read_csv(type_path) if type_path.exists() else pd.DataFrame()
    hallucinations = read_json(halluc_path) if halluc_path.exists() else []

    st.title("RAG Evaluation Dashboard")
    st.caption(f"Run directory: `{run_dir}`")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Questions", summary["total_questions"])
    c2.metric("Retrieval Hit Rate", f"{summary['retrieval_hit_rate']:.1%}")
    c3.metric("Answer Hit Rate", f"{summary['answer_hit_rate']:.1%}")
    c4.metric("Hallucination Rate", f"{summary['hallucination_rate']:.1%}")
    c5.metric("Avg Latency", f"{summary.get('avg_latency_seconds', 0.0):.2f}s")

    st.caption(f"**Avg tokens per query:** {summary.get('avg_total_tokens', 0):.0f} (Prompt: {summary.get('avg_prompt_tokens', 0):.0f} | Completion: {summary.get('avg_completion_tokens', 0):.0f})")

    with st.expander("Run configuration"):
        st.json(summary)

    st.subheader("Similarity distribution")
    fig = px.histogram(
        results,
        x="top_similarity",
        nbins=20,
        title="Top retrieval similarity scores",
    )
    st.plotly_chart(fig, use_container_width=True)

    if not type_summary.empty:
        st.subheader("Performance by question type")
        fig2 = px.bar(
            type_summary,
            x="question_type",
            y=["retrieval_hit_rate", "answer_hit_rate", "hallucination_rate"],
            barmode="group",
            title="Hit rates by question type",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(type_summary, use_container_width=True)

    st.subheader("Weak questions (missed retrieval or answer)")
    weak = results[(results["retrieval_hit"] == 0) | (results["answer_hit"] == 0)]
    st.dataframe(
        weak[["id", "question_type", "question", "retrieval_hit", "answer_hit", "top_similarity"]],
        use_container_width=True,
    )

    st.subheader("All results")
    st.dataframe(results, use_container_width=True)

    if hallucinations:
        st.subheader("Hallucination examples")
        for item in hallucinations:
            with st.expander(f"{item['id']}: {item['question'][:80]}"):
                st.write("**Expected phrases:**", item.get("expected_answer_contains", []))
                st.write("**Generated answer:**")
                st.write(item.get("generated_answer", ""))


def main() -> None:
    args = parse_args()
    default_run = Path(__file__).resolve().parent.parent / "runs" / "latest"
    run_dir = args.run_dir or default_run
    load_run_dir(run_dir)


if __name__ == "__main__":
    main()

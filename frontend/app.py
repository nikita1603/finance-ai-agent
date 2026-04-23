"""Streamlit frontend for the Stocks AI Agent."""

from datetime import date
import re
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stocks AI Agent",
    page_icon="📈",
    layout="wide",
)

EXAMPLE_QUESTIONS = [
    "What was the net profit for Q3 FY25-26?",
    "Summarize recent news.",
    "What is the current P/E ratio?",
    "What were the key revenue drivers this quarter?",
    "What is the stock price on 2026-04-23 and explain the movement?",
]

TOOL_LABELS = {
    "rag_tool": "📄 RAG (Documents)",
    "get_gnews_articles": "📰 News",
    "historical_price_tool": "📈 Historical Price",
    "fundamental_tool": "🔢 Fundamentals",
}

def parse_answer(raw: str) -> tuple[str, list[str]]:
    """Strip RAG markers from answer and extract source file names."""
    # Extract sources
    sources = []
    match = re.search(r"\[SOURCES_USED:\s*([^\]]+)\]", raw)
    if match:
        sources = [s.strip() for s in match.group(1).split(";") if s.strip()]

    # Strip [SOURCES_USED: ...] marker appended by rag_tool
    clean = re.sub(r"\[SOURCES_USED:[^\]]*\]", "", raw)
    return clean.strip(), sources


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stocks AI Agent")
    st.caption("Powered by Gemini · LlamaIndex")
    st.divider()

    selected_date = st.date_input("Date", value=date.today())

    company = st.selectbox("Company", ["HDFC", "RELIANCE"])

    financial_year = st.selectbox("Financial Year", ["2025-26", "2024-25", "2023-24"])

    quarter = st.selectbox(
        "Quarter",
        ["None", "Q1", "Q2", "Q3", "Q4"],
        index=0,
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("💬 Ask about your stock")

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# Example question buttons
st.caption("Try an example:")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, question in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(question, use_container_width=True):
        st.session_state.query_text = question
        st.rerun()

query = st.text_area(
    "Ask a question about the selected stock:",
    key="query_text",
    height=100,
)

if st.button("Ask", type="primary"):
    if query:
        final_query = (
            f"Date: {selected_date}\n"
            f"Company: {company}\n"
            f"Financial Year: {financial_year}\n"
            f"Quarter: {quarter}\n"
            f"Question: {query}"
        )

        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"query": final_query},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                raw_answer = data.get("answer", "No answer returned.")
                tools_used = data.get("tools_used", [])
            except requests.exceptions.ConnectionError:
                raw_answer = "⚠️ Could not reach the backend. Make sure the server is running on port 8000."
                tools_used = []
            except requests.exceptions.Timeout:
                raw_answer = "⚠️ The request timed out. The agent may be overloaded — please try again."
                tools_used = []
            except Exception as exc:
                raw_answer = f"⚠️ Unexpected error: {exc}"
                tools_used = []

        answer, sources = parse_answer(raw_answer)

        # Tools used badges
        if tools_used:
            badge_str = " &nbsp; ".join(
                f"`{TOOL_LABELS.get(t, t)}`" for t in tools_used
            )
            st.markdown(f"**Tools used:** {badge_str}")

        # Answer
        st.markdown("---")
        st.markdown(answer)

        # Source citations
        if sources:
            with st.expander("📎 Sources"):
                for src in sources:
                    st.markdown(f"- {src}")
    else:
        st.warning("Please enter a question.")

"""Streamlit frontend for the Stocks AI Agent."""

from datetime import date
import requests
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stocks AI Agent",
    page_icon="📈",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": ..., "content": ...}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stocks AI Agent")
    st.caption("Powered by Gemini · LlamaIndex")
    st.divider()

    selected_date = st.date_input("Date", value=date.today())

    company = st.selectbox(
        "Company",
        ["HDFC", "RELIANCE", "TCS", "INFOSYS", "WIPRO"],
    )

    financial_year = st.selectbox(
        "Financial Year",
        ["2025-2026", "2024-2025", "2023-2024"],
    )

    quarter = st.selectbox(
        "Quarter",
        ["None", "Q1", "Q2", "Q3", "Q4"],
        index=0,
    )

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("💬 Ask about your stock")

# Render existing chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input at the bottom
query = st.chat_input("Ask a question about the selected stock…")

if query:
    # Show the user's message immediately
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Build the structured prompt
    final_query = (
        f"Date: {selected_date}\n"
        f"Company: {company}\n"
        f"Financial Year: {financial_year}\n"
        f"Quarter: {quarter}\n"
        f"Question: {query}"
    )

    # Call the backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"query": final_query},
                    timeout=120,
                )
                resp.raise_for_status()
                answer = resp.json().get("answer", "No answer returned.")
            except requests.exceptions.ConnectionError:
                answer = "⚠️ Could not reach the backend. Make sure the server is running on port 8000."
            except requests.exceptions.Timeout:
                answer = "⚠️ The request timed out. The agent may be overloaded — please try again."
            except Exception as exc:
                answer = f"⚠️ Unexpected error: {exc}"

        st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})

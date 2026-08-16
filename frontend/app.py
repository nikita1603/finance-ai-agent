"""Streamlit frontend for the Stocks AI Agent."""

import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stocks AI Agent",
    page_icon="📈",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BACKEND_URL = "http://127.0.0.1:8000/ask"

TICKER_MAP = {
    "HDFC": "HDFCBANK.NS",
    "RELIANCE": "RELIANCE.NS",
}

PERIOD_DAYS = {
    "1M": 30,
    "3M": 90,
    "1Y": 365,
    "5Y": 1825,
    "All": 3650,
}

TOOL_LABELS = {
    "rag_tool": "📄 RAG (Documents)",
    "get_gnews_articles": "📰 News",
    "historical_price_tool": "📈 Historical Price",
    "fundamental_tool": "🔢 Fundamentals",
    "quant_analytics_tool": "📊 Quant Analytics",
}

EXAMPLE_QUESTIONS = [
    "What was the net profit for Q3 FY25-26?",
    "What is the Sharpe ratio over the last year?",
    "What is the current P/E ratio?",
    "Summarize recent news.",
    "What is the max drawdown over the last 6 months?",
    "What were the key revenue drivers this quarter?",
]

GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank\s+you|good\s+morning|good\s+afternoon|good\s+evening)\b",
    re.IGNORECASE,
)

GREETING_REPLY = (
    "Hello! I'm your Stocks AI Agent. Ask me anything about "
    "**HDFC Bank** or **Reliance** — financials, price history, "
    "news, valuation ratios, or risk analytics."
)

# Chart colour palette (works on both light and dark Streamlit themes)
COLORS = {
    "price": "#2196F3",
    "sma20": "#FF9800",
    "sma50": "#9C27B0",
    "volume": "#90CAF9",
    "rsi": "#26A69A",
    "rsi_ob": "rgba(239,83,80,0.15)",
    "rsi_os": "rgba(38,166,154,0.15)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_answer(raw: str) -> tuple[str, list[str]]:
    """Strip RAG markers from the answer and extract source file names."""
    sources = []
    m = re.search(r"\[SOURCES_USED:\s*([^\]]+)\]", raw)
    if m:
        sources = [s.strip() for s in m.group(1).split(";") if s.strip()]
    clean = re.sub(r"\[SOURCES_USED:[^\]]*\]", "", raw).strip()
    return clean, sources


def call_backend(query: str) -> tuple[str, list[str], list[str]]:
    """POST query to backend and return (answer, tools_used, sources)."""
    try:
        resp = requests.post(BACKEND_URL, json={"query": query}, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("answer", "No answer returned.")
        tools = data.get("tools_used", [])
        answer, sources = parse_answer(raw)
        return answer, tools, sources
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Could not reach the backend. Make sure the server is running on port 8000.",
            [], [],
        )
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out — the agent may be overloaded. Please try again.", [], []
    except Exception as exc:
        return f"⚠️ Unexpected error: {exc}", [], []


@st.cache_data(ttl=3600)
def fetch_price_data(ticker: str, days: int) -> pd.DataFrame:
    """Download OHLCV + adjusted close for a ticker. Cached for 1 hour."""
    end = date.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Build a three-panel Plotly figure: price+SMA, volume, RSI."""
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = compute_rsi(close)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("Price & Moving Averages", "Volume", "RSI (14)"),
    )

    # Panel 1 — price + SMAs
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Close",
                             line=dict(color=COLORS["price"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sma20.index, y=sma20, name="SMA 20",
                             line=dict(color=COLORS["sma20"], width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sma50.index, y=sma50, name="SMA 50",
                             line=dict(color=COLORS["sma50"], width=1, dash="dash")), row=1, col=1)

    # Panel 2 — volume
    fig.add_trace(go.Bar(x=volume.index, y=volume, name="Volume",
                         marker_color=COLORS["volume"], showlegend=False), row=2, col=1)

    # Panel 3 — RSI with overbought/oversold bands
    fig.add_hrect(y0=70, y1=100, fillcolor=COLORS["rsi_ob"], line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["rsi_os"], line_width=0, row=3, col=1)
    fig.add_hline(y=70, line=dict(color="rgba(239,83,80,0.5)", width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="rgba(38,166,154,0.5)", width=1, dash="dot"), row=3, col=1)
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI",
                             line=dict(color=COLORS["rsi"], width=1.5), showlegend=False), row=3, col=1)

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis3_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stocks AI Agent")
    st.caption("Powered by Gemini · LlamaIndex")
    st.divider()

    selected_date = st.date_input("Date", value=date.today())
    company = st.selectbox("Company", list(TICKER_MAP.keys()))
    financial_year = st.selectbox("Financial Year", ["2025-26", "2024-25", "2023-24"])
    quarter = st.selectbox("Quarter", ["None", "Q1", "Q2", "Q3", "Q4"], index=0)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_viz = st.tabs(["💬 Chat", "📊 Visualize"])

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("Ask about your stock")

    # Example question buttons
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True, key=f"ex_{question}"):
            st.session_state.messages.append({"role": "user", "content": question,
                                               "tools_used": [], "sources": []})
            if GREETING_RE.match(question):
                st.session_state.messages.append({"role": "assistant", "content": GREETING_REPLY,
                                                   "tools_used": [], "sources": []})
            else:
                full_query = (
                    f"Date: {selected_date}\n"
                    f"Company: {company}\n"
                    f"Financial Year: {financial_year}\n"
                    f"Quarter: {quarter}\n"
                    f"Question: {question}"
                )
                with st.spinner("Thinking…"):
                    answer, tools_used, sources = call_backend(full_query)
                st.session_state.messages.append({"role": "assistant", "content": answer,
                                                   "tools_used": tools_used, "sources": sources})
            st.rerun()

    st.divider()

    # Conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("tools_used"):
                badge_str = " &nbsp; ".join(
                    f"`{TOOL_LABELS.get(t, t)}`" for t in msg["tools_used"]
                )
                st.markdown(f"**Tools used:** {badge_str}")
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")

    # Pinned chat input
    st.markdown(
        "<style>.stChatInput{position:sticky;bottom:0;background:var(--background-color);}</style>",
        unsafe_allow_html=True,
    )
    user_input = st.chat_input("Ask a question about the selected stock…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input,
                                           "tools_used": [], "sources": []})

        if GREETING_RE.match(user_input):
            reply = GREETING_REPLY
            tools_used, sources = [], []
        else:
            full_query = (
                f"Date: {selected_date}\n"
                f"Company: {company}\n"
                f"Financial Year: {financial_year}\n"
                f"Quarter: {quarter}\n"
                f"Question: {user_input}"
            )
            with st.spinner("Thinking…"):
                reply, tools_used, sources = call_backend(full_query)

        st.session_state.messages.append({"role": "assistant", "content": reply,
                                           "tools_used": tools_used, "sources": sources})
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.header(f"Price Chart — {company}")

    period_col, _ = st.columns([2, 8])
    with period_col:
        selected_period = st.radio(
            "Period", list(PERIOD_DAYS.keys()), index=2, horizontal=True, label_visibility="collapsed"
        )

    ticker = TICKER_MAP[company]
    days = PERIOD_DAYS[selected_period]

    with st.spinner(f"Loading {ticker} data…"):
        df = fetch_price_data(ticker, days)

    if df.empty:
        st.warning(f"No price data available for {ticker}.")
    else:
        fig = build_chart(df, ticker)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats row
        close = df["Close"].squeeze()
        period_return = float(close.iloc[-1] / close.iloc[0] - 1)
        high_52w = float(close.max())
        low_52w = float(close.min())
        latest_price = float(close.iloc[-1])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest Close", f"₹{latest_price:,.2f}")
        m2.metric("Period Return", f"{period_return:+.2%}")
        m3.metric("Period High", f"₹{high_52w:,.2f}")
        m4.metric("Period Low", f"₹{low_52w:,.2f}")

        with st.expander("📥 Raw data table"):
            display_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            display_df.index = display_df.index.strftime("%Y-%m-%d")
            st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)

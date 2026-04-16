"""Simple Streamlit frontend for the Stocks AI Agent.

This lightweight UI lets a user pick a date, company, financial year,
and quarter and ask a natural language question. The app sends the
formatted query to the backend `/ask` endpoint and displays the
returned answer.

Note: This file is intentionally minimal. It assumes the backend
API is running locally at `http://127.0.0.1:8000`.
"""

from datetime import date
import streamlit as st
import requests


st.title("💰 Stocks AI Agent (Gemini Powered)")

st.write(
    "Welcome to the Stocks AI Agent! This application allows you to ask "
    "questions about specific stocks based on the date, company, financial "
    "year, and quarter. The agent will fetch relevant news articles and "
    "provide insights to help you make informed decisions."
)


# Layout: four compact columns for date/company/year/quarter selections
col1, col2, col3, col4 = st.columns(4)

with col1:
    
    selected_date = st.date_input(
    "Select Date",
    value=date.today()  # default = today
    )

with col2: company = st.selectbox("Select Company", ["HDFC", "RELIANCE"])

with col3: financial_year = st.selectbox("Select Financial Year", ["2025-2026"])

with col4:    quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4","None"], index=4)

# Query input field for free-text questions
query = st.text_input("Ask a question related to the stock selected:")


if st.button("Ask"):
    if query:
        # Format the query the backend expects (same template used elsewhere)
        final_query = (
            f"Date: {selected_date}\n"
            f"Company: {company}\n"
            f"Financial Year: {financial_year}\n"
            f"Quarter: {quarter}\n"
            f"Question: {query}"
        )

        # Call the local backend API and display the structured answer
        response = requests.post("http://127.0.0.1:8000/ask", json={"query": final_query})
        st.write("**Answer:**", response.json().get("answer"))
    else:
        st.warning("Please enter a question.")


# if __name__ == "__main__":
#     st.write("Finance AI Agent is running. Upload your documents and ask questions!")


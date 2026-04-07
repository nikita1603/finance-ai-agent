# streamlit run app.py
from datetime import date
import streamlit as st
import requests


st.title("💰 Stocks AI Agent (Gemini Powered)")

st.write("Welcome to the Stocks AI Agent! This application allows you to ask questions about specific stocks based on the date, company, financial year, and quarter. The agent will fetch relevant news articles and provide insights to help you make informed decisions.")


col1, col2, col3, col4 = st.columns(4)

with col1:
    
    selected_date = st.date_input(
    "Select Date",
    value=date.today()  # default = today
    )

with col2: company = st.selectbox("Select Company", ["HDFC", "RELIANCE"])

with col3: financial_year = st.selectbox("Select Financial Year", ["2025-2026"])

with col4:    quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4","None"], index=4)

# Query input
query = st.text_input("Ask a question related to the stock selected:")

if st.button("Ask"):
    if query:
        final_query = f"Date: {selected_date}\nCompany: {company}\nFinancial Year: {financial_year}\nQuarter: {quarter}\nQuestion: {query}"
        response = requests.post("http://127.0.0.1:8000/ask", json={"query": final_query})
        st.write("**Answer:**", response.json().get("answer"))
    else:
        st.warning("Please enter a question.")


# if __name__ == "__main__":
#     st.write("Finance AI Agent is running. Upload your documents and ask questions!")


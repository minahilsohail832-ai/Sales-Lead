"""
AI Sales Copilot - Live Analytics Dashboard
=============================================
Reads directly from the same Google Sheet the n8n workflow writes to.

Setup:
1. pip install streamlit gspread google-auth pandas plotly
2. Create a Google Cloud service account with access to the Sheets API,
   share your "AI Sales Copilot CRM" sheet with the service account's email
   (Viewer access is enough - this dashboard is read-only).
3. Locally: create .streamlit/secrets.toml with:

   SHEET_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"

   [gcp_service_account]
   type = "service_account"
   project_id = "..."
   private_key_id = "..."
   private_key = "..."
   client_email = "..."
   client_id = "..."
   # (paste the rest of your downloaded service account JSON fields here)

4. Run: streamlit run streamlit_dashboard.py
5. Deploy free on Streamlit Community Cloud - paste the same secrets in
   the app's Settings > Secrets panel there.
"""

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Sales Copilot Dashboard", layout="wide", page_icon="🚀")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


@st.cache_data(ttl=60)
def load_data():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    gc = gspread.authorize(creds)
    sheet = gc.open_by_url(st.secrets["SHEET_URL"]).worksheet("Leads")
    records = sheet.get_all_records()
    return pd.DataFrame(records)


st.title("🚀 AI Sales Copilot — Live Dashboard")

try:
    df = load_data()
except Exception as e:
    st.error(f"Couldn't load the sheet. Check your secrets.toml setup. Error: {e}")
    st.stop()

if df.empty:
    st.info("No leads yet. Send a test lead through the n8n webhook to see data here.")
    st.stop()

df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Leads", len(df))
col2.metric("🔥 Hot", int((df["Category"] == "Hot").sum()))
col3.metric("🌤 Warm", int((df["Category"] == "Warm").sum()))
col4.metric("Avg Score", round(df["Score"].mean(), 1) if df["Score"].notna().any() else "—")

c1, c2 = st.columns(2)

with c1:
    cat_counts = df["Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig = px.pie(
        cat_counts, names="Category", values="Count", title="Lead Distribution",
        color="Category",
        color_discrete_map={"Hot": "#e74c3c", "Warm": "#f39c12", "Cold": "#3498db"},
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    daily = df.dropna(subset=["Timestamp"]).groupby(df["Timestamp"].dt.date).size().reset_index(name="Leads")
    if not daily.empty:
        fig2 = px.bar(daily, x="Timestamp", y="Leads", title="Leads Over Time")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough timestamped leads yet for a trend chart.")

st.subheader("Recent Leads")
show_cols = [c for c in ["Timestamp", "Full Name", "Company", "Score", "Category", "Status", "Email Sent"] if c in df.columns]
st.dataframe(
    df.sort_values("Timestamp", ascending=False)[show_cols].head(25),
    use_container_width=True,
)

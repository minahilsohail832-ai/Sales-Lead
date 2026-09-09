"""
AI Sales Copilot - Lead Prospector (simple form UI)
======================================================
No more curl commands - just type a business type and location(s), click the button.
Results are also automatically saved to Airtable by the CrewAI service.
"""

import threading
import time
import uvicorn
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Lead Prospector", page_icon="🔍", layout="wide")

# --- FASTAPI BACKEND AUTOMATIC START IN BACKGROUND ---
@st.cache_resource
def start_fastapi_backend():
    def run_server():
        try:
            from crewai_service import app as fastapi_app
            uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="error")
        except Exception as e:
            print(f"Backend Server Error: {e}")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    time.sleep(3) # Waiting for FastAPI to initialize

# Launch backend once when the app starts
start_fastapi_backend()

# Set local backend URL
CREWAI_SERVICE_URL = "http://127.0.0.1:8000"
# ----------------------------------------------------

st.title("🔍 AI Sales Copilot — Lead Prospector")
st.caption("Type any business type and location(s). Results are scraped fresh and saved to Airtable automatically.")

with st.form("prospect_form"):
    col1, col2 = st.columns([2, 2])
    with col1:
        business_type = st.text_input(
            "What type of business are you looking for?",
            placeholder="e.g. dental clinic, gym, hair salon, law firm, restaurant",
        )
    with col2:
        locations_input = st.text_input(
            "Location(s) - comma-separated for multiple",
            placeholder="e.g. Austin USA, Manchester UK, Toronto Canada",
        )
    max_companies = st.slider("Max businesses per location", 1, 10, 5)
    min_reviews = st.slider(
        "Minimum reviews (higher = skip tiny/street-corner shops, target more established businesses)",
        0, 100, 15,
    )
    min_employees = st.slider(
        "Minimum employees (0 = don't check - uses 1 Apollo credit per business with a website)",
        0, 200, 0,
    )
    submitted = st.form_submit_button("🔍 Find Leads", use_container_width=True)

if submitted:
    locations = [loc.strip() for loc in locations_input.split(",") if loc.strip()]

    if not business_type.strip():
        st.warning("Please enter a business type first.")
    elif not locations:
        st.warning("Please enter at least one location.")
    else:
        with st.spinner(
            f"Searching for '{business_type}' in {len(locations)} location(s)... "
            "this takes 30-90 seconds per location, please wait"
        ):
            try:
                resp = requests.post(
                    f"{CREWAI_SERVICE_URL}/find-leads",
                    json={
                        "search_query": business_type,
                        "locations": locations,
                        "max_companies": max_companies,
                        "min_reviews": min_reviews,
                        "min_employees": min_employees,
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                leads = data.get("leads", [])

                if not leads:
                    st.info("No businesses found for that search. Try a broader business type or a different location.")
                else:
                    with_email = [l for l in leads if l.get("email")]
                    without_email = [l for l in leads if not l.get("email")]
                    fresh_opportunities = [l for l in leads if l.get("has_existing_automation") is False]

                    st.success(f"Found {len(leads)} businesses — saved to Airtable automatically.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("✅ With email (ready for outreach)", len(with_email))
                    c2.metric("📞 Phone only (needs manual follow-up)", len(without_email))
                    c3.metric("🎯 No automation detected (fresh opportunity)", len(fresh_opportunities))

                    df = pd.DataFrame(leads)
                    df["automation_signals"] = df["automation_signals"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
                    display_cols = [c for c in ["company", "email", "phone", "industry", "reviews_count", "company_size", "prospected_location",
                                                 "needs_manual_followup", "has_existing_automation", "automation_signals"] if c in df.columns]
                    st.dataframe(df[display_cols], use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't reach the inner backend service. Please wait a few seconds for background startup or check your environment keys."
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
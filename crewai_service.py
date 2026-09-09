"""
AI Sales Copilot - CrewAI Multi-Agent Microservice
====================================================
3 specialist agents (Researcher, Qualifier, Outreach) orchestrated with CrewAI,
exposed as a single FastAPI endpoint that n8n calls instead of hitting Groq directly.

Run locally:
    pip install -r requirements.txt
    export GROQ_API_KEY=your_groq_key
    uvicorn crewai_service:app --host 0.0.0.0 --port 8000

Deploy free: Railway.app or Render.com (both have free web service tiers as of 2026 -
check current limits before relying on them for a real client demo, free tiers change often).
"""

import os
import re
import json
import requests
from urllib.parse import quote
from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from pinecone import Pinecone

app = FastAPI(title="AI Sales Copilot - CrewAI Service")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")  # only used for organizations/enrich - works on free plan
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
RAG_INDEX_NAME = "sales-copilot-case-studies"

# For saving prospected leads to Airtable (more visual/demo-friendly than a plain sheet).
# Create a Personal Access Token at https://airtable.com/create/tokens with scopes
# data.records:read + data.records:write, and grant it access to your base.
AIRTABLE_PAT = os.getenv("AIRTABLE_PAT")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")          # looks like "appXXXXXXXXXXXXXX"
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Prospected Leads")


def save_leads_to_airtable(leads: list):
    """Best-effort save - if Airtable isn't configured or the call fails, prospecting
    still returns results normally, it just won't be logged automatically."""
    if not AIRTABLE_PAT or not AIRTABLE_BASE_ID or not leads:
        return
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{quote(AIRTABLE_TABLE_NAME)}"
    headers = {"Authorization": f"Bearer {AIRTABLE_PAT}", "Content-Type": "application/json"}

    # Airtable caps batch creates at 10 records per request
    for i in range(0, len(leads), 10):
        batch = leads[i:i + 10]
        records = [{
            "fields": {
                "Full Name": lead.get("full_name", ""),
                "Email": lead.get("email", ""),
                "Phone": lead.get("phone", ""),
                "Company": lead.get("company", ""),
                "Company Domain": lead.get("company_domain", ""),
                "Industry": lead.get("industry", ""),
                "Lead Source": lead.get("lead_source", ""),
                "Location": lead.get("prospected_location", ""),
                "Needs Manual Follow-Up": bool(lead.get("needs_manual_followup")),
                "Has Existing Automation": bool(lead.get("has_existing_automation")),
                "Automation Signals": ", ".join(lead.get("automation_signals", []) or []),
            }
        } for lead in batch]
        try:
            resp = requests.post(url, headers=headers, json={"records": records}, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            body = getattr(e, "response", None)
            body_text = body.text if body is not None else ""
            print(f"Airtable save failed for a batch: {e} | Response: {body_text}")

# Lazy-loaded so the ~80MB embedding model only loads into memory the first time
# it's actually needed, not on every cold start of the service.
_embed_model = None
_pinecone_index = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None and PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(RAG_INDEX_NAME)
    return _pinecone_index


def retrieve_relevant_case_study(industry: str, message: str) -> dict:
    """RAG lookup: finds the single most relevant past case study (from Pinecone)
    to use as social proof in the outreach email. Returns {} if RAG isn't configured
    or nothing relevant enough is found, so the rest of the pipeline degrades gracefully."""
    index = _get_pinecone_index()
    if not index:
        return {}
    try:
        query_text = f"{industry} {message}".strip()
        if not query_text:
            return {}
        embedding = _get_embed_model().encode(query_text).tolist()
        result = index.query(vector=embedding, top_k=1, include_metadata=True)
        if result.matches and result.matches[0].score > 0.25:  # loose relevance floor
            m = result.matches[0]
            return {
                "title": m.metadata.get("title"),
                "summary": m.metadata.get("summary"),
                "relevance_score": round(m.score, 3),
            }
        return {}
    except Exception as e:
        return {"error": str(e)}

# NOTE: llama-3.1-70b-versatile and llama-3.3-70b-versatile are both deprecated on Groq.
# Using the current recommended replacement. Verify at https://console.groq.com/docs/models
# before deploying, since Groq's lineup changes frequently.
llm = LLM(model="groq/openai/gpt-oss-120b", api_key=GROQ_API_KEY, temperature=0.3)


class LeadInput(BaseModel):
    full_name: str
    email: str
    company: str
    company_domain: str = ""   # e.g. "techcorp.com"
    company_size: str = ""
    budget: str = ""
    timeline: str = ""
    industry: str = ""
    message: str = ""
    lead_source: str = "Website Form"


def apollo_enrich_company(domain: str) -> dict:
    """Real Apollo Organization Enrichment call - the ONE Apollo endpoint that works on
    the free plan. Free tier: ~50-180 credits/month, 1 credit per lookup. Returns {} if
    no domain given or no API key set. Do not use any other Apollo endpoint (search,
    people/match, etc.) - those are blocked on the free plan."""
    if not domain or not APOLLO_API_KEY:
        return {}
    try:
        resp = requests.get(
            "https://api.apollo.io/api/v1/organizations/enrich",
            headers={"x-api-key": APOLLO_API_KEY, "Content-Type": "application/json"},
            params={"domain": domain},
            timeout=10,
        )
        resp.raise_for_status()
        org = resp.json().get("organization", {}) or {}
        return {
            "name": org.get("name"),
            "industry": org.get("industry"),
            "estimated_num_employees": org.get("estimated_num_employees"),
            "founded_year": org.get("founded_year"),
            "latest_funding_stage": org.get("latest_funding_stage"),
            "short_description": org.get("short_description"),
        }
    except Exception as e:
        # Don't let enrichment failures break lead scoring - just proceed without it.
        return {"error": str(e)}



def build_crew(lead: LeadInput, enrichment: dict, case_study: dict):
    researcher = Agent(
        role="Sales Researcher",
        goal="Turn raw company data into a short, sales-relevant brief",
        backstory=(
            "You are an SDR researcher. You take whatever company data is available "
            "(enrichment API data or just what the lead typed) and summarize anything "
            "a sales rep could actually use in an opening line - size, industry, funding, "
            "recent signals. If there's nothing useful, say so plainly instead of inventing details."
        ),
        llm=llm,
        verbose=False,
    )

    qualifier = Agent(
        role="Lead Qualifier",
        goal="Score leads 0-100 on how likely they are to close soon",
        backstory=(
            "You are a strict, consistent B2B sales qualification analyst. You score based "
            "on budget signals, timeline/urgency, company fit, and intent signals in the "
            "lead's own message. You never inflate scores to be polite."
        ),
        llm=llm,
        verbose=False,
    )

    outreach = Agent(
        role="Outreach Copywriter",
        goal="Write a short, specific, non-salesy first-touch email",
        backstory=(
            "You write first-touch sales emails under 120 words. You reference something "
            "concrete from the research brief when there is one, keep tone matched to how "
            "qualified the lead is, and never sound like a template."
        ),
        llm=llm,
        verbose=False,
    )

    research_task = Task(
        description=f"""Summarize this company for a sales rep in 2-3 sentences.
Enrichment data (may be empty): {json.dumps(enrichment)}
Company name as given by lead: {lead.company}
Industry as given by lead: {lead.industry}
If enrichment data is empty or has an 'error' key, say "No enrichment data available"
and work only from what the lead typed. Do not invent facts.""",
        expected_output="A 2-3 sentence company brief in plain text.",
        agent=researcher,
    )

    qualify_task = Task(
        description=f"""Score this lead 0-100 using: budget signals (30pts), timeline/urgency (25pts),
company fit (20pts), intent signals in their message (25pts). Hot=70-100, Warm=40-69, Cold=0-39.

Name: {lead.full_name}
Company: {lead.company}
Company size: {lead.company_size}
Budget: {lead.budget}
Timeline: {lead.timeline}
Industry: {lead.industry}
Message: {lead.message}
Lead source: {lead.lead_source}

Use the researcher's company brief as extra context if it adds signal.
Return ONLY valid JSON, nothing else: {{"score": <int>, "category": "Hot"|"Warm"|"Cold", "reasoning": "<1-2 sentences>"}}""",
        expected_output='A JSON object: {"score": int, "category": string, "reasoning": string}',
        agent=qualifier,
        context=[research_task],
    )

    outreach_task = Task(
        description=f"""Write a first-touch email body (under 120 words, no subject line) to
{lead.full_name} at {lead.company}. Reference something specific from the company brief if it's
genuinely useful - don't force it if the brief has no real data. Match tone to the qualifier's
category: Hot = direct call-to-action, Warm = helpful + soft CTA, Cold = low-pressure nurture.

Relevant past case study (may be empty if nothing matched well): {json.dumps(case_study)}
If it's genuinely relevant to this lead's industry or need, you may briefly mention it as social
proof (e.g. "we helped a similar {lead.industry or 'business'} team with..."). Do not force it or
fabricate details if the case study is empty or a weak match.

Sign off as "Minahil". Return ONLY the email body text.""",
        expected_output="Plain text email body, no JSON, no subject line.",
        agent=outreach,
        context=[research_task, qualify_task],
    )

    return Crew(
        agents=[researcher, qualifier, outreach],
        tasks=[research_task, qualify_task, outreach_task],
        process=Process.sequential,
        verbose=False,
    ), research_task, qualify_task, outreach_task


APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")


class ProspectCriteria(BaseModel):
    search_query: str = "real estate agency"   # type of business, e.g. "dental clinic", "law firm"
    locations: list = ["Lahore, Pakistan"]     # e.g. ["Austin, Texas, USA", "Manchester, UK", "Toronto, Canada"]
    max_companies: int = 5                     # keep small - this is PER LOCATION, so 3 locations x 5 = 15 total runs
    min_reviews: int = 15                      # skip tiny/unestablished businesses (proxy for "can afford automation")
    require_website: bool = False              # True = skip businesses with no website at all (very small shops rarely have one)
    min_employees: int = 0                     # 0 = no check. Uses Apollo organizations/enrich (1 credit per business with a website)


def apify_find_companies(search_query: str, location: str, max_results: int) -> list:
    """Reuses the same Apify Google Maps Scraper actor from your B2B Lead Gen Engine.
    Apify's free tier includes monthly platform credit, enough for light prospecting runs.
    Fetches 3x the requested amount so post-filtering (by reviews/website) still leaves
    close to max_results businesses."""
    if not APIFY_API_TOKEN:
        return []
    try:
        resp = requests.post(
            "https://api.apify.com/v2/acts/compass~crawler-google-places/run-sync-get-dataset-items",
            params={"token": APIFY_API_TOKEN},
            json={
                "searchStringsArray": [search_query],
                "locationQuery": location,
                "maxCrawledPlacesPerSearch": max_results * 3,
                "language": "en",
                "maxImages": 0,
                "maxReviews": 0,
            },
            timeout=180,  # Google Maps scraping runs take a while - don't set this too low
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


# Known automation/chatbot/booking tool signatures - if any of these show up in a
# business's website HTML, they likely already have some automation in place.
AUTOMATION_SIGNATURES = {
    "Live chat / chatbot": ["intercom", "drift.com", "driftt.com", "tawk.to", "crisp.chat",
                             "tidio.co", "livechatinc.com", "chatbase.co", "landbot.io",
                             "chatling.ai", "hs-scripts.com"],
    "Voice AI agent": ["vapi.ai", "retellai.com", "bland.ai", "synthflow.ai", "elevenlabs.io"],
    "WhatsApp automation": ["wati.io", "interakt.shop", "gupshup.io", "wa.me/message"],
    "Scheduling automation": ["calendly.com", "acuityscheduling.com", "setmore.com"],
    "Marketing chatbot": ["manychat.com", "voiceflow.com"],
}


def detect_existing_automation(html: str) -> dict:
    """Scans the same page HTML already fetched for email scraping - no extra
    requests needed. Returns which automation categories (if any) were detected."""
    if not html:
        return {"has_existing_automation": False, "automation_signals": []}
    html_lower = html.lower()
    found = [category for category, sigs in AUTOMATION_SIGNATURES.items() if any(s in html_lower for s in sigs)]
    return {"has_existing_automation": bool(found), "automation_signals": found}


def scrape_email_from_website(website_url: str) -> dict:
    """No signup, no API key needed: fetches the business's own website and looks for a
    contact email directly in the page HTML (mailto: links or plain-text addresses),
    and separately scans the same HTML for signs of existing automation tools."""
    if not website_url:
        return {}
    try:
        resp = requests.get(website_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text

        automation = detect_existing_automation(html)

        mailto_matches = re.findall(r'mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', html)
        plain_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', html)
        candidates = mailto_matches or plain_matches
        candidates = [c for c in candidates if not any(x in c.lower() for x in [".png", ".jpg", ".gif", "example.com", "sentry", "wixpress"])]
        email = candidates[0] if candidates else None

        return {
            "full_name": "Team",
            "email": email,
            "title": "",
            "has_existing_automation": automation["has_existing_automation"],
            "automation_signals": automation["automation_signals"],
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/find-leads")
def find_leads(criteria: ProspectCriteria):
    """Outbound prospecting: Apify Google Maps Scraper finds businesses matching the search
    query across one or more locations, then their own website is scraped directly for a
    contact email - no third-party signup needed for this step. Returns leads in the same
    shape /score-lead expects, so n8n can loop over this and feed each one in.

    Businesses with no website/email aren't dropped - they're still returned with
    needs_manual_followup=True and their phone number, so no lead is silently lost."""
    found_leads = []
    for location in criteria.locations:
        companies = apify_find_companies(criteria.search_query, location, criteria.max_companies)

        # Filter out very small/unestablished businesses - reviewsCount is the best
        # available proxy for business scale since Google Maps doesn't give employee counts.
        filtered = [
            biz for biz in companies
            if (biz.get("reviewsCount") or 0) >= criteria.min_reviews
            and (not criteria.require_website or bool(biz.get("website")))
        ]

        for biz in filtered[: criteria.max_companies]:
            website = biz.get("website") or ""
            domain = website.replace("https://", "").replace("http://", "").split("/")[0] if website else ""
            contact = scrape_email_from_website(website) if website else {}
            email = contact.get("email")

            employee_count = None
            if domain and criteria.min_employees > 0:
                enrichment = apollo_enrich_company(domain)
                employee_count = enrichment.get("estimated_num_employees")
                # Only reject if we actually got a number and it's below the floor -
                # don't reject businesses just because enrichment had no data at all.
                if isinstance(employee_count, int) and employee_count < criteria.min_employees:
                    continue

            found_leads.append({
                "full_name": contact.get("full_name", "Team"),
                "email": email or "",
                "phone": biz.get("phone", ""),
                "company": biz.get("title", ""),
                "company_domain": domain,
                "company_size": str(employee_count) if employee_count is not None else "",
                "budget": "",
                "timeline": "",
                "industry": biz.get("categoryName", criteria.search_query),
                "reviews_count": biz.get("reviewsCount", 0),
                "message": f"Prospected lead via Google Maps ({location}) - {biz.get('categoryName', criteria.search_query)} business at {biz.get('address', '')}. No inbound message; score on firmographic fit only.",
                "lead_source": "Apify Prospecting + Website Scrape",
                "needs_manual_followup": not bool(email),
                "prospected_location": location,
                "has_existing_automation": contact.get("has_existing_automation", False) if website else None,
                "automation_signals": contact.get("automation_signals", []) if website else [],
            })
    save_leads_to_airtable(found_leads)
    return {"leads": found_leads, "count": len(found_leads)}


@app.post("/score-lead")
def score_lead(lead: LeadInput):
    enrichment = apollo_enrich_company(lead.company_domain)
    case_study = retrieve_relevant_case_study(lead.industry, lead.message)
    crew, research_task, qualify_task, outreach_task = build_crew(lead, enrichment, case_study)
    crew.kickoff()

    company_brief = research_task.output.raw if research_task.output else ""
    qualifier_raw = qualify_task.output.raw if qualify_task.output else "{}"
    outreach_email = outreach_task.output.raw if outreach_task.output else ""

    try:
        cleaned = re.sub(r"```json|```", "", qualifier_raw).strip()
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {"score": 0, "category": "Cold", "reasoning": "Failed to parse qualifier output - manual review needed."}

    return {
        "email": lead.email,
        "full_name": lead.full_name,
        "company": lead.company,
        "company_brief": company_brief,
        "score": parsed.get("score", 0),
        "category": parsed.get("category", "Cold"),
        "reasoning": parsed.get("reasoning", ""),
        "outreach_email": outreach_email,
        "enrichment_raw": enrichment,
        "matched_case_study": case_study,
    }


@app.get("/health")
def health():
    return {"status": "ok"}